"""Black out SSNs, ITINs and EINs in a document before it is sent to a model.

TAX-4858 (option A). The redaction is real: PyMuPDF removes the glyphs under
each box and blanks the image pixels there, so neither the text layer nor the
rendered page carries the number afterwards.

Only the digits before the last four are removed — the form IRS truncated
TINs take. The model still reads `XXX-XX-6789`, so the extraction keeps an
identifier field to fill, and the full number is restored locally from the
document's own text layer or OCR by those last four digits.

Where the numbers are found:
  * a page with a usable text layer — its words (digital PDFs and PDFs that
    already carry an OCR text layer);
  * a page without one (a scan or a photo) — OCR words from the configured
    backend: RapidOCR by default, Paddle under the OCR_BACKEND=paddle
    rollback. The default path never imports or loads Paddle (TAX-6876).

Before anything is returned the result is checked: every page's text layer
is searched again, and every page that needed OCR is OCRed again. If an
identifier is still readable anywhere, the redaction fails — a partial result
is never returned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

import fitz
import structlog

log = structlog.get_logger()

# 123-45-6789 / 123 45 6789: an SSN or ITIN. ZIP+4 (12345-6789) is 5-4.
_SSN = re.compile(r"(?<![\d-])(\d{3})([- ])(\d{2})\2(\d{4})(?![\d-])")
# 12-3456789: an EIN.
_EIN = re.compile(r"(?<![\d-])(\d{2})-(\d{7})(?![\d-])")
# Nine bare digits shortly after a label naming an identifier. Unlabeled runs
# (routing and account numbers) are not identifiers and stay readable.
_LABELED_BARE = re.compile(
    r"\b(?:SSN|TIN|EIN|FEIN|ITIN|social\s+security(?:\s+number)?|"
    r"(?:taxpayer|employer|payer'?s?|recipient'?s?|borrower'?s?|student'?s?)\s+"
    r"(?:identification|id)(?:\s+(?:number|no\.?))?|identification\s+number)\b"
    r"[^\d\n]{0,40}(\d{9})(?!\d)",
    re.IGNORECASE,
)
_SSN_LABEL = re.compile(r"\b(SSN|ITIN|social\s+security)\b", re.IGNORECASE)
_EIN_LABEL = re.compile(r"\b(EIN|FEIN|employer)\b", re.IGNORECASE)

# A text layer counts as usable when it has some length and real words;
# scans often carry garbage glyphs whose "words" average under 3 characters
# (same heuristic as pdf_service.ocr_pages).
_MIN_TEXT_CHARS = 50
_MIN_AVG_TOKEN_LEN = 3.0
# A page whose images cover this share of it may show an identifier that its
# text layer does not carry (a scanner app's text layer over a photo), so it is
# OCRed too.
_IMAGE_COVERAGE_FOR_OCR = 0.5


class RedactionIncompleteError(Exception):
    """An identifier is still readable after redaction."""


@dataclass(frozen=True)
class Word:
    text: str
    rect: fitz.Rect
    # "word", or "line" for a whole recognized OCR line (searched on its own).
    kind: str = "word"


@dataclass(frozen=True)
class Found:
    page: int
    kind: str  # "ssn" | "ein" | "tin"
    last4: str
    source: str  # "text_layer" | "ocr"
    # The boxes to black out: everything of the identifier but its last four digits.
    rects: tuple[fitz.Rect, ...]


# OCR confusions read as digits inside numeric-looking tokens (TAX-6876).
_CONFUSIONS = str.maketrans({
    "I": "1", "l": "1", "|": "1", "O": "0", "o": "0", "D": "0", "S": "5", "s": "5",
    "B": "8", "Z": "2", "G": "6", "q": "9",
})
_CONFUSABLE = set("Il|OoDSsBZGq")
# Number shapes that are not identifiers; masked before digit runs are read.
_NOT_IDENTIFIER = re.compile(
    r"\$?\d{1,3}(?:,\d{3})+(?:\.\d{2})?(?!\d)"   # 1,234 / 75,250.00
    r"|\$?\d+\.\d{2}(?!\d)"                       # 75250.00
    r"|(?<!\d)\d{1,2}/\d{1,2}/\d{2,4}(?!\d)"        # dates
    r"|(?<!\d)\d{5}-\d{4}(?!\d)"                    # ZIP+4
    r"|\(\d{3}\)\s?\d{3}[-.]\d{4}(?!\d)"             # (555) 123-4567
    r"|(?<!\d)\d{3}[-.]\d{3}[-.]\d{4}(?!\d)"          # 555-123-4567
)
# A digit run: digits joined by at most two separator characters at a time.
_DIGIT_RUN = re.compile(r"\d(?:[-\u2010-\u2015./_:\u00b7\u2022 ]{0,2}\d)*")
# A run of this many digits that is not an identifier may be a misread one.
_SUSPECT_DIGITS = 7


def normalize_confusions(text: str) -> str:
    """Map OCR letter/digit confusions to digits inside numeric-looking tokens.

    Character-for-character, so positions are kept. A token (between spaces)
    is normalized when it holds at least one digit and digits plus confusable
    letters make up at least half of its letters and digits: "I23-45-678S"
    becomes "123-45-6785"; "ID", "SSN" and "Box" stay words.
    """
    out = []
    for token in re.split(r"(\s+)", text):
        alnum = [c for c in token if c.isalnum() or c == "|"]
        numeric = sum(c.isdigit() or c in _CONFUSABLE for c in alnum)
        if any(c.isdigit() for c in token) and alnum and numeric * 2 >= len(alnum):
            token = token.translate(_CONFUSIONS)
        out.append(token)
    return "".join(out)


def _digit_runs(text: str) -> Iterable[tuple[int, int, str]]:
    """(start, end, digits) of digit runs, non-identifier number shapes masked."""
    masked = _NOT_IDENTIFIER.sub(lambda m: "x" * len(m.group(0)), text)
    for m in _DIGIT_RUN.finditer(masked):
        yield m.start(), m.end(), re.sub(r"\D", "", m.group(0))


def _run_kind(text: str, start: int, end: int) -> str:
    groups = [len(g) for g in re.findall(r"\d+", text[start:end])]
    if groups == [3, 2, 4]:
        return "ssn"
    if groups == [2, 7]:
        return "ein"
    label = text[max(0, start - 60):start]
    return "ssn" if _SSN_LABEL.search(label) else "ein" if _EIN_LABEL.search(label) else "tin"


def suspect_runs(text: str) -> list[str]:
    """Digit runs of _SUSPECT_DIGITS+ digits that are not identifier-shaped."""
    text = normalize_confusions(text)
    return [
        digits for _, _, digits in _digit_runs(text)
        if len(digits) >= _SUSPECT_DIGITS and len(digits) != 9
    ]


def _matches(line: str, ocr: bool = False) -> Iterable[tuple[int, int, str, str]]:
    """(start, end, kind, digits) for every identifier in one line of text.

    OCR text (``ocr``) is matched after confusion normalization, and any run of
    exactly nine digits, whatever its separators, is an identifier: a misread
    "I23-45-678S" or "123 45  6789" is still an SSN.
    """
    if ocr:
        normalized = normalize_confusions(line)
        for start, end, digits in _digit_runs(normalized):
            if len(digits) == 9:
                yield start, end, _run_kind(normalized, start, end), digits
        return
    for m in _SSN.finditer(line):
        yield m.start(), m.end(), "ssn", m.group(1) + m.group(3) + m.group(4)
    for m in _EIN.finditer(line):
        yield m.start(), m.end(), "ein", m.group(1) + m.group(2)
    for m in _LABELED_BARE.finditer(line):
        # A TIN box may hold either kind; only the label can tell.
        label = m.group(0)
        kind = "ssn" if _SSN_LABEL.search(label) else "ein" if _EIN_LABEL.search(label) else "tin"
        yield m.start(1), m.end(1), kind, m.group(1)


def _lines(words: list[Word]) -> list[list[Word]]:
    """Group words into reading lines by vertical overlap, left to right."""
    lines: list[list[Word]] = []
    for w in sorted(words, key=lambda w: (round(w.rect.y0, 0), w.rect.x0)):
        mid = (w.rect.y0 + w.rect.y1) / 2
        for line in lines:
            ref = line[0].rect
            if ref.y0 - 1 <= mid <= ref.y1 + 1:
                line.append(w)
                break
        else:
            lines.append([w])
    return [sorted(line, key=lambda w: w.rect.x0) for line in lines]


def _prefix_end(text: str, start: int, end: int) -> int:
    """Index in `text` where the identifier's last four digits begin."""
    seen = 0
    for i in range(end - 1, start - 1, -1):
        if text[i].isdigit():
            seen += 1
            if seen == 4:
                return i
    return start


def _glyph_columns(page: fitz.Page, rect: fitz.Rect) -> list[tuple[float, float]]:
    """Left/right page x of each inked glyph in a line box, by vertical ink projection."""
    zoom = 4.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, colorspace=fitz.csGRAY)
    width, height, samples = pix.width, pix.height, pix.samples
    inked = [any(samples[y * width + x] < 128 for y in range(height)) for x in range(width)]
    glyphs: list[tuple[float, float]] = []
    start = None
    for x, ink in enumerate(inked + [False]):
        if ink and start is None:
            start = x
        elif not ink and start is not None:
            glyphs.append((rect.x0 + start / zoom, rect.x0 + x / zoom))
            start = None
    return glyphs


def _part_rect(word: Word, lo: int, hi: int, page: fitz.Page | None, source: str) -> fitz.Rect:
    """The box of characters [lo, hi) of one word.

    From the text layer the characters are located exactly. PaddleOCR returns a
    whole text line as one "word", so its characters are located on the page
    image: glyphs are separated by the ink gaps between them and matched to the
    line's non-space characters in order. When the counts disagree the line is
    split by approximate glyph widths instead.
    """
    if lo <= 0 and hi >= len(word.text):
        return fitz.Rect(word.rect)
    if page is not None and source == "text_layer":
        hits = page.search_for(word.text[lo:hi], clip=word.rect + (-1, -1, 1, 1))
        if hits:
            ordered = sorted(hits, key=lambda r: r.x0)
            return ordered[0] if lo == 0 else ordered[-1]
    if page is not None and source == "ocr":
        glyphs = _glyph_columns(page, word.rect)
        visible = [i for i, ch in enumerate(word.text) if not ch.isspace()]
        if len(glyphs) == len(visible):
            wanted = [n for n, i in enumerate(visible) if lo <= i < hi]
            if wanted:
                x0 = glyphs[wanted[0]][0]
                x1 = glyphs[wanted[-1]][1]
                # Up to halfway into the gap on each side, never onto a neighbour.
                if wanted[0] > 0:
                    x0 = (glyphs[wanted[0] - 1][1] + x0) / 2
                if wanted[-1] + 1 < len(glyphs):
                    x1 = (x1 + glyphs[wanted[-1] + 1][0]) / 2
                return fitz.Rect(x0, word.rect.y0, x1, word.rect.y1)

    # Approximate glyph widths (Helvetica-like): narrow separators and
    # lowercase, wide capitals.
    def weight(ch: str) -> float:
        if ch in "-. /,()'":
            return 0.3
        if ch.islower():
            return 0.5
        if ch.isupper():
            return 0.68
        return 0.56

    weights = [weight(ch) for ch in word.text]
    unit = word.rect.width / max(sum(weights), 1e-6)
    x0 = word.rect.x0 + unit * sum(weights[:lo])
    x1 = word.rect.x0 + unit * sum(weights[:hi])
    # Outward on the redacted side only: the kept last four digits stay whole.
    left = x0 - unit * 0.3
    right = x1 + unit * 0.3 if hi >= len(word.text) else x1 - unit * 0.1
    return fitz.Rect(left, word.rect.y0, right, word.rect.y1)


def find_identifiers(
    page_index: int, words: list[Word], source: str, page: fitz.Page | None = None
) -> list[Found]:
    """Locate identifiers in a page's words, spanning word boundaries.

    `page` lets the boxes to redact be located exactly (text layer) or on the
    page image (OCR lines).
    """
    found: list[Found] = []
    groups = _lines([w for w in words if w.kind == "word"])
    # A recognized line is searched on its own, never joined to its words.
    groups += [[w] for w in words if w.kind == "line"]
    for line in groups:
        text = ""
        spans: list[tuple[int, int, Word]] = []
        for w in line:
            if text:
                text += " "
            start = len(text)
            text += w.text
            spans.append((start, len(text), w))
        ocr = source == "ocr"
        for start, end, kind, digits in _matches(text, ocr):
            cut = _prefix_end(normalize_confusions(text) if ocr else text, start, end)
            rects = []
            for s, e, w in spans:
                lo, hi = max(start, s), min(cut, e)
                if lo < hi:
                    rects.append(_part_rect(w, lo - s, hi - s, page, source))
            if rects:
                found.append(Found(page_index, kind, digits[-4:], source, tuple(rects)))
    return found


def text_layer_words(page: fitz.Page) -> list[Word]:
    return [Word(w[4], fitz.Rect(w[:4])) for w in page.get_text("words")]


def has_usable_text_layer(page: fitz.Page) -> bool:
    text = page.get_text().strip()
    if len(text) < _MIN_TEXT_CHARS:
        return False
    tokens = text.split()
    return bool(tokens) and sum(len(t) for t in tokens) / len(tokens) >= _MIN_AVG_TOKEN_LEN


def needs_ocr(page: fitz.Page) -> bool:
    if not has_usable_text_layer(page):
        return True
    area = abs(page.rect)
    if area == 0:
        return False
    covered = 0.0
    for info in page.get_image_info():
        covered += abs(fitz.Rect(info["bbox"]) & page.rect)
    return covered / area >= _IMAGE_COVERAGE_FOR_OCR


OcrWords = Callable[[bytes, list[int]], dict[int, list[Word]]]
"""(pdf bytes, page indices) → words per page, boxes in unrotated page points."""


def ocr_words(pdf_bytes: bytes, pages: list[int]) -> dict[int, list[Word]]:
    """Words of the page images, read by the backend OCR_BACKEND selects."""
    from app.config import settings

    if settings.ocr_backend == "paddle":
        return paddle_ocr_words(pdf_bytes, pages)
    return rapid_ocr_words(pdf_bytes, pages)


def rapid_ocr_words(pdf_bytes: bytes, pages: list[int]) -> dict[int, list[Word]]:
    """RapidOCR words through the shared OCR workers; no search layer is built."""
    from app.services import rapid_ocr

    result = rapid_ocr.ocr_pages(
        pdf_bytes, pages, force_ocr=True, with_pdf=False, low_confidence=True
    )
    return _words(result, pdf_bytes)


def paddle_ocr_words(pdf_bytes: bytes, pages: list[int]) -> dict[int, list[Word]]:
    from app.services import pdf_service

    result = pdf_service.ocr_pages(pdf_bytes, pages, force_ocr=True, backend="paddle")
    return _words(result, pdf_bytes)


# Every page OCR read is checked for print that OCR did not turn into words.
# A false refusal costs a manual review; a false pass leaks an identifier, so
# any single test below refuses.
#
# Marks. The page is rendered at the OCR DPI in grayscale and compared with
# its own background, a 31x31 median filter (robust to print, which never fills
# half a window, and to paper tone and shading). A pixel is a mark when it is
# more than _MARK_CONTRAST levels darker. 6 levels catches light-gray print up
# to gray 246 on white at 4 pt and stays above sigma-4 sensor noise. Solid
# black areas (redaction boxes, black bars) carry no readable print and are
# removed first: their interior survives a 9x9 opening, strokes do not.
#
# Components. Connected marks with a side of at least 5 px and 6 px of area. A
# *speck* (both sides <= 10 px, area <= 80) is dust or a noise cluster. *Line
# art* — rules (thinner than 4 px, longer than 30 px) and box grids (longer
# than 150 px, under 15% filled) — is form structure, not print.
#
# Coverage. A component is read when at least half of its pixels fall inside
# OCR word boxes widened by max(4 px, 30% of the box height). Of the components
# that are not read and not line art, the page refuses when:
# (a) one is a merged word or line: at least 2.5x longer than wide, either
#     orientation, no size cap (blur merges a whole SSN into one blob);
# (b) two glyph-sized ones (side 5-150 px) sit next to each other on one line,
#     horizontally or vertically (rotated print): centers within half a glyph
#     across the line, gap at most 1.5 glyph sizes along it;
# (c) together, outside specks, they cover more than 0.01% of the page (~440
#     px on Letter at 216 DPI).
#
# Measured on 216-DPI renders of 200-DPI scans: "123-45-6789" at 4-10 pt and
# grays 0-246, blurred (sigma 1-3) or rotated 90/270 degrees, refuses when OCR
# missed it; a blank page, sigma-4 noise and up to 200 dust specks pass.
_MARK_CONTRAST = 6
_BACKGROUND_WINDOW = 31
_SOLID_BLACK, _SOLID_KERNEL = 40, 9
_MIN_SIDE, _MIN_AREA = 5, 6
_GLYPH_MAX_SIDE = 150
_SPECK_MAX_SIDE, _SPECK_MAX_AREA = 10, 80
_ID_MIN_RUN = 4
_ID_MIN_DIGITS = 9
_DIGIT_MIN_SIZE, _DIGIT_MAX_SIZE = 10, 60
_DIGIT_MIN_CONTRAST = 18
_DARK_REGION = 200
_SHAPE_TONE = 0.35
_SHAPE_RING = 0.25
_READ_FRACTION = 0.5


def _identifier_run(components, along: int) -> bool:
    """A run of print along one line that could hold an identifier.

    along=0 is a horizontal line, 1 a vertical one (rotated print). Marks that
    are flat across the line (at most 4 px: dot leaders, dashes, rules) are not
    print here. Neighbours: centers within half a glyph size across the line,
    gap at most 1.5 glyph sizes along it. A mark holds about
    length / (0.6 x size) digits, so a merged blob counts as several. True when
    a run has _ID_MIN_RUN marks or _ID_MIN_DIGITS digit-equivalents.
    """
    across = 1 - along
    marks = sorted(
        (c for c in components if 4 < c[2 + across] <= _GLYPH_MAX_SIDE),
        key=lambda c: c[along],
    )
    parent = list(range(len(marks)))
    count = [1] * len(marks)
    # Thin bars (under 7 px across) are not merged digits, however long.
    digits = [
        max(1, round(c[2 + along] / (0.6 * c[2 + across]))) if c[2 + across] >= 7 else 1
        for c in marks
    ]
    if any(d >= _ID_MIN_DIGITS for d in digits):
        return True

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for index, first in enumerate(marks):
        for offset in range(index + 1, len(marks)):
            second = marks[offset]
            gap = second[along] - (first[along] + first[2 + along])
            if gap > 1.5 * _GLYPH_MAX_SIDE:
                break  # sorted along the line: every later mark is farther
            size = max(first[2 + across], second[2 + across])
            if gap > 1.5 * size:
                continue
            first_center = first[across] + first[2 + across] / 2
            second_center = second[across] + second[2 + across] / 2
            if abs(first_center - second_center) >= 0.5 * size:
                continue
            a, b = root(index), root(offset)
            if a == b:
                continue
            parent[b] = a
            count[a] += count[b]
            digits[a] += digits[b]
            if count[a] >= _ID_MIN_RUN or digits[a] >= _ID_MIN_DIGITS:
                return True
    return False


def _part_of_larger_shape(gray, background, labels, i, x, y, w, h) -> bool:
    """A mark that continues, at the same tone, into unmarked surroundings.

    The 31-px median background absorbs most of a large filled shape (a logo,
    a shaded box), leaving only its corners as marks. Such a corner has the
    tone of a good part (_SHAPE_RING) of the pixels around it; print — dark
    on paper, dark inside a shaded box, light on a dark fill — differs from
    nearly all of its surroundings.
    """
    import cv2
    import numpy as np

    pad = 3
    y0, y1 = max(0, y - pad), min(gray.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(gray.shape[1], x + w + pad)
    own = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
    ring = (cv2.dilate(own, np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)) > 0) & (own == 0)
    region = gray[y0:y1, x0:x1].astype(np.int16)
    if not ring.any():
        return False
    mark = float(np.median(region[own > 0]))
    base = float(np.median(background[y0:y1, x0:x1][own > 0]))
    # Print stands out from all its surroundings; a shape's corner continues,
    # at its own tone, into the unmarked rest of the shape.
    if abs(mark - base) < 12:
        return False  # faint print; a shape this light never counts as digits
    tolerance = max(4.0, _SHAPE_TONE * abs(mark - base))
    same = np.abs(region[ring] - mark) < tolerance
    return float(same.mean()) >= _SHAPE_RING


def _digit_budget(unread) -> int:
    """Digit-equivalents in digit-sized unread marks, upright or rotated.

    A mark is digit-sized when its height across the line is 10-60 px (a digit
    of about 5-28 pt at 216 DPI; dust specks stay under 10 px) and it is printed, not noise: at its darkest it
    differs from the background by at least _DIGIT_MIN_CONTRAST levels (gray
    237 on white; sensor noise stays below). A mark holds about
    length / (0.6 x height) digits. The larger of the upright and the rotated reading counts.
    """
    def digits(c, along):
        across = 1 - along
        length, size = c[2 + along], c[2 + across]
        if not _DIGIT_MIN_SIZE <= size <= _DIGIT_MAX_SIZE or c[5] < _DIGIT_MIN_CONTRAST:
            return 0
        # A small round filled blob is dust: a single digit is taller than
        # wide, and merged digits are larger.
        if (max(c[2], c[3]) <= 14 and 0.75 <= c[2] / c[3] <= 1.33
                and c[4] >= 0.6 * c[2] * c[3]):
            return 0
        return 1 if length <= 0.9 * size else max(1, round(length / (0.6 * size)))

    return max(sum(digits(c, 0) for c in unread), sum(digits(c, 1) for c in unread))


def _is_box_outline(component) -> bool:
    """An empty rectangular frame: nearly all its pixels lie on its bounding box.

    A checkbox or field frame qualifies; a glyph does not — even "0" or "O"
    touches its box only at four points, and a filled square is not hollow.
    """
    import numpy as np

    h, w = component.shape
    if min(w, h) < 12 or not 0.7 <= w / h <= 1.43 and min(w, h) < 24:
        return False
    border = np.zeros_like(component)
    band = max(2, round(min(w, h) * 0.12))
    border[:band, :] = border[-band:, :] = border[:, :band] = border[:, -band:] = True
    pixels = component.sum()
    on_border = (component & border).sum()
    hollow = pixels < 0.6 * w * h
    return bool(hollow and on_border >= 0.9 * pixels)


def unread_text(
    pdf_bytes: bytes, page_index: int, dpi: float, words: list[Word] | None = None
) -> dict:
    """Print on the page that OCR word boxes do not cover; see the notes above.

    ``text_like`` is True when the page must be refused.
    """
    import cv2
    import numpy as np

    from app.services import rapid_ocr
    from app.services.pdf_service import _open_pdf

    zoom = dpi / 72
    with rapid_ocr.FITZ_LOCK:
        with _open_pdf(pdf_bytes) as doc:
            page = doc[page_index]
            pix = page.get_pixmap(
                matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False
            )
            gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
            gray = gray.copy()
            sx, sy = pix.width / page.rect.width, pix.height / page.rect.height
    background = cv2.medianBlur(gray, _BACKGROUND_WINDOW)
    contrast = np.abs(background.astype(np.int16) - gray)
    marks = ((background.astype(np.int16) - gray) > _MARK_CONTRAST).astype(np.uint8)
    kernel = np.ones((_SOLID_KERNEL, _SOLID_KERNEL), np.uint8)
    solid = cv2.morphologyEx((gray <= _SOLID_BLACK).astype(np.uint8), cv2.MORPH_OPEN, kernel)
    marks[cv2.dilate(solid, np.ones((7, 7), np.uint8)) > 0] = 0
    # Inverted print: inside dark regions (black or gray fills, a redaction box
    # included) light marks are print too — white on black, white on gray.
    # The region is eroded so the paper just outside a dark area is not print.
    dark = cv2.erode((background < _DARK_REGION).astype(np.uint8), np.ones((11, 11), np.uint8))
    light = (gray.astype(np.int16) - background) > _MARK_CONTRAST
    marks[(dark > 0) & light] = 1
    # Faint print breaks into stroke fragments; a 3x3 closing rejoins them.
    marks = cv2.morphologyEx(marks, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    read = np.zeros_like(marks)
    for word in words or []:
        height = word.rect.height * sy
        margin = max(4.0, 0.3 * height)
        x0 = int(max(0, word.rect.x0 * sx - margin))
        y0 = int(max(0, word.rect.y0 * sy - margin))
        x1 = int(min(read.shape[1], word.rect.x1 * sx + margin + 1))
        y1 = int(min(read.shape[0], word.rect.y1 * sy + margin + 1))
        read[y0:y1, x0:x1] = 1

    count, labels, stats, _ = cv2.connectedComponentsWithStats(marks, 8)
    height_px, width_px = marks.shape
    if count > 1:
        inside = np.bincount(labels[read > 0].ravel(), minlength=count)
    else:
        inside = np.zeros(1, dtype=np.int64)
    unread = []
    for i in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[i][:5])
        if max(w, h) < _MIN_SIDE or area < _MIN_AREA:
            continue
        if min(w, h) <= 4 and max(w, h) >= 8 * min(w, h):
            continue  # a rule
        if max(w, h) > _GLYPH_MAX_SIDE and area < 0.15 * w * h:
            continue  # a box grid
        if _is_box_outline(labels[y:y + h, x:x + w] == i):
            continue  # a checkbox or field frame
        if max(w, h) <= 15 and (x == 0 or y == 0 or x + w >= width_px or y + h >= height_px):
            continue  # a small mark cut by the page edge: a scan border artifact
        if inside[i] >= _READ_FRACTION * area:
            continue
        own = labels[y:y + h, x:x + w] == i
        if _part_of_larger_shape(gray, background, labels, i, x, y, w, h):
            continue  # a corner of a large filled shape (logo, shaded box)
        peak = int(contrast[y:y + h, x:x + w][own].max())
        unread.append((x, y, w, h, area, peak))

    def speck(c):
        return max(c[2], c[3]) <= _SPECK_MAX_SIDE and c[4] <= _SPECK_MAX_AREA

    reasons = []
    # Identifier-shaped unread print: a run of at least _ID_MIN_RUN marks, or
    # any run (a merged blob included) long enough for about nine digits.
    if _identifier_run(unread, 0) or _identifier_run(unread, 1):
        reasons.append("identifier_shaped")
    # Page budget: digit-sized unread marks anywhere on the page, whatever the
    # gaps, dot leaders, boxes or line breaks between them.
    budget = _digit_budget(unread)
    if budget >= _ID_MIN_DIGITS:
        reasons.append("digit_budget")
    ink = sum(c[4] for c in unread if not speck(c)) / marks.size
    return {
        "unread_digits": budget,
        "unread_components": len(unread),
        "unread_ink_fraction": ink,
        "regions": [
            [round(c[0] / sx, 1), round(c[1] / sy, 1), round(c[2] / sx, 1), round(c[3] / sy, 1)]
            for c in unread if not speck(c)
        ][:20],
        "reasons": reasons,
        "text_like": bool(reasons),
    }


def text_like_marks(pdf_bytes: bytes, page_index: int, dpi: float) -> dict:
    """The unread-print test for a page OCR read no word from."""
    return unread_text(pdf_bytes, page_index, dpi, [])


def _words(result: dict, pdf_bytes: bytes) -> dict[int, list[Word]]:
    """Words per page; refuses any page OCR could not be trusted to have read.

    A ``failed`` page always refuses. Every other page refuses when it shows
    print that no OCR word covers (``unread_text``): words OCR found elsewhere
    on the page are no evidence about the line it missed. Both backends tag
    failed pages.
    """
    failed = sorted(p["page_index"] for p in result["pages"] if p.get("status") == "failed")
    if failed:
        raise RedactionIncompleteError(
            f"OCR failed on page(s) {failed}; identifiers there cannot be ruled out"
        )
    dpi = (result.get("provenance") or {}).get("render_dpi") or 216
    def word(w, kind="word"):
        box = w["bbox"]
        return Word(w["text"], fitz.Rect(box["x"], box["y"], box["x"] + box["w"],
                                         box["y"] + box["h"]), kind)

    out: dict[int, list[Word]] = {}
    for page in result["pages"]:
        # Words below the review threshold, and whole recognized lines
        # (RapidOCR may split a read line into fragments), are searched and
        # count as read print: an uncertain reading of an SSN is redacted.
        out[page["page_index"]] = [
            *(word(w) for w in page["words"]),
            *(word(w) for w in page.get("_low_confidence_words", [])),
            *(word(w, "line") for w in page.get("_lines", [])),
        ]
    unread = []
    for index, words in out.items():
        check = unread_text(pdf_bytes, index, dpi, words)
        if check["text_like"]:
            log.warning("redaction_unread_text", page_index=index, reasons=check["reasons"],
                        regions=check["regions"])
            unread.append(index)
    if unread:
        raise RedactionIncompleteError(
            f"page(s) {sorted(unread)} contain text OCR did not read; "
            "identifiers there cannot be ruled out"
        )
    return out


def _as_pdf(data: bytes, media_type: str) -> bytes:
    if media_type == "application/pdf":
        return data
    image = fitz.open(stream=data, filetype=media_type.split("/")[-1])
    pdf = image.convert_to_pdf()
    return pdf


def _scan(doc: fitz.Document, pdf_bytes: bytes, ocr: OcrWords) -> tuple[list[Found], list[int]]:
    found: list[Found] = []
    ocr_pages: list[int] = []
    for i, page in enumerate(doc):
        found.extend(find_identifiers(i, text_layer_words(page), "text_layer", page))
        if needs_ocr(page):
            ocr_pages.append(i)
    if ocr_pages:
        suspect: dict[int, list[str]] = {}
        for i, words in ocr(pdf_bytes, ocr_pages).items():
            for text in [w.text for w in words if w.kind == "line"] or [
                " ".join(w.text for w in line) for line in _lines(words)
            ]:
                if suspect_runs(text):
                    suspect.setdefault(i, []).extend(suspect_runs(text))
            for hit in find_identifiers(i, words, "ocr", doc[i]):
                # A word and its recognized line can both carry the identifier;
                # both boxes are redacted, reported once.
                twin = next(
                    (f for f in found
                     if (f.page, f.kind, f.last4, f.source) == (hit.page, hit.kind,
                                                                 hit.last4, hit.source)
                     and any(a.intersects(b) for a in f.rects for b in hit.rects)),
                    None,
                )
                if twin is None:
                    found.append(hit)
                else:
                    found[found.index(twin)] = Found(
                        twin.page, twin.kind, twin.last4, twin.source,
                        twin.rects + hit.rects,
                    )
        if suspect:
            # Seven or more digits read in a row that are not an identifier
            # may be a misread one; the model must not see it either way.
            raise RedactionIncompleteError(
                f"page(s) {sorted(suspect)} contain digit runs that may be a misread "
                "identifier; identifiers there cannot be ruled out"
            )
    return found, ocr_pages


def redact_identifiers(
    data: bytes,
    media_type: str = "application/pdf",
    ocr: OcrWords = ocr_words,
) -> dict:
    """Redact every identifier and prove none is left.

    Returns {"pdf_bytes", "redactions": [{page, kind, last4, source}], "ocr_pages"}.
    Raises RedactionIncompleteError when the check finds an identifier after redaction.
    """
    pdf_bytes = _as_pdf(data, media_type)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    found, ocr_pages = _scan(doc, pdf_bytes, ocr)

    for f in found:
        for rect in f.rects:
            # Vertical margin: OCR boxes hug the glyph ink.
            doc[f.page].add_redact_annot(rect + (0, -2, 0, 2), fill=(0, 0, 0))
    for i in {f.page for f in found}:
        doc[i].apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)
    redacted = doc.tobytes(garbage=4, deflate=True)

    # The check: search the redacted document the same way, OCR included for
    # every page that needed it the first time.
    check_doc = fitz.open(stream=redacted, filetype="pdf")
    remaining: list[Found] = []
    for i, page in enumerate(check_doc):
        remaining.extend(find_identifiers(i, text_layer_words(page), "text_layer", page))
    if ocr_pages:
        for i, words in ocr(redacted, ocr_pages).items():
            remaining.extend(find_identifiers(i, words, "ocr"))
    if remaining:
        pages = sorted({f.page for f in remaining})
        raise RedactionIncompleteError(
            f"{len(remaining)} identifier(s) still readable after redaction on page(s) {pages}"
        )

    return {
        "pdf_bytes": redacted,
        "redactions": [
            {"page": f.page, "kind": f.kind, "last4": f.last4, "source": f.source} for f in found
        ],
        "ocr_pages": ocr_pages,
    }
