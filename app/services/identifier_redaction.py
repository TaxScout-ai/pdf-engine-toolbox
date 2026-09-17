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
  * a page without one (a scan or a photo) — PaddleOCR words.

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


@dataclass(frozen=True)
class Found:
    page: int
    kind: str  # "ssn" | "ein" | "tin"
    last4: str
    source: str  # "text_layer" | "ocr"
    # The boxes to black out: everything of the identifier but its last four digits.
    rects: tuple[fitz.Rect, ...]


def _matches(line: str) -> Iterable[tuple[int, int, str, str]]:
    """(start, end, kind, digits) for every identifier in one line of text."""
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
    for line in _lines(words):
        text = ""
        spans: list[tuple[int, int, Word]] = []
        for w in line:
            if text:
                text += " "
            start = len(text)
            text += w.text
            spans.append((start, len(text), w))
        for start, end, kind, digits in _matches(text):
            cut = _prefix_end(text, start, end)
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


def paddle_ocr_words(pdf_bytes: bytes, pages: list[int]) -> dict[int, list[Word]]:
    from app.services import pdf_service

    result = pdf_service.ocr_pages(pdf_bytes, pages, force_ocr=True)
    out: dict[int, list[Word]] = {}
    for page in result["pages"]:
        out[page["page_index"]] = [
            Word(
                w["text"],
                fitz.Rect(
                    w["bbox"]["x"],
                    w["bbox"]["y"],
                    w["bbox"]["x"] + w["bbox"]["w"],
                    w["bbox"]["y"] + w["bbox"]["h"],
                ),
            )
            for w in page["words"]
        ]
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
        for i, words in ocr(pdf_bytes, ocr_pages).items():
            found.extend(find_identifiers(i, words, "ocr", doc[i]))
    return found, ocr_pages


def redact_identifiers(
    data: bytes,
    media_type: str = "application/pdf",
    ocr: OcrWords = paddle_ocr_words,
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
