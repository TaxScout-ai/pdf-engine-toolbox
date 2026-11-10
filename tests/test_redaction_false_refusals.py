"""TAX-9096: clean scans refused by identifier redaction, and the risks that stay refused.

Two false refusals measured on the synthetic TAX-7286 set:

1. A W-2 control number such as "NW-2025-004417" is a 10-digit run one digit
   off an SSN read without its second dash (3-6), so every page carrying one
   was refused as "digit runs that may be a misread identifier". A run that
   is the digits of a lettered reference number is now blacked out WHOLE,
   last four included, instead; a bare near-identifier run still refuses.
2. On a 1095-A, the black boxes over the SSNs of neighbouring table rows left
   a thin strip of paper between them. In the check pass that strip, inside
   a dark median background, read as "light print on a dark fill", i.e. as
   text OCR did not read. The check now paints the module's own boxes in the
   paper tone before it looks for unread print.

Synthetic identifiers only (area 9xx SSNs, 00-prefixed EINs).
"""

from __future__ import annotations

import math

import fitz
import pytest

from app.services import identifier_redaction as ir

DPI = 200


def _scan(draw) -> tuple[bytes, list[tuple[str, fitz.Rect]], fitz.Document]:
    """An image-only page, its source words, and the source page (for search)."""
    source = fitz.open()
    page = source.new_page(width=612, height=792)
    draw(page)
    words = [(w[4], fitz.Rect(w[:4])) for w in page.get_text("words")]
    png = page.get_pixmap(dpi=DPI).tobytes("png")
    out = fitz.open()
    out.new_page(width=612, height=792).insert_image(fitz.Rect(0, 0, 612, 792), stream=png)
    return out.tobytes(), words, source


def _part(source: fitz.Document, rect: fitz.Rect, text: str) -> fitz.Rect:
    hits = source[0].search_for(text, clip=rect + (-1, -1, 1, 1))
    assert hits, text
    return sorted(hits, key=lambda r: r.x0)[-1]


def _ocr(words, source, check_reads=None):
    """First call: every source word. Later calls (the check pass): what
    ``check_reads(text, rect)`` says is still legible — by default the kept
    last four digits of an identifier, nothing of a masked reference number,
    and every other word as it is."""
    calls = []

    def default(text, rect):
        digits = sum(c.isdigit() for c in text)
        if digits == 9:
            return [(text[-4:], _part(source, rect, text[-4:]))]
        if text.startswith("NW-") or text.startswith("XQ-"):
            return [(text[:3], _part(source, rect, text[:3]))]
        return [(text, rect)]

    def ocr(pdf_bytes, pages):
        calls.append(1)
        out = []
        for text, rect in words:
            if len(calls) == 1:
                out.append(ir.Word(text, rect))
            else:
                out.extend(ir.Word(t, r) for t, r in (check_reads or default)(text, rect))
        return {0: out}

    return ocr


# --- 1. Reference numbers ------------------------------------------------


def _w2(control: str):
    def draw(p):
        p.insert_text((72, 100), "a Employee's social security number", fontsize=8)
        p.insert_text((90, 115), "912-34-5678", fontsize=11)
        p.insert_text((72, 140), "b Employer identification number (EIN)", fontsize=8)
        p.insert_text((90, 155), "00-1234567", fontsize=11)
        p.insert_text((72, 180), "d Control number", fontsize=8)
        p.insert_text((90, 195), control, fontsize=11)
        p.insert_text((300, 130), "1 Wages, tips, other comp. 78,000.00", fontsize=10)
    return draw


def _dark(pdf: bytes, rect: fitz.Rect) -> bool:
    """Every column across the rect is blacked out at mid-height."""
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        pix = doc[0].get_pixmap(dpi=144, clip=rect, colorspace=fitz.csGRAY)
    y = pix.height // 2
    return all(pix.samples[y * pix.width + x] < 60 for x in range(1, pix.width - 1))


def test_a_w2_control_number_no_longer_refuses_and_is_masked_whole():
    scan, words, source = _scan(_w2("NW-2025-004417"))
    result = ir.redact_identifiers(scan, ocr=_ocr(words, source))
    assert sorted(r["kind"] for r in result["redactions"]) == ["ein", "ssn"]
    assert result["masked_references"] >= 1
    control = next(r for t, r in words if t.startswith("NW-"))
    # Every digit of the control number, its last four included, is black.
    digits = _part(source, control, "2025-004417")
    assert _dark(result["pdf_bytes"], digits + (1, 0, -1, 0))


@pytest.mark.parametrize("reading", ["XQ-12-34567", "XQ-123-45-678"])
def test_a_reference_shaped_misread_identifier_reaches_no_model(reading):
    """If a "reference" was really a misread identifier, none of it is sent."""
    scan, words, source = _scan(_w2(reading))
    result = ir.redact_identifiers(scan, ocr=_ocr(words, source))
    run = next(r for t, r in words if t == reading)
    assert _dark(result["pdf_bytes"], _part(source, run, reading[3:]) + (1, 0, -1, 0))


@pytest.mark.parametrize(
    "reading",
    [
        "123-45-678",      # an SSN read with a digit lost
        "SSN-123-45-678",  # a label is never a reference prefix
        "EIN-12-345678",
        "NUM-123-45-678",
        "TIN/123-45-678",
        "A23-45-6789",     # a digit read as a letter, no separator
        "1 2 3 4 5 6 7 8",  # a comb field with a box missed
    ],
)
def test_a_bare_or_labelled_near_identifier_still_refuses(reading):
    scan, words, source = _scan(_w2("A-0417"))
    words = [(reading if t == "912-34-5678" else t, r) for t, r in words]
    with pytest.raises(ir.RedactionIncompleteError, match="misread identifier"):
        ir.redact_identifiers(scan, ocr=_ocr(words, source))


def test_a_masked_reference_still_readable_in_the_output_refuses():
    scan, words, source = _scan(_w2("NW-2025-004417"))

    def still_reads(text, rect):
        if text.startswith("NW-"):
            return [(text, rect)]  # the mask missed: the check reads it all
        if sum(c.isdigit() for c in text) == 9:
            return [(text[-4:], _part(source, rect, text[-4:]))]
        return [(text, rect)]

    with pytest.raises(ir.RedactionIncompleteError, match="still show digit runs"):
        ir.redact_identifiers(scan, ocr=_ocr(words, source, still_reads))


def test_reference_detection_is_narrow():
    norm = ir.normalize_confusions
    assert ir._is_reference(norm("NW-2025-004417"), 3)
    assert ir._is_reference(norm("Policy CA-POL-2025-004417"), len("Policy CA-POL-"))
    for text, start in [("SSN-123-45-678", 4), ("ID-12-345678", 3), ("A23-45-6789", 1),
                        ("123-45-678", 0), ("SS-123-45-678", 3), ("NW 2025 004417", 3)]:
        assert not ir._is_reference(norm(text), start), text


# --- 2. Our own boxes in the check pass --------------------------------------


SSNS = ["912-34-5671", "912-34-5672", "912-34-5673", "912-34-5674"]


def _covered_individuals(p):
    """1095-A Part IV: SSNs on table rows 28 pt apart with rules between them."""
    p.insert_text((72, 260), "B. Covered individual SSN", fontsize=8)
    for n, ssn in enumerate(SSNS):
        y = 285 + 28 * n
        p.insert_text((250, y), ssn, fontsize=11)
        p.draw_line((60, y + 9), (560, y + 9), color=(0, 0, 0), width=0.8)


def test_boxes_on_neighbouring_rows_do_not_read_as_unread_print():
    scan, words, source = _scan(_covered_individuals)
    result = ir.redact_identifiers(scan, ocr=_ocr(words, source))
    assert [r["kind"] for r in result["redactions"]] == ["ssn"] * len(SSNS)


def _redacted_table(extra=None) -> tuple[bytes, list[ir.Word], dict[int, list[fitz.Rect]]]:
    """The table redacted the way the module does it, plus the check-pass words."""
    def draw(p):
        _covered_individuals(p)
        if extra:
            extra(p)

    scan, words, source = _scan(draw)
    found = ir.find_identifiers(0, [ir.Word(t, r) for t, r in words if t in SSNS], "ocr",
                                fitz.open(stream=scan, filetype="pdf")[0])
    doc = fitz.open(stream=scan, filetype="pdf")
    boxes = [r + (0, -2, 0, 2) for f in found for r in f.rects]
    for box in boxes:
        doc[0].add_redact_annot(box, fill=(0, 0, 0))
    doc[0].apply_redactions(images=fitz.PDF_REDACT_IMAGE_PIXELS)
    check_words = [ir.Word(t[-4:], _part(source, r, t[-4:])) for t, r in words if t in SSNS]
    check_words += [ir.Word(t, r) for t, r in words if t not in SSNS and not t[0].isdigit()]
    return doc.tobytes(), check_words, {0: boxes}


def _unread_with_fills(pdf, words, fills):
    token = ir._OWN_FILLS.set(fills)
    try:
        return ir.unread_text(pdf, 0, 216, words)
    finally:
        ir._OWN_FILLS.reset(token)


def test_the_check_still_sees_an_unread_identifier_beside_our_boxes():
    """Painting our boxes in paper tone never hides print next to them."""
    def beside(p):
        p.insert_text((380, 285 + 28), "912-34-5699", fontsize=11)  # right of a box, unread

    pdf, words, fills = _redacted_table(beside)
    assert _unread_with_fills(pdf, words, fills)["text_like"]


def test_the_check_still_sees_light_print_in_a_dark_fill_that_is_not_ours():
    def bar(p):
        p.draw_rect(fitz.Rect(300, 450, 520, 480), color=None, fill=(0, 0, 0))
        p.insert_text((310, 470), "912-34-5699", fontsize=14, color=(1, 1, 1))

    pdf, words, fills = _redacted_table(bar)
    assert _unread_with_fills(pdf, words, fills)["text_like"]


def test_the_clean_redacted_table_passes_the_check_only_with_fills():
    """The regression: without knowing our boxes, the strip of paper between
    two of them read as unread print, and the page was refused."""
    pdf, words, fills = _redacted_table()
    assert not _unread_with_fills(pdf, words, fills)["text_like"]
    assert ir.unread_text(pdf, 0, 216, words)["text_like"]


# --- 3. Images saved without a usable DPI -------------------------------------


def _png(width: int, height: int, draw=None, scale: float = 1.0) -> bytes:
    """A PNG with no DPI recorded (as OpenCV and many phone apps write them)."""
    import cv2
    import numpy as np

    if draw is None:
        image = np.full((height, width), 250, np.uint8)
    else:
        with fitz.open() as src:
            page = src.new_page(width=width / scale, height=height / scale)
            draw(page)
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csGRAY)
            image = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


def _render_pixels(pdf: bytes) -> int:
    from app.services.rapid_ocr import MAX_DPI

    with fitz.open(stream=pdf, filetype="pdf") as doc:
        zoom = MAX_DPI / 72
        return max(math.ceil(p.rect.width * zoom) * math.ceil(p.rect.height * zoom) for p in doc)


def test_an_image_without_dpi_becomes_a_page_ocr_can_render():
    from app.services.rapid_ocr import MAX_RENDER_PIXELS

    raw = fitz.open(stream=_png(1700, 2200), filetype="png").convert_to_pdf()
    assert _render_pixels(raw) > MAX_RENDER_PIXELS  # the old page: OCR failed
    pdf = ir._as_pdf(_png(1700, 2200), "image/png")
    assert _render_pixels(pdf) <= MAX_RENDER_PIXELS
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        # Rendered at the OCR DPI, the page is the image at its own size.
        assert doc[0].rect.width * 3 == pytest.approx(1700, abs=1)


def test_a_huge_photo_is_scaled_within_the_limit():
    from app.services.rapid_ocr import MAX_RENDER_PIXELS

    pdf = ir._as_pdf(_png(4000, 6000), "image/png")
    assert _render_pixels(pdf) <= MAX_RENDER_PIXELS


def test_an_image_with_its_dpi_keeps_its_page():
    with fitz.open() as src:
        src.new_page(width=612, height=792)
        png = src[0].get_pixmap(dpi=300).tobytes("png")  # 300 DPI recorded
    assert ir._as_pdf(png, "image/png") == fitz.open(stream=png, filetype="png").convert_to_pdf()


def test_an_identifier_on_a_resized_image_page_is_blacked_out():
    """Redaction blanks the image's own pixels on the resized page."""
    scale = 1700 / 612
    png = _png(1700, 2200, lambda p: p.insert_text((90, 200), "912-34-5678", fontsize=11),
               scale=scale)
    pdf = ir._as_pdf(png, "image/png")
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        factor = doc[0].rect.width / 612
    with fitz.open() as src:
        page = src.new_page(width=612, height=792)
        page.insert_text((90, 200), "912-34-5678", fontsize=11)
        [(text, rect)] = [(w[4], fitz.Rect(w[:4]) * factor) for w in page.get_text("words")]
    calls = []

    def ocr(pdf_bytes, pages):
        calls.append(1)
        if len(calls) == 1:
            return {0: [ir.Word(text, rect)]}
        return {0: [ir.Word(text[-4:], fitz.Rect(rect.x1 - rect.width * 4 / 11, rect.y0,
                                                  rect.x1, rect.y1))]}

    result = ir.redact_identifiers(png, "image/png", ocr=ocr)
    assert [r["kind"] for r in result["redactions"]] == ["ssn"]
    prefix = fitz.Rect(rect.x0, rect.y0, rect.x0 + rect.width * 6 / 11, rect.y1)
    assert _dark(result["pdf_bytes"], prefix + (1, 0, -1, 0))
