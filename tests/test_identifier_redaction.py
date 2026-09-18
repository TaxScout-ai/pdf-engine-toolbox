"""TAX-4858: identifiers are blacked out before a document reaches a model."""

import importlib.util
import re

import fitz
import pytest

from app.services import identifier_redaction as ir

# Synthetic identifiers only: SSA never issues area 9xx as SSNs; EIN prefix 00 is unassigned.
SSN = "912-34-5678"
EIN = "00-1234567"
IDENTIFIERS = (SSN, EIN, "923456789")


def _w2_page(doc: fitz.Document) -> fitz.Page:
    page = doc.new_page(width=612, height=792)
    lines = [
        f"a Employee's social security number {SSN}",
        f"b Employer identification number (EIN) {EIN}",
        "c Employer's name Northwind Traders, Austin TX 78701-1234",
        "1 Wages, tips, other compensation 75250.00",
        "Phone 512-555-0100 Routing 021000021",
        "RECIPIENT'S TIN 923456789",
    ]
    for n, line in enumerate(lines):
        page.insert_text((72, 100 + 30 * n), line, fontsize=11, fontname="helv")
    return page


def _digital_w2() -> bytes:
    doc = fitz.open()
    _w2_page(doc)
    return doc.tobytes()


def _scanned_w2() -> tuple[bytes, dict[str, fitz.Rect]]:
    """A page that is only an image of the W-2, plus where each word sits."""
    source = fitz.open()
    page = _w2_page(source)
    positions = {w[4]: fitz.Rect(w[:4]) for w in page.get_text("words")}
    words = [(w[4], fitz.Rect(w[:4])) for w in page.get_text("words")]
    pix = page.get_pixmap(dpi=150)
    scan = fitz.open()
    scan_page = scan.new_page(width=612, height=792)
    scan_page.insert_image(scan_page.rect, stream=pix.tobytes("png"))
    return scan.tobytes(), positions | {"__words__": words}


def _pixel_reading_ocr(words: list[tuple[str, fitz.Rect]]):
    """A stand-in for PaddleOCR that reads, character by character, what is not blacked out.

    Glyph ink is dark too, so a character counts as blacked out only when its
    whole cell, sampled across the line's height band, is black.
    """

    def ocr(pdf_bytes: bytes, pages: list[int]) -> dict[int, list[ir.Word]]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        out: dict[int, list[ir.Word]] = {}
        for i in pages:
            pix = doc[i].get_pixmap(dpi=144)
            scale = 2
            readable = []
            for text, rect in words:
                cell = rect.width / len(text)
                visible = ""
                for n, ch in enumerate(text):
                    x0 = int((rect.x0 + cell * n) * scale) + 1
                    x1 = int((rect.x0 + cell * (n + 1)) * scale) - 1
                    y = int((rect.y0 + rect.y1) / 2 * scale)
                    xs = range(x0, max(x0 + 1, x1))
                    black = all(
                        max(pix.pixel(min(x, pix.width - 1), min(y, pix.height - 1))[:3]) < 20
                        for x in xs
                    )
                    if not black:
                        visible += ch
                if visible:
                    readable.append(ir.Word(visible, rect))
            out[i] = readable
        return out

    return ocr


def test_digital_pdf_loses_every_identifier_and_keeps_other_numbers():
    result = ir.redact_identifiers(
        _digital_w2(), ocr=lambda *_: pytest.fail("no OCR for a text page")
    )
    text = fitz.open(stream=result["pdf_bytes"], filetype="pdf")[0].get_text()
    assert SSN not in text and EIN not in text and "923456789" not in text
    # Everything but the last four digits is gone; the last four stay for the model.
    assert "912-34" not in text and "00-123" not in text and "92345" not in text
    assert "5678" in text and "4567" in text and "6789" in text
    assert "78701-1234" in text and "512-555-0100" in text and "021000021" in text
    assert "75250.00" in text
    assert sorted((r["kind"], r["last4"], r["source"]) for r in result["redactions"]) == [
        ("ein", "4567", "text_layer"),
        ("ssn", "5678", "text_layer"),
        ("tin", "6789", "text_layer"),
    ]
    assert result["ocr_pages"] == []


def test_identifier_split_across_words_is_found():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "SSN", fontsize=11)
    page.insert_text((110, 100), "912", fontsize=11)
    page.insert_text((140, 100), "34", fontsize=11)
    page.insert_text((165, 100), "5678", fontsize=11)
    page.insert_text((72, 140), "filler text for a usable text layer, " * 3, fontsize=8)
    result = ir.redact_identifiers(doc.tobytes(), ocr=lambda *_: {})
    words = [
        w[4] for w in fitz.open(stream=result["pdf_bytes"], filetype="pdf")[0].get_text("words")
    ]
    assert words[0] == "SSN"
    assert "912" not in words and "34" not in words and "5678" in words


def test_scanned_page_is_redacted_from_ocr_boxes_and_checked_by_ocr():
    scan, positions = _scanned_w2()
    ocr = _pixel_reading_ocr(positions["__words__"])
    result = ir.redact_identifiers(scan, ocr=ocr)
    assert result["ocr_pages"] == [0]
    assert {r["last4"] for r in result["redactions"]} == {"5678", "4567", "6789"}
    assert all(r["source"] == "ocr" for r in result["redactions"])
    # The same reader now sees no identifier on the outgoing page.
    after = ocr(result["pdf_bytes"], [0])[0]
    seen = " ".join(w.text for w in after)
    assert not any(identifier in seen for identifier in IDENTIFIERS), seen
    assert "912-34" not in seen and "00-123" not in seen
    assert "021000021" in [w.text for w in after]  # a routing number stays readable
    assert "75250.00" in [w.text for w in after]


def test_a_photo_is_converted_and_redacted():
    scan, positions = _scanned_w2()
    png = fitz.open(stream=scan, filetype="pdf")[0].get_pixmap(dpi=72).tobytes("png")
    result = ir.redact_identifiers(png, "image/png", ocr=_pixel_reading_ocr(positions["__words__"]))
    assert fitz.open(stream=result["pdf_bytes"], filetype="pdf").page_count == 1
    assert {r["last4"] for r in result["redactions"]} == {"5678", "4567", "6789"}


def test_fails_closed_when_an_identifier_survives(monkeypatch):
    """Deliberate breakage: redaction that removes nothing must raise, never return."""
    monkeypatch.setattr(fitz.Page, "apply_redactions", lambda self, **_: None)
    with pytest.raises(ir.RedactionIncompleteError):
        ir.redact_identifiers(_digital_w2(), ocr=lambda *_: {})
    scan, positions = _scanned_w2()
    with pytest.raises(ir.RedactionIncompleteError):
        ir.redact_identifiers(scan, ocr=_pixel_reading_ocr(positions["__words__"]))


@pytest.mark.skipif(importlib.util.find_spec("paddleocr") is None, reason="PaddleOCR not installed")
def test_real_paddle_ocr_reads_no_identifier_on_the_redacted_scan():
    scan, _ = _scanned_w2()
    result = ir.redact_identifiers(scan)
    after = ir.paddle_ocr_words(result["pdf_bytes"], [0])[0]
    joined = " ".join(w.text for w in after)
    digits = re.sub(r"\D", "", joined)
    for identifier in IDENTIFIERS:
        assert re.sub(r"\D", "", identifier) not in digits, joined
    assert "75250" in joined
    # The last four digits stay readable for the model.
    assert "5678" in digits and "4567" in digits


def test_image_heavy_page_with_a_text_layer_is_still_ocred():
    """A scanner app's text layer over a photo: the photo may show what the text omits."""
    scan, _ = _scanned_w2()
    doc = fitz.open(stream=scan, filetype="pdf")
    doc[0].insert_text(
        (72, 760), "Scanned with a phone app; text layer without the identifiers " * 2, fontsize=6
    )
    assert ir.has_usable_text_layer(doc[0])
    assert ir.needs_ocr(doc[0])
    assert not ir.needs_ocr(fitz.open(stream=_digital_w2(), filetype="pdf")[0])


def test_route_redacts_a_digital_pdf_synchronously(client, auth_headers):
    import base64
    import json

    body = json.dumps(
        {
            "content_base64": base64.b64encode(_digital_w2()).decode(),
            "media_type": "application/pdf",
        }
    )
    response = client.post(
        "/redact/identifiers",
        content=body,
        headers=auth_headers("POST", "/redact/identifiers", body),
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    out = fitz.open(stream=base64.b64decode(data["pdf_base64"]), filetype="pdf")[0].get_text()
    assert SSN not in out and EIN not in out
    assert {r["last4"] for r in data["redactions"]} == {"5678", "4567", "6789"}


def test_route_queues_a_scan_as_a_task(client, auth_headers, monkeypatch):
    import base64
    import json

    from app.routes import redact as route

    queued = []
    monkeypatch.setattr(route, "_run_identifier_redaction", lambda *args: queued.append(args))
    scan, _ = _scanned_w2()
    body = json.dumps(
        {"content_base64": base64.b64encode(scan).decode(), "media_type": "application/pdf"}
    )
    response = client.post(
        "/redact/identifiers",
        content=body,
        headers=auth_headers("POST", "/redact/identifiers", body),
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "pending"
    assert len(queued) == 1


def test_route_refuses_invalid_base64(client, auth_headers):
    import json

    body = json.dumps({"content_base64": "not base64!!", "media_type": "application/pdf"})
    response = client.post(
        "/redact/identifiers",
        content=body,
        headers=auth_headers("POST", "/redact/identifiers", body),
    )
    assert response.status_code == 400


def test_route_text_layer_only_declines_a_scan_without_starting_ocr(
    client, auth_headers, monkeypatch
):
    """TAX-4858 C: a caller that redacts text-layer documents only gets an
    immediate answer for a scan, and no minutes-long OCR task is queued."""
    import base64
    import json

    from app.routes import redact as route

    queued = []
    monkeypatch.setattr(route, "_run_identifier_redaction", lambda *args: queued.append(args))
    scan, _ = _scanned_w2()
    body = json.dumps(
        {
            "content_base64": base64.b64encode(scan).decode(),
            "media_type": "application/pdf",
            "text_layer_only": True,
        }
    )
    response = client.post(
        "/redact/identifiers",
        content=body,
        headers=auth_headers("POST", "/redact/identifiers", body),
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "needs_ocr"
    assert "data" not in response.json()
    assert queued == []


def test_route_text_layer_only_still_redacts_a_digital_pdf(client, auth_headers):
    import base64
    import json

    body = json.dumps(
        {
            "content_base64": base64.b64encode(_digital_w2()).decode(),
            "media_type": "application/pdf",
            "text_layer_only": True,
        }
    )
    response = client.post(
        "/redact/identifiers",
        content=body,
        headers=auth_headers("POST", "/redact/identifiers", body),
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    out = fitz.open(stream=base64.b64decode(data["pdf_base64"]), filetype="pdf")[0].get_text()
    assert SSN not in out and EIN not in out
    assert "5678" in out
