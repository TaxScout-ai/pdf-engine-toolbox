"""TAX-9543: a document whose image cannot be sent masked is read as masked text.

Synthetic identifiers only (area 9xx SSNs, 00-prefixed EINs).
"""

from __future__ import annotations

import base64
import re

import fitz
import pytest

from app.services import identifier_redaction as ir

SSN = "912-34-5678"
EIN = "00-1234567"


def _no_full_identifier(text: str) -> None:
    digits = re.sub(r"\D", "", text)
    assert "91234" not in digits  # the hidden part of the SSN
    assert "00123" not in digits  # the hidden part of the EIN


@pytest.mark.parametrize(
    "line",
    [
        f"Employee's SSN {SSN}",
        f"EIN {EIN}",
        f"TaxID-{SSN}",
        f"ID{SSN}",
        "SSN 912345678",
        "I23-45-678S",       # a confusion-garbled reading
        "912-34-\n5678",     # wrapped over two lines
        "9 1 2 3 4 5 6 7",   # a comb field read box by box, one box missed
        "123-45-678",        # a near-identifier run
        # Review round 1: groups that OCR returns as separate lines or boxes.
        "912-34\n5678",
        "912-34\n-5678",
        "912\n34\n5678",
        "9\n1\n2\n3\n4\n5\n6\n7\n8",  # a comb field, one box per line
        "Taxpayer SSN 912.34.5678",
        "EIN 00 1234567",
    ],
)
def test_identifiers_and_near_identifiers_are_masked(line):
    masked = ir.mask_text_for_model(line)
    assert len(masked) == len(line)
    _no_full_identifier(masked)
    assert "1234" not in re.sub(r"\D", "", masked).replace("5678", "")


def test_an_exact_identifier_keeps_its_last_four():
    assert ir.mask_text_for_model(f"SSN {SSN}") == "SSN XXX-XX-5678"
    assert ir.mask_text_for_model(f"EIN {EIN}") == "EIN XX-XXX4567"


@pytest.mark.parametrize(
    "line",
    [
        "Wages 78,000.00 Federal tax 6,100.00",
        "Date 04/15/2026",
        "Austin, TX 78701-1234",
        "Phone (512) 555-0100",
        "Box 12a D 1,500.00",
        "Control number NW-2025",
    ],
)
def test_amounts_dates_zip_and_phone_stay_readable(line):
    assert ir.mask_text_for_model(line) == line


def _scan(lines: list[str]) -> bytes:
    src = fitz.open()
    page = src.new_page(width=612, height=792)
    for n, line in enumerate(lines):
        page.insert_text((72, 100 + 24 * n), line, fontsize=11)
    png = page.get_pixmap(dpi=150).tobytes("png")
    out = fitz.open()
    out.new_page(width=612, height=792).insert_image(fitz.Rect(0, 0, 612, 792), stream=png)
    return out.tobytes()


def test_a_scan_is_read_through_ocr_and_masked():
    lines = [f"a Employee's social security number {SSN}", "1 Wages 78,000.00"]
    pages = ir.masked_text_pages(
        _scan(lines), ocr_text=lambda pdf, idx: {i: "\n".join(lines) for i in idx}
    )
    assert pages == [
        {"page_index": 0,
         # The box number on the next line joins the run and is masked with
         # it; the SSN keeps its last four.
         "text": "a Employee's social security number XXX-XX-5678\nX Wages 78,000.00"}
    ]


def test_a_text_layer_page_is_read_from_its_text_without_ocr():
    doc = fitz.open()
    doc.new_page(width=612, height=792).insert_text(
        (72, 100), f"Payer TIN {EIN} and a long enough line of text here", fontsize=11)

    def no_ocr(*_):
        raise AssertionError("no OCR for a text-layer page")

    [page] = ir.masked_text_pages(doc.tobytes(), ocr_text=no_ocr)
    _no_full_identifier(page["text"])
    assert "XX-XXX4567" in page["text"]


@pytest.mark.parametrize("sync", [True, False])
def test_the_route_answers_a_refusal_with_masked_text(client, auth_headers, monkeypatch, sync):
    import json

    from app.routes import redact as route

    def refuse(*_a, **_k):
        raise ir.RedactionIncompleteError("1 identifier(s) still readable after redaction")

    monkeypatch.setattr(route.identifier_redaction, "redact_identifiers", refuse)
    monkeypatch.setattr(
        route.identifier_redaction, "masked_text_pages",
        lambda data, media_type: [{"page_index": 0, "text": "SSN XXX-XX-5678"}],
    )
    monkeypatch.setattr(route.identifier_redaction, "needs_ocr", lambda page: not sync)
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), "a page with enough words on it to read", fontsize=11)
    body = json.dumps({"content_base64": base64.b64encode(doc.tobytes()).decode(),
                       "text_fallback": True})
    r = client.post("/redact/identifiers", content=body,
                    headers=auth_headers("POST", "/redact/identifiers", body))
    assert r.status_code == 200
    payload = r.json()
    if not sync:
        task = payload["task_id"]
        path = f"/tasks/{task}"
        status = client.get(path, headers=auth_headers("GET", path, "")).json()
        data = status.get("data", status).get("result")
    else:
        data = payload["data"]
    assert data["text_only"] is True
    assert data["pages"] == [{"page_index": 0, "text": "SSN XXX-XX-5678"}]
    assert "pdf_base64" not in data


def _ocr_page(*boxes):
    """An OCR page result: recognized lines as (text, x, y, w, h) boxes."""
    return {"_lines": [{"text": t, "bbox": {"x": x, "y": y, "w": w, "h": h}}
                       for t, x, y, w, h in boxes]}


@pytest.mark.parametrize(
    "boxes",
    [
        # One printed row read as three boxes, at slightly different heights.
        [("Recipient TIN", 50, 100, 70, 10), ("912", 130, 101, 18, 10),
         ("34", 152, 99, 12, 10), ("5678", 168, 100.5, 24, 10)],
        # A comb field read box by box, glyph heights jittering.
        [(d, 100 + 12 * n, 200 + (n % 3), 8, 10) for n, d in enumerate("912345678")],
        # Wrapped onto the next row, no dash at the end of the first.
        [("SSN 912-34", 50, 300, 60, 10), ("5678 Box 1", 50, 313, 60, 10)],
    ],
)
def test_fragments_read_from_real_geometry_are_one_row_and_masked(boxes):
    text = ir.page_text_from_ocr(_ocr_page(*boxes))
    masked = ir.mask_text_for_model(text)
    _no_full_identifier(masked)
    assert "1234" not in re.sub(r"\D", "", masked)


def test_rows_read_in_order():
    text = ir.page_text_from_ocr(_ocr_page(
        ("Wages", 300, 100.4, 40, 10), ("Box 1", 50, 100, 30, 10),
        ("Employer", 50, 140, 50, 10),
    ))
    assert text == "Box 1 Wages\nEmployer"

