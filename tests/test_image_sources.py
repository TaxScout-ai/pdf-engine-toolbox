"""Image documents in the toolbox (TAX-5613).

Clients upload phone photos and scans as JPEG/PNG. MuPDF sniffs the bytes and
opens them as image documents even when told the type is PDF, and PyMuPDF
1.27.1 then fails: ``Document.permissions`` raises TypeError (500 on /info) and
``Document.tobytes()`` raises AssertionError (500 on every transform).
"""

import json
from unittest.mock import AsyncMock, patch

import fitz
import pytest

from app.services import pdf_service


def _image_bytes(fmt: str) -> bytes:
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 120, 160), False)
    pix.clear_with(230)
    return pix.tobytes(fmt)


@pytest.mark.parametrize("fmt", ["png", "jpg"])
def test_open_pdf_turns_an_image_into_a_one_page_pdf(fmt):
    doc = pdf_service._open_pdf(_image_bytes(fmt))
    assert doc.is_pdf
    assert doc.page_count == 1
    doc.tobytes()


def test_open_pdf_keeps_a_pdf_as_it_is(sample_pdf_bytes):
    doc = pdf_service._open_pdf(sample_pdf_bytes)
    assert doc.is_pdf
    assert doc.page_count == 5


def test_info_of_an_image(client, auth_headers):
    body = json.dumps({"source_url": "https://example.com/photo.png"})
    headers = auth_headers("POST", "/info", body)
    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=_image_bytes("png"),
    ):
        response = client.post("/info", content=body, headers=headers)
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["page_count"] == 1
    assert data["permissions"]["print"] is True


def test_deskew_of_an_image_returns_a_pdf():
    out = pdf_service.deskew_pages(_image_bytes("png"), [0])
    assert out.startswith(b"%PDF")
