"""PDF/A output through Ghostscript (TAX-5637)."""

import json
import re
import shutil
from unittest.mock import AsyncMock, patch

import fitz
import pytest

from app.services import pdfa_service

needs_gs = pytest.mark.skipif(
    shutil.which("gs") is None, reason="Ghostscript is not installed"
)


@needs_gs
def test_converts_to_pdfa_2b_with_an_output_intent(sample_pdf_bytes):
    out = pdfa_service.pdf_to_pdfa(sample_pdf_bytes)
    doc = fitz.open(stream=out, filetype="pdf")
    assert doc.page_count == 5
    xmp = doc.get_xml_metadata()
    assert "pdfaid:part" in xmp
    assert re.search(r"pdfaid:part(=['\"]2['\"]|>2<)", xmp), xmp[:400]
    catalog = doc.xref_object(doc.pdf_catalog())
    assert "/OutputIntents" in catalog
    assert "Page 1" in doc[0].get_text()


@needs_gs
def test_refuses_a_password_protected_pdf(sample_pdf_bytes):
    src = fitz.open(stream=sample_pdf_bytes, filetype="pdf")
    locked = src.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u"
    )
    with pytest.raises(pdfa_service.PdfaConversionError):
        pdfa_service.pdf_to_pdfa(locked)


def test_route_requires_auth(client):
    assert client.post("/convert/pdfa", json={"source_url": "https://e/x.pdf"}).status_code == 401


@needs_gs
def test_route_returns_a_pdf(client, auth_headers, sample_pdf_bytes):
    body = json.dumps({"source_url": "https://example.com/f.pdf"})
    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=sample_pdf_bytes,
    ):
        response = client.post(
            "/convert/pdfa", content=body, headers=auth_headers("POST", "/convert/pdfa", body)
        )
    assert response.status_code == 200
    assert response.content.startswith(b"%PDF")
