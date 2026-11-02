import hashlib

import fitz
import pytest

from app.services.input_provenance import input_provenance, native_input_provenance, verify_effective_input
from app.services.ocr_queue import _as_pdf_bytes


def test_image_receipt_keeps_original_and_converted_byte_identities(sample_pdf_bytes):
    with fitz.open(stream=sample_pdf_bytes, filetype="pdf") as doc:
        source = doc[0].get_pixmap().tobytes("png")
    converted = _as_pdf_bytes(source)
    receipt = input_provenance(source, converted)
    assert receipt["input_sha256"] == hashlib.sha256(source).hexdigest()
    assert receipt["effective_pdf_sha256"] == hashlib.sha256(converted).hexdigest()
    assert receipt["input_sha256"] != receipt["effective_pdf_sha256"]
    assert receipt["page_count"] == 1
    verify_effective_input(receipt, converted)
    assert native_input_provenance(source) is None


def test_changed_sealed_input_cannot_publish_a_receipt(sample_pdf_bytes):
    receipt = input_provenance(sample_pdf_bytes, sample_pdf_bytes)
    with pytest.raises(ValueError, match="differs"):
        verify_effective_input(receipt, sample_pdf_bytes + b"changed")
