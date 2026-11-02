"""Identity of bytes actually acquired by this engine, not a caller's claim.

The source can be an image converted once to a PDF. The second digest binds
that conversion to the sealed input resumed by queue workers. This receipt
attests neither an application's document/version nor OCR accuracy.
"""
import hashlib

import fitz


def input_provenance(source: bytes, effective_pdf: bytes) -> dict:
    from app.services.rapid_ocr import FITZ_LOCK

    with FITZ_LOCK, fitz.open(stream=effective_pdf, filetype="pdf") as doc:
        if not doc.is_pdf or doc.needs_pass or len(doc) < 1:
            raise ValueError("Input provenance requires a readable PDF")
        page_count = len(doc)
    return {
        "schema_version": 1,
        "input_sha256": hashlib.sha256(source).hexdigest(),
        "effective_pdf_sha256": hashlib.sha256(effective_pdf).hexdigest(),
        "input_size_bytes": len(source),
        "page_count": page_count,
    }


def verify_effective_input(receipt: dict, effective_pdf: bytes) -> None:
    if receipt.get("effective_pdf_sha256") != hashlib.sha256(effective_pdf).hexdigest():
        raise ValueError("OCR input differs from its acquisition receipt")


def native_input_provenance(source: bytes) -> dict | None:
    """Native PDF path only: do not claim a separately converted image as read."""
    from app.services.rapid_ocr import FITZ_LOCK

    with FITZ_LOCK, fitz.open(stream=source, filetype="pdf") as doc:
        if not doc.is_pdf:
            return None
    return input_provenance(source, source)
