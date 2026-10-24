"""One table page on the shared OCR queue, including native PDF extraction."""
import time

import structlog

from app.config import settings
from app.services import rapid_ocr, table_ocr
from app.services.ocr_preprocessing import text_layer_quality

log = structlog.get_logger()


def read_page(source, index: int, params: dict) -> dict:
    from app.services import pdf_service

    started = time.monotonic()
    strategy = params.get("strategy", "auto")
    result = None
    if strategy in ("auto", "pymupdf"):
        try:
            with rapid_ocr.FITZ_LOCK, rapid_ocr.open_source(source) as doc:
                page = doc[index]
                if strategy == "pymupdf" or text_layer_quality(page)["accepted"]:
                    tables = pdf_service._extract_tables_pymupdf(doc, index)
                    if tables or strategy == "pymupdf":
                        blank = (not page.get_text().strip() and not page.get_images()
                                 and not page.get_drawings())
                        result = {
                            "page_index": index, "tables": tables, "source": "native_pdf",
                            "status": "complete" if tables else "empty" if blank else "unsupported",
                            "reason": None if tables else "blank_page" if blank else "native_table_not_resolved",
                        }
        except MemoryError:
            raise
        except Exception as exc:
            log.warning("native_table_page_failed", page_index=index, error_type=type(exc).__name__)
            if strategy == "pymupdf":
                result = {"page_index": index, "tables": [], "source": "native_pdf",
                          "status": "failed", "reason": "native_table_recognition_failed"}
    if result is None:
        if params.get("language", "en") != "en":
            raise pdf_service.PdfEngineError(
                "UNSUPPORTED_LANGUAGE", "Table recognition on page images supports only language=en", 422)
        result = table_ocr.read_page_result(
            source, index, params.get("dpi", 300), prefer_native=strategy == "auto")
    # Explicit native-only / RapidTable comparisons never silently use another model.
    # Geometry warnings remain visible for review; a complete native table does
    # not justify loading a second model and replacing its correct text.
    needs_fallback = result["status"] in ("unsupported", "failed")
    if strategy == "auto" and needs_fallback and settings.table_fallback == "docling":
        claim = params.get("claim_fallback")
        allowed = claim() if callable(claim) else params.get("allow_fallback", True)
        if allowed:
            from app.services import table_docling

            alternative = table_docling.read_page_result(source, index)
            alternative["fallback_from"] = result["source"]
            # Preserve usable partial evidence if the more expensive reader fails.
            if alternative["status"] in ("complete", "empty") or not result["tables"]:
                result = alternative
            else:
                result["status"] = "unsupported"
                result["reason"] = alternative.get("reason") or "fallback_unresolved"
        else:
            result["status"] = "unsupported"
            result["reason"] = "fallback_page_limit"
    result["processing_time_ms"] = round((time.monotonic() - started) * 1000, 2)
    return result
