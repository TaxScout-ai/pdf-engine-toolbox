"""Optional local TableFormer fallback; lazy, serialized and offline.

The request must already be running on the shared OCR queue. One converter
is retained per process, rather than one PyTorch model per request/worker.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import threading
from pathlib import Path

import structlog

from app.config import settings
from app.services import rapid_ocr
from app.services.table_evidence import MAX_CELLS, TableStructureError, pdf_bbox

log = structlog.get_logger()
_lock = threading.Lock()
_converter = None


def _get_converter():
    global _converter
    if _converter is not None:
        return _converter
    if importlib.metadata.version("docling") != "2.140.0":
        raise RuntimeError("The evaluated fallback requires docling==2.140.0")
    root = Path(settings.docling_model_dir)
    manifest = json.loads(Path(__file__).with_name("docling_model_hashes.json").read_text())
    for name, expected in manifest.items():
        with (root / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise RuntimeError("Docling model hash mismatch")
    rapid_ocr.validate_runtime()
    # No model/network fallback is allowed when reading customer documents.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption
    import torch

    threads = rapid_ocr.intra_op_threads()
    torch.set_num_threads(threads)
    models = Path(settings.rapidocr_model_dir)
    options = PdfPipelineOptions(
        artifacts_path=root, enable_remote_services=False, allow_external_plugins=False,
        do_ocr=True, do_table_structure=True, document_timeout=120,
        accelerator_options=AcceleratorOptions(num_threads=threads, device=AcceleratorDevice.CPU),
    )
    options.ocr_options = RapidOcrOptions(
        lang=["en"],
        det_model_path=str(models / "PP-OCRv6_det_small.onnx"),
        rec_model_path=str(models / "PP-OCRv6_rec_small.onnx"),
        cls_model_path=str(models / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"),
        rapidocr_params={
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "EngineConfig.onnxruntime.use_cuda": False,
        },
    )
    options.table_structure_options.mode = TableFormerMode.ACCURATE
    options.table_structure_options.do_cell_matching = True
    _converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=options),
    })
    log.info("tableformer_ready", threads=threads, profile="docling-2.140.0-accurate")
    return _converter


class MappedTables(list):
    unresolved_tables = 0


def map_document(document, page_index: int, *, strict: bool = True) -> list[dict]:
    """Map single-page Docling coordinates and spans without inventing cells."""
    tables = MappedTables()
    for item in document.tables:
        try:
            if len(item.prov) != 1 or item.prov[0].page_no not in document.pages:
                raise TableStructureError("Unresolved table page")
            provenance = item.prov[0]
            page = document.pages[provenance.page_no]
            width, height = page.size.width, page.size.height
            data = item.data
            if data.num_rows <= 0 or data.num_cols <= 0 or data.num_rows * data.num_cols > MAX_CELLS:
                raise TableStructureError("Invalid table dimensions")
            rows = [[""] * data.num_cols for _ in range(data.num_rows)]
            cells, occupied = [], set()
            for cell in data.table_cells:
                r, c = cell.start_row_offset_idx, cell.start_col_offset_idx
                rs, cs = cell.row_span, cell.col_span
                if not (0 <= r < data.num_rows and 0 <= c < data.num_cols and
                        1 <= rs <= data.num_rows - r and 1 <= cs <= data.num_cols - c):
                    raise TableStructureError("Invalid table cell span")
                slots = {(y, x) for y in range(r, r + rs) for x in range(c, c + cs)}
                if slots & occupied:
                    raise TableStructureError("Overlapping table cells")
                occupied.update(slots)
                b = cell.bbox.to_top_left_origin(height) if cell.bbox else None
                bounds = pdf_bbox([b.l, b.t, b.r, b.b], (0, 0), (1, 1), width, height) if b else None
                text = " ".join(cell.text.split())
                rows[r][c] = text
                cells.append({
                    "page_index": page_index, "row_index": r, "column_index": c,
                    "row_span": rs, "column_span": cs, "raw_text": cell.text,
                    "text": text, "is_header": cell.column_header,
                    "bbox": bounds, "geometry_status": "available" if bounds else "unavailable",
                    "source": "docling_tableformer",
                })
            b = provenance.bbox.to_top_left_origin(height)
            bbox = pdf_bbox([b.l, b.t, b.r, b.b], (0, 0), (1, 1), width, height)
            if bbox is None:
                raise TableStructureError("Invalid table geometry")
            tables.append({
                "page_index": page_index, "page_width": width, "page_height": height,
                "bbox": bbox, "rows": rows, "cells": cells,
                "row_count": data.num_rows, "col_count": data.num_cols,
                "source": "docling_tableformer",
                "warnings": ["cell_geometry_unavailable"] if any(c["bbox"] is None for c in cells) else [],
            })
        except TableStructureError:
            if strict:
                raise
            tables.unresolved_tables += 1
    return tables


def read_page_result(source, index: int) -> dict:
    result = {"page_index": index, "tables": [], "source": "docling_tableformer",
              "status": "failed", "reason": "fallback_failed"}
    try:
        # Only this page enters the expensive reader; never resend the whole PDF.
        with rapid_ocr.FITZ_LOCK, rapid_ocr.open_source(source) as original:
            import fitz
            with fitz.open() as single:
                single.insert_pdf(original, from_page=index, to_page=index)
                data = single.tobytes()
        from docling.datamodel.base_models import DocumentStream

        with _lock:
            converted = _get_converter().convert(
                DocumentStream(name="page.pdf", stream=io.BytesIO(data)), raises_on_error=False)
            if converted.status.value != "success":
                result["reason"] = "fallback_conversion_incomplete"
                return result
            tables = map_document(converted.document, index, strict=False)
        if tables.unresolved_tables:
            result.update(tables=tables, status="unsupported", reason="fallback_invalid_table_structure")
        elif tables:
            result.update(tables=tables, status="complete", reason=None)
        else:
            # Absence of a model detection is not evidence that no table exists.
            result.update(status="unsupported", reason="fallback_no_table_structure")
    except MemoryError:
        raise
    except Exception as exc:
        log.warning("tableformer_page_failed", page_index=index, error_type=type(exc).__name__)
    return result
