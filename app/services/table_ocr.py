"""Table structure on scanned pages with RapidTable (TAX-6995).

Replaces PP-StructureV3, which pulled Paddle and a ~1.5 GB chart model into
the engine. The pieces, all ONNX on the CPU and all pinned:

* **Where the tables are.** Ruled tables are found from their lines: the
  page is binarized, long horizontal and vertical strokes are kept by
  morphological opening, and a region whose rules cross in a grid (at least
  two rows and two columns of cells) is a table. Borderless tables on a scan
  are not found; the page then reports unsupported structure.
* **Its structure.** Each region goes to RapidTable's SLANet-plus model
  (``slanet-plus.onnx``, hash-pinned), which predicts rows, columns and
  spans.
* **Its text.** The page is read once with the worker's own RapidOCR engine
  (the pinned PP-OCRv6 weights) and each table's lines are matched to its
  cells by RapidTable's matcher.

Runs as one unit per page on the OCR queue workers (app/services/ocr_queue.py)
so a table request shares the bounded worker pool with OCR and never builds
engines on request threads. A RapidTable object is never shared between
threads; each worker builds its own.
"""

from __future__ import annotations

import importlib.metadata
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import fitz
import numpy as np
import structlog

from app.config import settings
from app.services.table_evidence import TableStructureError, parse_table, pdf_bbox

log = structlog.get_logger()

RAPID_TABLE_VERSION = "3.0.2"

# Scans render at no more than the OCR cap; enough for 6-pt print.
MAX_DPI = 216
# A table region must cover at least this share of the page and this many
# points in each direction, and its rules must cross at least this many times
# (a 2 x 2 grid of cells has 9 crossings).
MIN_REGION_SHARE = 0.005
MIN_REGION_POINTS = 36
MIN_CROSSINGS = 9

_local = threading.local()


def _get_engine():
    """The calling thread's own RapidTable (structure + matcher only)."""
    engine = getattr(_local, "engine", None)
    if engine is None:
        from rapid_table import ModelType, RapidTable, RapidTableInput

        from app.services import rapid_ocr
        from app.services.rapid_ocr_models import TABLE_MODEL

        rapid_ocr.validate_runtime()
        if importlib.metadata.version("rapid-table") != RAPID_TABLE_VERSION:
            raise RuntimeError(f"Table recognition requires rapid-table=={RAPID_TABLE_VERSION}")
        threads = rapid_ocr.intra_op_threads()
        engine = _local.engine = RapidTable(
            RapidTableInput(
                model_type=ModelType.SLANETPLUS,
                model_dir_or_path=str(Path(settings.rapidocr_model_dir) / TABLE_MODEL),
                use_ocr=False,
                engine_cfg={"intra_op_num_threads": threads, "inter_op_num_threads": 1},
            )
        )
        log.info("rapidtable_engine_ready", thread=threading.current_thread().name)
    return engine


def table_regions(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Pixel boxes (x0, y0, x1, y1) of ruled tables on a grayscale page."""
    height, width = gray.shape
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )
    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(width // 40, 10), 1))
    )
    vertical = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(height // 60, 10)))
    )
    # Bridge the small breaks a scan leaves in its rules.
    joint = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    horizontal = cv2.dilate(horizontal, joint)
    vertical = cv2.dilate(vertical, joint)
    grid = cv2.bitwise_or(horizontal, vertical)
    crossings = cv2.bitwise_and(horizontal, vertical)
    count, _, stats, _ = cv2.connectedComponentsWithStats(grid, 8)
    regions = []
    for i in range(1, count):
        x, y, w, h, _ = stats[i]
        if w * h < MIN_REGION_SHARE * width * height:
            continue
        if w < MIN_REGION_POINTS * width / 612 or h < MIN_REGION_POINTS * height / 792:
            continue
        joints, _ = cv2.connectedComponents(crossings[y : y + h, x : x + w], connectivity=8)
        if joints - 1 >= MIN_CROSSINGS:
            regions.append((int(x), int(y), int(x + w), int(y + h)))
    regions.sort(key=lambda r: (r[1], r[0]))
    return regions


def html_rows(table_html: str) -> list[list[str]]:
    return parse_table(table_html)[0]


class TableReadResult(list):
    """Resolved regions plus explicit accounting of regions we could not read."""

    def __init__(self):
        super().__init__()
        self.unresolved_regions = 0
        self.failed_regions = 0


def read_tables(image_bgr: np.ndarray, page_width: float, page_height: float,
                page_index: int, ocr_engine) -> list[dict]:
    """Tables on one rendered page, with bboxes in PDF points."""
    from rapid_table.utils import format_ocr_results

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    regions = table_regions(gray)
    if not regions:
        return []
    read = ocr_engine(image_bgr)
    boxes = np.asarray(read.boxes, dtype=float) if read.boxes is not None else np.zeros((0, 4, 2))
    texts = list(read.txts or [])
    scores = list(read.scores or [])
    engine = _get_engine()
    sx, sy = page_width / gray.shape[1], page_height / gray.shape[0]
    tables = TableReadResult()
    for x0, y0, x1, y1 in regions:
        centers = boxes.mean(axis=1) if len(boxes) else np.zeros((0, 2))
        inside = [
            i for i, (cx, cy) in enumerate(centers) if x0 <= cx <= x1 and y0 <= cy <= y1
        ]
        if not inside:
            tables.unresolved_regions += 1
            continue
        try:
            crop = np.ascontiguousarray(image_bgr[y0:y1, x0:x1])
            local = boxes[inside] - np.array([x0, y0], dtype=float)
            dt_boxes, rec_res = format_ocr_results(
                (local, [texts[i] for i in inside], [scores[i] for i in inside]),
                crop.shape[0], crop.shape[1],
            )
            structures, cell_boxes = engine.table_structure([crop])
            # Upstream drops all-zero boxes without retaining HTML cell indexes.
            # Matching a compacted box list would shift text into another column.
            structure = structures[0][0] if len(structures) else []
            _, physical_cells = parse_table(structure if isinstance(structure, str) else "".join(structure))
            predicted = cell_boxes[0] if len(cell_boxes) else []
            if (not physical_cells or len(predicted) != len(physical_cells) or
                any(pdf_bbox(np.asarray(box).reshape(-1), (x0, y0), (sx, sy),
                             page_width, page_height) is None for box in predicted)):
                tables.unresolved_regions += 1
                continue
            # Use the exact assignment that produces the cell text for its source
            # geometry too. SLANet boxes describe a structure hypothesis: on real
            # scans they can collapse into a thin line or miss the printed text.
            matcher = engine.table_matcher
            # The pinned formatter preserves order but clips boxes to the crop.
            # Carry original page-token identities through the matcher's filter;
            # clipped matching rectangles must never become source evidence.
            indexed_readings = [(text, score, inside[i])
                                for i, (text, score) in enumerate(rec_res)]
            text_boxes, filtered_readings = matcher.filter_ocr_result(
                predicted, dt_boxes, indexed_readings)
            text_readings = [(text, score) for text, score, _ in filtered_readings]
            source_indices = [index for _, _, index in filtered_readings]
            assignment = matcher.match_result(predicted, text_boxes)
            matched, _ = matcher.get_pred_html(structure, assignment, text_readings)
            rows, cells = parse_table(matched or "")
            if not rows or len(cells) != len(physical_cells):
                tables.unresolved_regions += 1
                continue
        except MemoryError:
            raise
        except TableStructureError:
            tables.unresolved_regions += 1
            continue
        except Exception as exc:
            tables.failed_regions += 1
            log.warning("rapidtable_region_failed", page_index=page_index,
                        error_type=type(exc).__name__)
            continue
        for cell_index, cell in enumerate(cells):
            # Preserve individual token identity (including duplicate amounts).
            # Never search for matching text elsewhere or fall back to a guessed
            # structure box when this cell has no observed text geometry.
            assigned = [pdf_bbox(boxes[source_indices[i]].reshape(-1), (0, 0), (sx, sy),
                                 page_width, page_height)
                        for i in assignment.get(cell_index, [])]
            bounds = None
            if assigned and all(box is not None for box in assigned):
                bounds = {"x0": min(box["x0"] for box in assigned),
                          "y0": min(box["y0"] for box in assigned),
                          "x1": max(box["x1"] for box in assigned),
                          "y1": max(box["y1"] for box in assigned)}
            cell.update(page_index=page_index, bbox=bounds, source="rapidtable",
                        geometry_status="available" if bounds else "unavailable")
        warnings = (["cell_geometry_unavailable"] if any(c["bbox"] is None for c in cells)
                    else [])
        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]
        tables.append({
            "page_index": page_index,
            "page_width": page_width, "page_height": page_height,
            "bbox": {"x0": round(x0 * sx, 2), "y0": round(y0 * sy, 2),
                     "x1": round(x1 * sx, 2), "y1": round(y1 * sy, 2)},
            "rows": rows,
            "cells": cells,
            "warnings": warnings,
            "source": "rapidtable",
            "text_source": getattr(read, "text_source", "rapidocr"),
            "row_count": len(rows),
            "col_count": width,
        })
    return tables


def read_page_result(source, index: int, dpi: int, *, prefer_native: bool = False) -> dict:
    """One OCR-queue unit: render page ``index`` and read its tables.

    A recognition failure and unsupported structure have distinct outcomes;
    a broken model/package configuration fails the request.
    """
    from app.services import rapid_ocr

    start = time.monotonic()
    def result(status, tables=None, reason=None):
        return {"page_index": index, "status": status, "tables": tables or [],
                "reason": reason, "source": "rapidtable",
                "processing_time_ms": round((time.monotonic() - start) * 1000, 2)}

    native_read = None
    with rapid_ocr.FITZ_LOCK:
        with rapid_ocr.open_source(source) as doc:
            page = doc[index]
            width, height = page.rect.width, page.rect.height
            area = max(width * height, 1.0)
            zoom = min(min(dpi, MAX_DPI) / 72, (rapid_ocr.MAX_RENDER_PIXELS / area) ** 0.5)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB,
                                  alpha=False)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            if prefer_native and rapid_ocr.text_layer_quality(page)["accepted"]:
                boxes, texts = [], []
                for x0, y0, x1, y1, text, *_ in page.get_text("words"):
                    box = fitz.Rect(x0, y0, x1, y1) * page.rotation_matrix
                    box &= page.rect
                    if box.is_empty:
                        continue
                    left, right = box.x0 * pix.width / width, box.x1 * pix.width / width
                    top, bottom = box.y0 * pix.height / height, box.y1 * pix.height / height
                    boxes.append([[left, top], [right, top], [right, bottom], [left, bottom]])
                    texts.append(text)
                native_read = SimpleNamespace(boxes=np.asarray(boxes), txts=texts,
                                              scores=[1.0] * len(texts), text_source="native_pdf")
    # A uniformly white page can be positively identified as empty. Absence
    # of a ruled region on other pages is unsupported, not evidence of absence.
    if int(bgr.min()) == 255:
        return result("empty", reason="blank_page")
    if not table_regions(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)):
        return result("unsupported", reason="no_ruled_table_region")
    try:
        ocr_engine = (lambda image: native_read) if native_read is not None else rapid_ocr._get_engine()
        _get_engine()
    except MemoryError:
        raise
    except Exception as exc:
        raise rapid_ocr.RapidConfigurationError(str(exc)) from exc
    try:
        tables = read_tables(bgr, width, height, index, ocr_engine)
        if getattr(tables, "failed_regions", 0):
            return result("failed", tables, reason="table_region_failed")
        if getattr(tables, "unresolved_regions", 0):
            return result("unsupported", tables, reason="table_region_unresolved")
        return result("complete", tables) if tables else result(
            "unsupported", reason="table_structure_not_resolved")
    except MemoryError:
        raise
    except Exception as exc:
        log.warning("rapidtable_page_failed", page_index=index, error_type=type(exc).__name__)
        return result("failed", reason="table_recognition_failed")


def read_page(source, index: int, dpi: int) -> list[dict]:
    """Legacy list consumer: never silently convert incomplete extraction to empty."""
    result = read_page_result(source, index, dpi)
    if result["status"] in ("failed", "unsupported"):
        raise RuntimeError(result["reason"])
    return result["tables"]
