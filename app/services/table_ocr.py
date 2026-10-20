"""Table structure on scanned pages with RapidTable (TAX-6995).

Replaces PP-StructureV3, which pulled Paddle and a ~1.5 GB chart model into
the engine. The pieces, all ONNX on the CPU and all pinned:

* **Where the tables are.** Ruled tables are found from their lines: the
  page is binarized, long horizontal and vertical strokes are kept by
  morphological opening, and a region whose rules cross in a grid (at least
  two rows and two columns of cells) is a table. Borderless tables on a scan
  are not found; the page then reports no table rather than guessing one.
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

import html
import importlib.metadata
import threading
from html.parser import HTMLParser
from pathlib import Path

import cv2
import fitz
import numpy as np
import structlog

from app.config import settings

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


class _HtmlTable(HTMLParser):
    """Rows of cell text from a table's HTML; spans repeat as empty cells."""

    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._colspan = 1

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
            self._colspan = 1
            for name, value in attrs:
                if name == "colspan":
                    try:
                        self._colspan = max(1, int(value))
                    except (TypeError, ValueError):
                        pass

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            row = self._row if self._row is not None else []
            row.append(" ".join("".join(self._cell).split()))
            row.extend([""] * (self._colspan - 1))
            self._row = row
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_rows(table_html: str) -> list[list[str]]:
    parser = _HtmlTable()
    parser.feed(table_html)
    parser.close()
    if parser._row:
        parser.rows.append(parser._row)
    return [[html.unescape(cell) for cell in row] for row in parser.rows]


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
    tables = []
    for x0, y0, x1, y1 in regions:
        centers = boxes.mean(axis=1) if len(boxes) else np.zeros((0, 2))
        inside = [
            i for i, (cx, cy) in enumerate(centers) if x0 <= cx <= x1 and y0 <= cy <= y1
        ]
        if not inside:
            continue
        crop = np.ascontiguousarray(image_bgr[y0:y1, x0:x1])
        local = boxes[inside] - np.array([x0, y0], dtype=float)
        dt_boxes, rec_res = format_ocr_results(
            (local, [texts[i] for i in inside], [scores[i] for i in inside]),
            crop.shape[0], crop.shape[1],
        )
        structures, cell_boxes = engine.table_structure([crop])
        matched = engine.table_matcher(structures, cell_boxes, [dt_boxes], [rec_res])
        rows = html_rows(matched[0] or "")
        if not rows:
            continue
        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]
        tables.append({
            "page_index": page_index,
            "bbox": {"x0": round(x0 * sx, 2), "y0": round(y0 * sy, 2),
                     "x1": round(x1 * sx, 2), "y1": round(y1 * sy, 2)},
            "rows": rows,
            "row_count": len(rows),
            "col_count": width,
        })
    return tables


def read_page(source, index: int, dpi: int) -> list[dict]:
    """One OCR-queue unit: render page ``index`` and read its tables.

    A page whose recognition fails reports no table (logged); a broken
    model/package configuration fails the request.
    """
    from app.services import rapid_ocr

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
    try:
        ocr_engine = rapid_ocr._get_engine()
        _get_engine()
    except MemoryError:
        raise
    except Exception as exc:
        raise rapid_ocr.RapidConfigurationError(str(exc)) from exc
    try:
        return read_tables(bgr, width, height, index, ocr_engine)
    except MemoryError:
        raise
    except Exception as exc:
        log.warning("rapidtable_page_failed", page_index=index, error_type=type(exc).__name__)
        return []
