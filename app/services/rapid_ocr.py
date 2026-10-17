"""PP-OCRv6 Small CPU backend for review OCR and identifier redaction.

No downloads, second-engine fallback, or claim of field verification. Word
polygons are RapidOCR's CTC-derived estimates in the rendered page coordinate
space, not independently detected character boundaries.

Work is split into one unit per page so the OCR job queue
(app/services/ocr_queue.py) can spread a document over its workers:

* ``read_page`` renders one page and runs inference on the calling worker's
  own engine. It returns a self-contained page result plus the search-layer
  lines for that page, in visual PDF points.
* ``assemble`` puts the page results back in page order, writes the invisible
  search layer and computes coverage, exactly as the single-threaded pass did.

Thread safety: a ``RapidOCR`` object mutates per-call state (``update_params``,
the detector's ``preprocess_op``), so one is never shared between threads;
each worker thread builds its own engine with its own ONNX Runtime sessions.
ONNX Runtime and OpenCV release the GIL while they compute. PyMuPDF does not
promise thread safety, so every PyMuPDF call here holds ``FITZ_LOCK``; each
unit opens its own ``Document``, and no ``Document`` crosses threads.
"""

import importlib.metadata
import math
import os
import threading
import time
from pathlib import Path

import cv2
import fitz
import numpy as np
import structlog

from app.config import settings
from app.services.ocr_preprocessing import map_polygon, prepare_scan, text_layer_quality
from app.services.rapid_ocr_models import MODEL_HASHES, validate_models

log = structlog.get_logger()

PACKAGES = {"rapidocr": "3.9.2", "onnxruntime": "1.30.0"}
# Change this when preprocessing, weights, or output semantics change.
PROFILE = "rapid-3.9.2-ort-1.30.0-v6-small-prepared-word-v6"
MAX_DPI = 216
MAX_RENDER_PIXELS = 16_000_000

# Serializes PyMuPDF work done by OCR workers (render, text layer, assembly).
FITZ_LOCK = threading.RLock()

_local = threading.local()
_engines_lock = threading.Lock()
_engines_created = 0


def available_cpus() -> int:
    """CPUs this process may use: affinity mask, capped by a cgroup v2 quota."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            cpus = min(cpus, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, cpus)


def worker_count() -> int:
    return settings.ocr_workers if settings.ocr_workers > 0 else available_cpus()


def intra_op_threads() -> int:
    if settings.ocr_intra_op_threads > 0:
        return settings.ocr_intra_op_threads
    return max(1, available_cpus() // worker_count())


def cache_profile(backend: str) -> dict:
    if backend == "rapid":
        return {"backend": PROFILE}
    return {"backend": "paddle-v5", "det_limit": settings.ocr_det_limit_side_len}


def validate_runtime():
    """Fail startup on missing/mismatched assets without loading a second OCR engine."""
    for name, expected in PACKAGES.items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f"RapidOCR requires {name}=={expected}")
    validate_models(Path(settings.rapidocr_model_dir))


def _get_engine():
    """Return the calling thread's own engine, building it on first use."""
    engine = getattr(_local, "engine", None)
    if engine is None:
        engine = _local.engine = _new_engine()
    return engine


def _new_engine():
    global _engines_created
    validate_runtime()
    root = Path(settings.rapidocr_model_dir)
    from rapidocr import RapidOCR

    threads = intra_op_threads()
    # Process-wide; workers x threads stays near the CPU count.
    cv2.setNumThreads(threads)
    engine = RapidOCR(
        params={
            "Det.model_path": str(root / "PP-OCRv6_det_small.onnx"),
            "Rec.model_path": str(root / "PP-OCRv6_rec_small.onnx"),
            "Cls.model_path": str(root / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"),
            "Global.text_score": 0.5,
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            "EngineConfig.onnxruntime.use_cuda": False,
        }
    )
    with _engines_lock:
        _engines_created += 1
    log.info("rapidocr_engine_ready", profile=PROFILE, intra_op_threads=threads,
             thread=threading.current_thread().name)
    return engine


def _word(text, score, polygon, pix, page):
    # The provider may share arrays between word and line geometry. Do not
    # rescale its line polygons while normalizing a word for the API.
    points = np.array(polygon, dtype=float, copy=True)
    score = float(score)
    if points.shape != (4, 2) or not np.isfinite(points).all() or not math.isfinite(score):
        raise ValueError("RapidOCR returned invalid geometry or confidence")
    if not text.strip():
        return None
    # Pixel dimensions can round up on fractional PDF page sizes. Scale each
    # axis by its actual render size and clip tiny model overshoots for display.
    points[:, 0] = np.clip(points[:, 0] / pix.width, 0, 1) * page.rect.width
    points[:, 1] = np.clip(points[:, 1] / pix.height, 0, 1) * page.rect.height
    low, high = points.min(axis=0), points.max(axis=0)
    if np.any(high <= low):
        raise ValueError("RapidOCR returned a degenerate word box")
    word = {
        "text": text.strip(),
        "confidence": round(score, 4),
        "bbox": {
            "x": float(low[0]),
            "y": float(low[1]),
            "w": float(high[0] - low[0]),
            "h": float(high[1] - low[1]),
        },
        "polygon": points.tolist(),
        "geometry_kind": "ctc_word_estimate",
    }
    if score < 0.5:
        # Below the review threshold: kept out of the response, but identifier
        # redaction still searches it and counts it as read print.
        word["low_confidence"] = True
    return word


class _Size:
    """Plain page/pixmap dimensions, so geometry runs without PyMuPDF objects."""

    __slots__ = ("width", "height", "rect")

    def __init__(self, width, height):
        self.width, self.height = float(width), float(height)
        self.rect = self


def _page_result(page, index, **values):
    return {
        "page_index": index, "words": [], "full_text": "",
        "visual_width": page.rect.width, "visual_height": page.rect.height,
        "status": "complete", "warnings": [], "discarded_words": 0, **values,
    }


class RapidConfigurationError(RuntimeError):
    """Broken model/package configuration must fail the request, not one page."""


def open_source(source) -> fitz.Document:
    """Open a spilled PDF path or in-memory PDF bytes. Call under FITZ_LOCK."""
    if isinstance(source, (str, Path)):
        return fitz.open(str(source), filetype="pdf")
    from app.services.pdf_service import _open_pdf

    return _open_pdf(source)


def plan(source, pages=None, language="en", dpi=300) -> tuple[list[int], int]:
    """Validate a request and return (page indices, effective DPI).

    Raises the same errors the single-pass reader raised for bad input, so a
    rejected request still fails its task with the same message.
    """
    from app.services.pdf_service import _validate_pages

    if language != "en":
        raise ValueError(
            "RapidOCR pilot supports only language=en; use the Paddle deployment otherwise"
        )
    if dpi <= 0:
        raise ValueError("OCR DPI must be positive")
    with FITZ_LOCK:
        with open_source(source) as doc:
            targets = list(pages) if pages is not None else list(range(len(doc)))
            _validate_pages(doc, targets)
    return targets, min(dpi, MAX_DPI)


def read_page(source, index, dpi, force_ocr=False) -> dict:
    """OCR one page on the calling thread's engine.

    The result carries ``_search_layer``: the recognized lines in visual PDF
    points, written into the PDF later by ``assemble``.
    """
    with FITZ_LOCK:
        with open_source(source) as doc:
            page = doc[index]
            size = _Size(page.rect.width, page.rect.height)
            try:
                prepared = _prepare_page(page, index, dpi, force_ocr)
            except (RapidConfigurationError, MemoryError):
                raise
            except Exception as exc:
                log.warning("rapidocr_page_preparation_failed", page_index=index,
                            error_type=type(exc).__name__)
                return _page_result(size, index, status="failed",
                                    warnings=["page_processing_error"])
    if "result" in prepared:
        return prepared["result"]
    try:
        return _recognize(size, index, **prepared)
    except (RapidConfigurationError, MemoryError):
        raise
    except Exception as exc:
        log.warning("rapidocr_page_preparation_failed", page_index=index,
                    error_type=type(exc).__name__)
        return _page_result(size, index, status="failed", warnings=["page_processing_error"])


def _prepare_page(page, index, dpi, force_ocr):
    """PyMuPDF half of a page: native text, or the rendered pixels to read."""
    existing = page.get_text().strip()
    quality = text_layer_quality(page)
    if quality["accepted"] and not force_ocr:
        words = []
        for x0, y0, x1, y1, text, *_ in page.get_text("words"):
            box = fitz.Rect(x0, y0, x1, y1) * page.rotation_matrix
            box &= page.rect
            if box.is_empty:
                continue
            words.append({"text": text, "confidence": 1.0,
                          "bbox": {"x": box.x0, "y": box.y0, "w": box.width, "h": box.height},
                          "geometry_kind": "native_word"})
        return {"result": _page_result(page, index, words=words, full_text=existing,
                                       text_source="native_pdf", text_layer_quality=quality)}
    zoom = dpi / 72
    if math.ceil(page.rect.width * zoom) * math.ceil(page.rect.height * zoom) > MAX_RENDER_PIXELS:
        raise ValueError("Page exceeds RapidOCR pilot render pixel limit")
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False)
    # Copy out of the pixmap: it is freed when the document closes.
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
    return {"rgb": rgb, "pix": _Size(pix.width, pix.height), "existing": bool(existing),
            "quality": quality}


def _recognize(page, index, rgb, pix, existing, quality):
    """Engine half of a page; runs outside FITZ_LOCK on the worker's own engine."""
    prepared, inverse, preprocessing = prepare_scan(rgb[:, :, ::-1])
    # Pin/hash failures remain fatal configuration errors, not page OCR failures.
    try:
        engine = _get_engine()
    except MemoryError:
        raise
    except Exception as exc:
        raise RapidConfigurationError(str(exc)) from exc
    try:
        prediction = engine(prepared, return_word_box=True, return_single_char_box=False)
    except (MemoryError, KeyboardInterrupt):
        raise
    except Exception as exc:
        # No raw provider error/text in diagnostics. Other pages remain available.
        log.warning("rapidocr_page_failed", page_index=index, error_type=type(exc).__name__)
        return _page_result(page, index, status="failed", warnings=["provider_error"],
                            text_source="rapidocr", text_layer_quality=quality,
                            preprocessing=preprocessing)
    texts = prediction.txts or []
    if not texts:
        # RapidOCR 3.9.2 uses a nested empty sentinel for word_results. It is not
        # a sequence of (text, score, polygon) triples. Empty detection is NOT
        # proof that a source page/field is blank.
        return _page_result(page, index, status="empty", warnings=["no_text_detected"],
                            text_source="rapidocr", text_layer_quality=quality,
                            preprocessing=preprocessing)
    words, low_confidence, warnings, discarded = [], [], set(), 0
    groups = prediction.word_results or []
    if len(groups) != len(texts) or any(not group for group in groups):
        warnings.add("word_geometry_missing")
    for group in groups:
        if not isinstance(group, (list, tuple)):
            warnings.add("word_geometry_missing")
            continue
        for item in group:
            try:
                text, confidence, polygon = item
                word = _word(text, confidence, map_polygon(polygon, inverse), pix, page)
            except (ValueError, TypeError, IndexError):
                discarded += 1
                warnings.add("invalid_word_geometry")
                continue
            if word is None or word.get("low_confidence"):
                discarded += 1
                warnings.add("unusable_word")
                if word is not None:
                    low_confidence.append(word)
                continue
            words.append(word)
    if not words:
        warnings.add("word_geometry_missing")
    # Recognized lines in visual points, whatever their word geometry: RapidOCR
    # sometimes reads a line but splits it into a few fragment "words".
    # Identifier redaction searches these and counts them as read print; they
    # never reach the OCR response.
    lines = []
    for text, polygon in zip(texts, prediction.boxes if prediction.boxes is not None else []):
        try:
            points = np.asarray(map_polygon(polygon, inverse), dtype=float)
            if points.shape != (4, 2) or not np.isfinite(points).all():
                continue
            points[:, 0] = np.clip(points[:, 0] / pix.width, 0, 1) * page.rect.width
            points[:, 1] = np.clip(points[:, 1] / pix.height, 0, 1) * page.rect.height
            low, high = points.min(axis=0), points.max(axis=0)
            if text.strip() and np.all(high > low):
                lines.append({
                    "text": text.strip(),
                    "bbox": {"x": float(low[0]), "y": float(low[1]),
                             "w": float(high[0] - low[0]), "h": float(high[1] - low[1])},
                })
        except (ValueError, TypeError):
            continue
    # A defective searchable line cannot erase otherwise usable word evidence.
    search_layer = []
    if not existing and not preprocessing["rotation_ccw"]:
        for text, polygon in zip(texts, prediction.boxes if prediction.boxes is not None else []):
            try:
                points = np.asarray(map_polygon(polygon, inverse), dtype=float)
                if points.shape != (4, 2) or not np.isfinite(points).all():
                    raise ValueError("Invalid search-layer polygon")
                points[:, 0] *= page.rect.width / pix.width
                points[:, 1] *= page.rect.height / pix.height
                low, high = points.min(axis=0), points.max(axis=0)
                if np.any(high <= low):
                    raise ValueError("Degenerate search-layer polygon")
                search_layer.append([text, float(low[0]), float(low[1]),
                                     float(high[0]), float(high[1])])
            except (ValueError, TypeError, RuntimeError):
                warnings.add("search_layer_incomplete")
    return _page_result(page, index, words=words, full_text=" ".join(texts),
                        text_source="rapidocr", text_layer_quality=quality,
                        preprocessing=preprocessing, discarded_words=discarded,
                        status="partial" if warnings else "complete", warnings=sorted(warnings),
                        _search_layer=search_layer, _low_confidence_words=low_confidence,
                        _lines=lines)


def _write_search_layer(page, result):
    for text, x0, y0, x1, y1 in result.pop("_search_layer", None) or []:
        try:
            font_size = min((y1 - y0) * .8,
                            (x1 - x0) / max(fitz.get_text_length(text, fontsize=1), 1))
            page.insert_text(fitz.Point(x0, y0 + font_size) * page.derotation_matrix,
                             text, fontsize=font_size, rotate=page.rotation,
                             fontname="helv", render_mode=3)
        except (ValueError, TypeError, RuntimeError):
            if "search_layer_incomplete" not in result["warnings"]:
                result["warnings"] = sorted([*result["warnings"], "search_layer_incomplete"])
            result["status"] = "partial"


def assemble(source, results, dpi, with_pdf=True, low_confidence=False) -> dict:
    """Combine page results (in request order) into the response document.

    ``low_confidence`` keeps each page's ``_low_confidence_words`` (words below
    the 0.5 review threshold) and ``_lines`` (recognized lines) for identifier
    redaction; the OCR response the app consumes never carries them.
    """
    results = [dict(r) for r in results]
    for result in results:
        if not low_confidence:
            result.pop("_low_confidence_words", None)
            result.pop("_lines", None)
    pdf_bytes = None
    with FITZ_LOCK:
        with open_source(source) as doc:
            for result in results:
                if with_pdf:
                    _write_search_layer(doc[result["page_index"]], result)
                else:
                    result.pop("_search_layer", None)
            if with_pdf:
                pdf_bytes = doc.tobytes(garbage=4, deflate=True)
    scores = [w["confidence"] for p in results for w in p["words"]]
    coverage = {"requested_pages": len(results),
                **{f"{status}_pages": sum(p["status"] == status for p in results)
                   for status in ["complete", "partial", "empty", "failed"]},
                "discarded_words": sum(p["discarded_words"] for p in results)}
    coverage["status"] = ("complete" if coverage["complete_pages"] == len(results)
                          else "partial" if scores else "unavailable")
    threads = intra_op_threads()
    return {
        "pages": results, "coverage": coverage,
        "total_words": len(scores),
        "avg_confidence": round(sum(scores)/len(scores), 4) if scores else 0,
        "pdf_bytes": pdf_bytes,
        "provenance": {
            "profile": PROFILE, "packages": PACKAGES, "model_sha256": MODEL_HASHES,
            # RapidOCR is the planned production engine (TAX-6876); the field
            # stays, now false, so readers of the old shape keep working.
            "experimental": False, "searchable_output_verified": False,
            "coordinate_space": "visual_pdf_points", "render_dpi": dpi,
            "geometry": "ctc_word_estimate", "intra_op_threads": threads, "inter_op_threads": 1,
            "ocr_workers": worker_count(),
        },
    }


def ocr_pages(pdf_bytes, pages=None, language="en", dpi=300, force_ocr=False, with_pdf=True,
              low_confidence=False):
    """OCR a document through the shared worker pool and wait for it.

    Pages are spread over the OCR workers and share them fairly with queued
    /text/ocr jobs; at most OCR_WORKERS inferences run in the process at once.
    Must not be called from an OCR worker thread.
    """
    from app.services import ocr_queue

    started = time.monotonic()
    try:
        return ocr_queue.get_queue().run_rapid(
            pdf_bytes, pages, language, dpi, force_ocr=force_ocr, with_pdf=with_pdf,
            low_confidence=low_confidence,
        )
    finally:
        log.info(
            "rapidocr_request_finished",
            profile=PROFILE,
            elapsed_ms=round((time.monotonic() - started) * 1000, 2),
        )
