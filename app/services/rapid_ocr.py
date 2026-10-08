"""Bounded, opt-in PP-OCRv6 Small CPU pilot (TAX-6490).

No downloads, second-engine fallback, or claim of field verification. Word
polygons are RapidOCR's CTC-derived estimates in the rendered page coordinate
space, not independently detected character boundaries.
"""

import hashlib
import importlib.metadata
import math
import threading
import time
from pathlib import Path

import cv2
import fitz
import numpy as np
import structlog

from app.config import settings
from app.services.ocr_preprocessing import map_polygon, prepare_scan, text_layer_quality

log = structlog.get_logger()
MODEL_HASHES = {
    "PP-OCRv6_det_small.onnx": "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
    "PP-OCRv6_rec_small.onnx": "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
    "ch_ppocr_mobile_v2.0_cls_mobile.onnx": "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",  # noqa: E501
}
PACKAGES = {"rapidocr": "3.9.2", "onnxruntime": "1.30.0"}
# Change this when preprocessing, weights, or output semantics change.
PROFILE = "rapid-3.9.2-ort-1.30.0-v6-small-prepared-word-v6"
MAX_DPI = 216
MAX_RENDER_PIXELS = 16_000_000
QUEUE_TIMEOUT_SECONDS = 30
_lock = threading.Lock()
_engine = None


def cache_profile(backend: str) -> dict:
    if backend == "rapid":
        return {"backend": PROFILE}
    return {"backend": "paddle-v5", "det_limit": settings.ocr_det_limit_side_len}


def _get_engine():
    """Called only while holding the inference lock; initialize once per process."""
    global _engine
    if _engine is not None:
        return _engine
    for name, expected in PACKAGES.items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f"RapidOCR pilot requires {name}=={expected}")
    root = Path(settings.rapidocr_model_dir)
    for filename, expected in MODEL_HASHES.items():
        path = root / filename
        if not path.is_file():
            raise RuntimeError(f"RapidOCR pilot model missing or hash mismatch: {filename}")
        with path.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(f"RapidOCR pilot model missing or hash mismatch: {filename}")
    from rapidocr import RapidOCR

    cv2.setNumThreads(2)
    _engine = RapidOCR(
        params={
            "Det.model_path": str(root / "PP-OCRv6_det_small.onnx"),
            "Rec.model_path": str(root / "PP-OCRv6_rec_small.onnx"),
            "Cls.model_path": str(root / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"),
            "Global.text_score": 0.5,
            "EngineConfig.onnxruntime.intra_op_num_threads": 2,
            "EngineConfig.onnxruntime.inter_op_num_threads": 2,
            "EngineConfig.onnxruntime.use_cuda": False,
        }
    )
    return _engine


def _word(text, score, polygon, pix, page):
    # The provider may share arrays between word and line geometry. Do not
    # rescale its line polygons while normalizing a word for the API.
    points = np.array(polygon, dtype=float, copy=True)
    score = float(score)
    if points.shape != (4, 2) or not np.isfinite(points).all() or not math.isfinite(score):
        raise ValueError("RapidOCR returned invalid geometry or confidence")
    if not text.strip() or score < 0.5:
        return None
    # Pixel dimensions can round up on fractional PDF page sizes. Scale each
    # axis by its actual render size and clip tiny model overshoots for display.
    points[:, 0] = np.clip(points[:, 0] / pix.width, 0, 1) * page.rect.width
    points[:, 1] = np.clip(points[:, 1] / pix.height, 0, 1) * page.rect.height
    low, high = points.min(axis=0), points.max(axis=0)
    if np.any(high <= low):
        raise ValueError("RapidOCR returned a degenerate word box")
    return {
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


def ocr_pages(pdf_bytes, pages=None, language="en", dpi=300):
    """Process one document at a time per process, one rendered page in memory."""
    if language != "en":
        raise ValueError(
            "RapidOCR pilot supports only language=en; use the Paddle deployment otherwise"
        )
    if not _lock.acquire(timeout=QUEUE_TIMEOUT_SECONDS):
        raise RuntimeError("RapidOCR pilot is busy; retry later")
    started = time.monotonic()
    try:
        return _ocr_pages(pdf_bytes, pages, dpi)
    finally:
        _lock.release()
        log.info(
            "rapidocr_request_finished",
            profile=PROFILE,
            elapsed_ms=round((time.monotonic() - started) * 1000, 2),
        )


def _page_result(page, index, **values):
    return {
        "page_index": index, "words": [], "full_text": "",
        "visual_width": page.rect.width, "visual_height": page.rect.height,
        "status": "complete", "warnings": [], "discarded_words": 0, **values,
    }


class RapidConfigurationError(RuntimeError):
    """Broken model/package configuration must fail the request, not one page."""


def _read_page(page, index, dpi):
    try:
        return _read_page_unchecked(page, index, dpi)
    except (RapidConfigurationError, MemoryError):
        raise
    except Exception as exc:
        log.warning("rapidocr_page_preparation_failed", page_index=index,
                    error_type=type(exc).__name__)
        return _page_result(page, index, status="failed", warnings=["page_processing_error"])


def _read_page_unchecked(page, index, dpi):
    existing = page.get_text().strip()
    quality = text_layer_quality(page)
    if quality["accepted"]:
        words = []
        for x0, y0, x1, y1, text, *_ in page.get_text("words"):
            box = fitz.Rect(x0, y0, x1, y1) * page.rotation_matrix
            box &= page.rect
            if box.is_empty:
                continue
            words.append({"text": text, "confidence": 1.0,
                          "bbox": {"x": box.x0, "y": box.y0, "w": box.width, "h": box.height},
                          "geometry_kind": "native_word"})
        return _page_result(page, index, words=words, full_text=existing,
                            text_source="native_pdf", text_layer_quality=quality)
    zoom = dpi / 72
    if math.ceil(page.rect.width * zoom) * math.ceil(page.rect.height * zoom) > MAX_RENDER_PIXELS:
        raise ValueError("Page exceeds RapidOCR pilot render pixel limit")
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
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
    words, warnings, discarded = [], set(), 0
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
            if word is None:
                discarded += 1
                warnings.add("unusable_word")
                continue
            words.append(word)
    if not words:
        warnings.add("word_geometry_missing")
    # A defective searchable line cannot erase otherwise usable word evidence.
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
                font_size = min((high[1]-low[1])*.8,
                                (high[0]-low[0])/max(fitz.get_text_length(text, fontsize=1), 1))
                page.insert_text(fitz.Point(low[0], low[1]+font_size)*page.derotation_matrix,
                                 text, fontsize=font_size, rotate=page.rotation,
                                 fontname="helv", render_mode=3)
            except (ValueError, TypeError, RuntimeError):
                warnings.add("search_layer_incomplete")
    return _page_result(page, index, words=words, full_text=" ".join(texts),
                        text_source="rapidocr", text_layer_quality=quality,
                        preprocessing=preprocessing, discarded_words=discarded,
                        status="partial" if warnings else "complete", warnings=sorted(warnings))


def _ocr_pages(pdf_bytes, pages, dpi):
    from app.services.pdf_service import _open_pdf, _validate_pages

    if dpi <= 0:
        raise ValueError("OCR DPI must be positive")
    dpi = min(dpi, MAX_DPI)
    with _open_pdf(pdf_bytes) as doc:
        targets = pages if pages is not None else list(range(len(doc)))
        _validate_pages(doc, targets)
        results = [_read_page(doc[index], index, dpi) for index in targets]
        scores = [w["confidence"] for p in results for w in p["words"]]
        coverage = {"requested_pages": len(targets),
                    **{f"{status}_pages": sum(p["status"] == status for p in results)
                       for status in ["complete", "partial", "empty", "failed"]},
                    "discarded_words": sum(p["discarded_words"] for p in results)}
        coverage["status"] = ("complete" if coverage["complete_pages"] == len(targets)
                              else "partial" if scores else "unavailable")
        return {
            "pages": results, "coverage": coverage,
            "total_words": len(scores),
            "avg_confidence": round(sum(scores)/len(scores), 4) if scores else 0,
            "pdf_bytes": doc.tobytes(garbage=4, deflate=True),
            "provenance": {
                "profile": PROFILE, "packages": PACKAGES, "model_sha256": MODEL_HASHES,
                "experimental": True, "searchable_output_verified": False,
                "coordinate_space": "visual_pdf_points", "render_dpi": dpi,
                "geometry": "ctc_word_estimate", "intra_op_threads": 2, "inter_op_threads": 2,
            },
        }
