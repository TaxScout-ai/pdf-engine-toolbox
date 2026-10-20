"""Cardinal page orientation from RapidOCR text lines (TAX-6995).

Replaces the PaddleX ``PP-LCNet_x1_0_doc_ori`` classifier that
/transform/orientation ran before Paddle left the engine. Nothing here is a
new model: it uses the pinned RapidOCR weights the engine already ships.

A page is judged in two independent steps:

1. **Axis.** The text detector finds text lines on the page as rendered. Long,
   thin boxes are lines; their long side says whether lines run across the
   page (upright or upside down) or down it (turned 90 or 270 degrees). Each
   line votes with its length. Square-ish boxes (single glyphs, check boxes)
   do not vote.
2. **Direction.** The page is turned so its lines run across, then the
   longest lines are read twice by the recognizer: as they are, and turned
   180 degrees. Text the right way up reads with clearly higher confidence
   than the same text upside down. A line votes when one of its readings is
   credible and beats the other by a clear margin; its vote weighs that
   margin.

Confidence is the product of the share of line length on the winning axis and
the (margin-weighted) share of the direction vote for the winning direction. A page with too few
lines or too few decisive readings gets confidence 0 and is never turned:
a missed rotation leaves the page as the scanner delivered it, a false one
turns a good page.

The rotation returned is the clockwise correction that makes the page
upright, the same contract as the PaddleX classifier: ``detected_orientation``
is the classifier's view of the page, and ``rotation`` = (360 - it) % 360.
"""

from __future__ import annotations

import cv2
import numpy as np
import structlog

log = structlog.get_logger()

# Render resolution of the page image (dots per inch). Lines are read from it.
DPI = 144
# Lines are found on the page scaled so its longer side is at most this; text
# lines stay several pixels tall on a letter page and detection is ~4x cheaper.
DETECT_SIDE = 1024
# A line is read over at most this many times its height (its start); enough
# glyphs to tell the way up, without paying for a whole long line twice.
MAX_READ_ASPECT = 16
# A box is a line when its long side is at least this many times its short one.
LINE_ASPECT = 2.0
# Lines read twice (as is, and turned 180 degrees) to decide the direction.
MAX_READ_LINES = 32
# A reading pair votes only when its confidences differ by at least this much
# and the better reading scores at least CREDIBLE_SCORE. Upside-down text
# typically reads at 0.3-0.6; upright text above 0.8.
DECISIVE_MARGIN = 0.15
CREDIBLE_SCORE = 0.5
# Below this many lines on the page, or this many decisive readings, the page
# is not judged at all.
MIN_LINES = 3
# Fewer lines than this at reduced size triggers a full-resolution look.
MIN_FAST_LINES = 8
MIN_DECISIVE = 3


def _line_axes(boxes: np.ndarray) -> tuple[float, float, list[int], list[int]]:
    """Length-weighted votes for (across, down) and the indices of each."""
    across = down = 0.0
    across_idx: list[int] = []
    down_idx: list[int] = []
    for index, quad in enumerate(boxes):
        quad = np.asarray(quad, dtype=float)
        width = float(np.linalg.norm(quad[1] - quad[0]))
        height = float(np.linalg.norm(quad[3] - quad[0]))
        if width >= height * LINE_ASPECT:
            across += width
            across_idx.append(index)
        elif height >= width * LINE_ASPECT:
            down += height
            down_idx.append(index)
    return across, down, across_idx, down_idx


def _turn_points(points: np.ndarray, width: int) -> np.ndarray:
    """Map (x, y) of an image of ``width`` to the image turned 90 degrees clockwise.

    cv2.ROTATE_90_CLOCKWISE sends pixel (x, y) of an H-row image to
    (H - 1 - y, x); ``width`` here is that H.
    """
    turned = np.empty_like(points)
    turned[:, 0] = width - 1 - points[:, 1]
    turned[:, 1] = points[:, 0]
    return turned


def _crops(image: np.ndarray, quads: list[np.ndarray]) -> list[np.ndarray]:
    height, width = image.shape[:2]
    crops = []
    for quad in quads:
        x0, y0 = np.floor(quad.min(axis=0)).astype(int) - 2
        x1, y1 = np.ceil(quad.max(axis=0)).astype(int) + 2
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, width), min(y1, height)
        x1 = min(x1, x0 + (y1 - y0) * MAX_READ_ASPECT)
        if x1 - x0 >= 8 and y1 - y0 >= 6:
            crops.append(np.ascontiguousarray(image[y0:y1, x0:x1]))
    return crops


def _detect(image_bgr: np.ndarray, engine, side: int | None) -> np.ndarray:
    """Text-line quads in ``image_bgr`` pixels, detected at most ``side`` pixels long."""
    scale = 1.0 if side is None else min(1.0, side / max(image_bgr.shape[:2]))
    small = image_bgr
    if scale < 1.0:
        small = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    det = engine.text_det(small)
    if det.boxes is None or len(det.boxes) == 0:
        return np.zeros((0, 4, 2))
    return np.asarray(det.boxes, dtype=float) / scale


def decide(
    pairs: list[tuple[float, float]], axis_share: float, sideways: bool
) -> tuple[int, float]:
    """(clockwise correction, confidence) from per-line (as is, turned) scores."""
    # A line is evidence only when one reading is credible text and clearly
    # beats the other; two poor readings say nothing about the way up.
    margins = [
        a - b for a, b in pairs
        if abs(a - b) >= DECISIVE_MARGIN and max(a, b) >= CREDIBLE_SCORE
    ]
    if len(margins) < MIN_DECISIVE:
        return 0, 0.0
    up = sum(m for m in margins if m > 0)
    down = -sum(m for m in margins if m < 0)
    flip = down > up
    direction_share = max(up, down) / (up + down)
    if sideways:
        # Upright after a clockwise quarter turn (90), or after that turn and
        # a half turn (270).
        rotation = 270 if flip else 90
    else:
        rotation = 180 if flip else 0
    return rotation, round(axis_share * direction_share, 4)


def classify(image_bgr: np.ndarray, engine) -> dict:
    """Clockwise correction (0/90/180/270) and confidence for one page image.

    ``engine`` is a RapidOCR instance; only its detector and recognizer are
    used. The caller owns the engine and must not share it between threads.
    """
    from rapidocr.ch_ppocr_rec import TextRecInput

    boxes = _detect(image_bgr, engine, DETECT_SIDE)
    across, down, across_idx, down_idx = _line_axes(boxes)
    if len(across_idx) + len(down_idx) < MIN_FAST_LINES:
        # Small print vanishes when the page is scaled down: look again at
        # full resolution before giving up on the page.
        boxes = _detect(image_bgr, engine, None)
        across, down, across_idx, down_idx = _line_axes(boxes)
    result = {"rotation": 0, "confidence": 0.0, "lines": len(across_idx) + len(down_idx)}
    if result["lines"] < MIN_LINES or across + down <= 0:
        return result

    sideways = down > across
    axis_share = max(across, down) / (across + down)
    chosen = [boxes[i] for i in (down_idx if sideways else across_idx)]
    image = image_bgr
    if sideways:
        # Turn the page so its lines run across; boxes follow.
        chosen = [_turn_points(quad, image.shape[0]) for quad in chosen]
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    # Longest lines first: they carry the most evidence per reading.
    chosen.sort(key=lambda q: float(q[:, 0].max() - q[:, 0].min()), reverse=True)
    crops = _crops(image, chosen[:MAX_READ_LINES])
    if len(crops) < MIN_DECISIVE:
        return result

    # One batch: every line as is, then every line turned a half turn.
    read = engine.text_rec(
        TextRecInput(img=crops + [cv2.rotate(c, cv2.ROTATE_180) for c in crops])
    )
    scores = [float(score) for score in read.scores]
    pairs = list(zip(scores[: len(crops)], scores[len(crops):], strict=True))
    rotation, confidence = decide(pairs, axis_share, sideways)
    result.update(rotation=rotation, confidence=confidence, sideways=sideways,
                  axis_share=round(axis_share, 4), pairs=pairs)
    return result


def read_page(source, index: int) -> dict:
    """One OCR-queue unit: render page ``index`` and judge its orientation.

    Runs on an OCR worker with that worker's own RapidOCR engine. A page that
    cannot be read is reported upright with confidence 0, never turned; a
    broken model/package configuration fails the request.
    """
    import fitz

    from app.services import rapid_ocr

    with rapid_ocr.FITZ_LOCK:
        with rapid_ocr.open_source(source) as doc:
            page = doc[index]
            area = max(page.rect.width * page.rect.height, 1.0)
            zoom = min(DPI / 72, (rapid_ocr.MAX_RENDER_PIXELS / area) ** 0.5)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB,
                                  alpha=False)
            rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    try:
        engine = rapid_ocr._get_engine()
    except MemoryError:
        raise
    except Exception as exc:
        raise rapid_ocr.RapidConfigurationError(str(exc)) from exc
    try:
        verdict = classify(bgr, engine)
    except MemoryError:
        raise
    except Exception as exc:  # one unreadable page never fails the document
        log.warning("orientation_page_failed", page_index=index, error_type=type(exc).__name__)
        return {"rotation": 0, "confidence": 0.0}
    return {"rotation": verdict["rotation"], "confidence": verdict["confidence"]}
