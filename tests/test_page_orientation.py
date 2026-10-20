"""Cardinal page orientation from RapidOCR text lines (TAX-6995).

Unit tests use a stand-in engine; the end-to-end test runs the real pinned
models when they are provisioned (RAPIDOCR_MODEL_DIR) and is skipped
otherwise. The measured comparison with the removed PaddleX classifier is
scripts/orientation_corpus.py.
"""

from pathlib import Path
from types import SimpleNamespace

import cv2
import fitz
import numpy as np
import pytest

from app.config import settings
from app.services import page_orientation as po
from app.services import pdf_service, rapid_ocr
from app.services.rapid_ocr_models import MODEL_HASHES, TABLE_MODEL_HASHES, valid_model


def _quad(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def test_lines_vote_by_length_and_glyph_boxes_do_not_vote():
    boxes = [_quad(0, 0, 300, 20), _quad(0, 40, 200, 60), _quad(0, 80, 20, 100),
             _quad(500, 0, 520, 400)]
    across, down, across_idx, down_idx = po._line_axes(np.array(boxes))
    assert (across, down) == (500, 400)
    assert across_idx == [0, 1] and down_idx == [3]


def test_turned_points_follow_a_clockwise_quarter_turn():
    image = np.zeros((40, 100), dtype=np.uint8)  # 40 rows, 100 columns
    image[5, 70] = 255
    turned = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    (y, x), = np.argwhere(turned == 255)
    assert po._turn_points(np.array([[70.0, 5.0]]), image.shape[0]).tolist() == [[x, y]]


@pytest.mark.parametrize(
    ("pairs", "sideways", "expected"),
    [
        ([(0.95, 0.4)] * 5, False, 0),
        ([(0.3, 0.9)] * 5, False, 180),
        ([(0.95, 0.4)] * 5, True, 90),
        ([(0.3, 0.9)] * 5, True, 270),
    ],
)
def test_direction_maps_to_the_clockwise_correction(pairs, sideways, expected):
    rotation, confidence = po.decide(pairs, 1.0, sideways)
    assert rotation == expected
    assert confidence == 1.0


def test_weak_or_scarce_evidence_is_never_confident():
    # Too few decisive lines: nothing is claimed.
    assert po.decide([(0.9, 0.2), (0.9, 0.2), (0.5, 0.45)], 1.0, False) == (0, 0.0)
    # Readings that are both poor are not evidence for either direction.
    assert po.decide([(0.45, 0.0), (0.0, 0.3), (0.4, 0.1), (0.2, 0.0)], 1.0, True) == (0, 0.0)
    # Split evidence stays well below the app's 0.85 threshold.
    rotation, confidence = po.decide([(0.9, 0.3)] * 3 + [(0.3, 0.9)] * 2, 1.0, False)
    assert confidence < 0.85


class _Engine:
    """Detector boxes as given; the recognizer scores text upright when the
    crop's left half is darker than its right (the test's "reading direction")."""

    def __init__(self, boxes):
        self.boxes = boxes
        self.text_det = lambda image: SimpleNamespace(boxes=self.boxes)

    def text_rec(self, rec_input):
        scores = []
        for crop in rec_input.img:
            half = crop.shape[1] // 2
            left, right = crop[:, :half].mean(), crop[:, half:].mean()
            scores.append(0.95 if left < right else 0.3)
        return SimpleNamespace(scores=scores)


def _page_with_lines(turns: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """An upright page of "lines" (dark start, light end), turned ``turns``
    counter-clockwise quarter turns, and its line boxes."""
    page = np.full((800, 600, 3), 255, dtype=np.uint8)
    boxes = []
    for i in range(6):
        y = 100 + 80 * i
        page[y:y + 20, 100:300] = 40
        page[y:y + 20, 300:500] = 200
        boxes.append(_quad(100, y, 500, y + 20))
    turned = np.ascontiguousarray(np.rot90(page, turns))
    height, width = page.shape[:2]
    mapped = []
    for quad in boxes:
        points = quad.copy()
        for _ in range(turns % 4):  # np.rot90: (x, y) -> (y, W - 1 - x)
            points = np.stack([points[:, 1], width - 1 - points[:, 0]], axis=1)
            width, height = height, width
        width, height = page.shape[1], page.shape[0]
        lo, hi = points.min(axis=0), points.max(axis=0)
        mapped.append(_quad(lo[0], lo[1], hi[0], hi[1]))
    return turned, mapped


@pytest.mark.parametrize(("turns", "correction"), [(0, 0), (1, 90), (2, 180), (3, 270)])
def test_classify_returns_the_correction_for_each_quarter_turn(turns, correction):
    image, boxes = _page_with_lines(turns)
    verdict = po.classify(image, _Engine(np.array(boxes)))
    assert verdict["rotation"] == correction
    assert verdict["confidence"] == 1.0


def test_a_page_without_lines_is_not_judged():
    verdict = po.classify(np.full((200, 200, 3), 255, np.uint8), _Engine(None))
    assert verdict == {"rotation": 0, "confidence": 0.0, "lines": 0}


def test_an_unreadable_page_is_reported_upright_not_failed(monkeypatch):
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: object())
    monkeypatch.setattr(po, "classify", lambda image, engine: 1 / 0)
    doc = fitz.open()
    doc.new_page()
    assert po.read_page(doc.tobytes(), 0) == {"rotation": 0, "confidence": 0.0}


def test_a_broken_model_configuration_fails_the_request(monkeypatch):
    def broken():
        raise RuntimeError("RapidOCR model missing or hash mismatch: x")

    monkeypatch.setattr(rapid_ocr, "_get_engine", broken)
    doc = fitz.open()
    doc.new_page()
    with pytest.raises(rapid_ocr.RapidConfigurationError):
        po.read_page(doc.tobytes(), 0)


def _models_present() -> bool:
    root = Path(settings.rapidocr_model_dir)
    hashes = {**MODEL_HASHES, **TABLE_MODEL_HASHES}
    return all(valid_model(root / name, digest) for name, digest in hashes.items())


@pytest.mark.skipif(not _models_present(), reason="RapidOCR models not provisioned")
def test_real_models_correct_a_turned_synthetic_letter():
    source = fitz.open()
    page = source.new_page(width=612, height=792)
    page.insert_text((72, 80), "Northwind Financial Services", fontsize=16)
    text = (
        "Please review the enclosed summary of your account activity for the year. "
        "Interest and dividends reported to the IRS are listed on the following pages. "
        "Questions about this statement may be directed to customer service. "
    ) * 4
    page.insert_textbox(fitz.Rect(72, 120, 540, 600), text, fontsize=11)
    pix = page.get_pixmap(dpi=150, colorspace=fitz.csGRAY)
    upright = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    scans = fitz.open()
    for turns in range(4):
        image = np.ascontiguousarray(np.rot90(upright, turns))
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        out = scans.new_page(width=image.shape[1] * 72 / 150, height=image.shape[0] * 72 / 150)
        out.insert_image(out.rect, stream=encoded.tobytes())
    result = pdf_service.detect_page_orientations(scans.tobytes())
    assert result["checked_page_count"] == 4
    assert [(p["page"], p["rotation"], p["detected_orientation"]) for p in result["pages"]] == [
        (1, 90, 270), (2, 180, 180), (3, 270, 90)
    ]
