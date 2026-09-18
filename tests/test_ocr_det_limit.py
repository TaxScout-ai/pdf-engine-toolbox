"""TAX-4858: OCR text detection is bounded so a scan cannot OOM the engine.

Unbounded detection on a 300-DPI letter page peaked at 5.7 GB and was
OOM-killed on the 4 GiB production host. The cap is on detection only;
recognition still reads each line from the full-DPI render.
"""

import sys
import types

from app.config import settings
from app.services import pdf_service


def _capture_paddle_kwargs(monkeypatch) -> dict:
    captured: dict = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake = types.ModuleType("paddleocr")
    fake.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake)
    monkeypatch.setattr(pdf_service, "_paddle_ocr_instances", {})
    return captured


def _build(monkeypatch) -> dict:
    captured = _capture_paddle_kwargs(monkeypatch)
    pdf_service._get_paddle_ocr("en")
    return captured


def test_detection_is_capped_by_default(monkeypatch):
    kwargs = _build(monkeypatch)
    assert settings.ocr_det_limit_side_len == 1280
    assert kwargs["text_det_limit_type"] == "max"
    assert kwargs["text_det_limit_side_len"] == 1280
    # Recognition and the rest of the pipeline are unchanged.
    assert kwargs["ocr_version"] == "PP-OCRv5"
    assert kwargs["use_textline_orientation"] is True


def test_zero_restores_the_unbounded_default(monkeypatch):
    monkeypatch.setattr(settings, "ocr_det_limit_side_len", 0)
    kwargs = _build(monkeypatch)
    assert "text_det_limit_type" not in kwargs
    assert "text_det_limit_side_len" not in kwargs
