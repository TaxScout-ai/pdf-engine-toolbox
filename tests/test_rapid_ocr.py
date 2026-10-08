"""Pilot contract and resource boundary tests. No real tax data or model loads."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import fitz
import numpy as np
import pytest

from app.config import Settings, settings
from app.services import pdf_service, rapid_ocr, task_service


def scan(rotation=0, width=612, height=792):
    with fitz.open() as doc:
        page = doc.new_page(width=width, height=height)
        page.set_rotation(rotation)
        return doc.tobytes()


def prediction():
    polygon = np.array([[30, 60], [330, 60], [330, 120], [30, 120]], dtype=float)
    return SimpleNamespace(
        txts=("12,345.67",),
        boxes=np.array([polygon]),
        word_results=[[("12,345.67", 0.98, polygon)]],
    )


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path / "cache"))
    monkeypatch.setattr(settings, "ocr_backend", "rapid")


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_words_use_visible_page_coordinates_and_search_pdf(monkeypatch, rotation):
    engine = MagicMock(return_value=prediction())
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: engine)
    result = pdf_service.ocr_pages(scan(rotation), backend="rapid")
    word = result["pages"][0]["words"][0]
    assert word["bbox"] == pytest.approx({"x": 10, "y": 20, "w": 100, "h": 20})
    assert word["polygon"][0] == [10, 20]
    assert result["provenance"]["render_dpi"] == 216
    assert word["geometry_kind"] == "ctc_word_estimate"
    with fitz.open(stream=result["pdf_bytes"], filetype="pdf") as doc:
        assert "12,345.67" in doc[0].get_text()
        assert doc[0].rotation == rotation
    assert engine.call_count == 1
    # A shared line/word polygon must stay in pixels for the search layer.
    assert engine.return_value.boxes[0][0].tolist() == [30, 60]
    assert engine.return_value.word_results[0][0][2][0].tolist() == [30, 60]


def test_native_pdf_does_not_initialize_either_engine(monkeypatch):
    fail = MagicMock(side_effect=AssertionError("unexpected engine load"))
    monkeypatch.setattr(rapid_ocr, "_get_engine", fail)
    monkeypatch.setattr(pdf_service, "_get_paddle_ocr", fail)
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text(
            (30, 50),
            "Existing selectable document content with sufficiently long words "
            "for native extraction.",
        )
        data = doc.tobytes()
    for backend in ["rapid", "paddle"]:
        result = pdf_service.ocr_pages(data, backend=backend)
        assert result["pages"][0]["words"] == []
        assert "selectable" in result["pages"][0]["full_text"]
    fail.assert_not_called()


def test_sensitive_default_stays_paddle_and_rapid_force_is_rejected(monkeypatch):
    paddle = MagicMock()
    paddle.predict.return_value = []
    rapid = MagicMock(side_effect=AssertionError("sensitive path used Rapid"))
    monkeypatch.setattr(pdf_service, "_get_paddle_ocr", lambda lang: paddle)
    monkeypatch.setattr(rapid_ocr, "_get_engine", rapid)
    pdf_service.ocr_pages(scan(), force_ocr=True)
    assert paddle.predict.call_count == 1
    with pytest.raises(ValueError, match="identifier redaction"):
        pdf_service.ocr_pages(scan(), force_ocr=True, backend="rapid")
    rapid.assert_not_called()


def test_route_cache_keeps_geometry_provenance_and_separates_backends(
    client, auth_headers, monkeypatch
):
    from app.services import download_service

    engine = MagicMock(return_value=prediction())
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: engine)
    monkeypatch.setattr(download_service, "download_pdf", AsyncMock(return_value=scan()))
    body = json.dumps({"source_url": "https://example.com/scan.pdf"})

    def request():
        return client.post(
            "/text/ocr", content=body, headers=auth_headers("POST", "/text/ocr", body)
        ).json()

    first = request()
    result = task_service.get_task(first["task_id"]).result
    cached = request()["data"]
    assert cached["pages"][0]["words"][0]["polygon"] == result["pages"][0]["words"][0]["polygon"]
    assert cached["provenance"]["profile"] == rapid_ocr.PROFILE
    assert engine.call_count == 1
    monkeypatch.setattr(settings, "ocr_backend", "paddle")
    paddle = MagicMock()
    paddle.predict.return_value = []
    monkeypatch.setattr(pdf_service, "_get_paddle_ocr", lambda lang: paddle)
    assert "task_id" in request()
    assert paddle.predict.call_count == 1


def test_inference_error_fails_task_without_paddle_fallback(client, auth_headers, monkeypatch):
    from app.services import download_service

    monkeypatch.setattr(download_service, "download_pdf", AsyncMock(return_value=scan()))
    monkeypatch.setattr(
        rapid_ocr, "_get_engine", MagicMock(side_effect=RuntimeError("model unavailable"))
    )
    paddle = MagicMock(side_effect=AssertionError("unexpected fallback"))
    monkeypatch.setattr(pdf_service, "_get_paddle_ocr", paddle)
    body = json.dumps({"source_url": "https://example.com/scan.pdf"})
    accepted = client.post(
        "/text/ocr", content=body, headers=auth_headers("POST", "/text/ocr", body)
    ).json()
    task = task_service.get_task(accepted["task_id"])
    assert task.status.value == "failed"
    assert task.result is None
    assert "model unavailable" in task.error
    paddle.assert_not_called()
    assert rapid_ocr._lock.acquire(blocking=False)  # failure releases the slot
    rapid_ocr._lock.release()


def test_oversized_page_rejected_before_render_or_engine_load(monkeypatch):
    engine = MagicMock(side_effect=AssertionError("unexpected model load"))
    monkeypatch.setattr(rapid_ocr, "_get_engine", engine)
    with pytest.raises(ValueError, match="pixel limit"):
        rapid_ocr.ocr_pages(scan(width=5000, height=5000))
    engine.assert_not_called()


def test_busy_queue_and_language_fail_explicitly(monkeypatch):
    monkeypatch.setattr(rapid_ocr, "QUEUE_TIMEOUT_SECONDS", 0.001)
    with rapid_ocr._lock:
        with pytest.raises(RuntimeError, match="busy"):
            rapid_ocr.ocr_pages(scan())
    with pytest.raises(ValueError, match="language=en"):
        rapid_ocr.ocr_pages(scan(), language="fr")


def test_bad_model_hash_prevents_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(rapid_ocr, "_engine", None)
    monkeypatch.setattr(settings, "rapidocr_model_dir", str(tmp_path))
    monkeypatch.setattr(rapid_ocr.importlib.metadata, "version", lambda n: rapid_ocr.PACKAGES[n])
    (tmp_path / "PP-OCRv6_det_small.onnx").write_bytes(b"wrong weights")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        rapid_ocr._get_engine()


def test_invalid_geometry_is_not_silently_reported_as_success(monkeypatch):
    result = prediction()
    result.word_results[0][0][2][0][0] = float("nan")
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: lambda *a, **kw: result)
    with pytest.raises(ValueError, match="invalid geometry"):
        rapid_ocr.ocr_pages(scan())


def test_pilot_refuses_multiple_workers():
    with pytest.raises(ValueError, match="WORKERS=1"):
        Settings(ocr_backend="rapid", workers=2, _env_file=None)
