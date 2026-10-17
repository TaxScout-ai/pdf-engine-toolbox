"""Pilot contract and resource boundary tests. No real tax data or model loads."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import fitz
import numpy as np
import pytest

from app.config import Settings, settings
from app.services import ocr_queue, pdf_service, rapid_ocr, task_service


def finished_task(task_id):
    """The task once the OCR queue has finished it."""
    ocr_queue.get_queue().wait(task_id)
    return task_service.get_task(task_id)


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
        if backend == "paddle":
            assert result["pages"][0]["words"] == []
        else:
            assert result["pages"][0]["words"][0]["geometry_kind"] == "native_word"
        assert "selectable" in result["pages"][0]["full_text"]
    fail.assert_not_called()


def test_force_ocr_reads_the_page_image_on_either_backend(monkeypatch):
    """Identifier redaction must read what a page shows, not its text layer."""
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text(
            (30, 50),
            "Existing selectable document content with sufficiently long words "
            "for native extraction.",
        )
        data = doc.tobytes()
    paddle = MagicMock()
    paddle.predict.return_value = []
    engine = MagicMock(return_value=prediction())
    monkeypatch.setattr(pdf_service, "_get_paddle_ocr", lambda lang: paddle)
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: engine)
    pdf_service.ocr_pages(data, force_ocr=True)
    assert paddle.predict.call_count == 1
    result = pdf_service.ocr_pages(data, force_ocr=True, backend="rapid")
    assert engine.call_count == 1
    assert result["pages"][0]["text_source"] == "rapidocr"
    assert result["pages"][0]["words"][0]["text"] == "12,345.67"


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
    result = finished_task(first["task_id"]).result
    cached = request()["data"]
    assert cached["pages"][0]["words"][0]["polygon"] == result["pages"][0]["words"][0]["polygon"]
    assert cached["provenance"]["profile"] == rapid_ocr.PROFILE
    assert engine.call_count == 1
    monkeypatch.setattr(settings, "ocr_backend", "paddle")
    paddle = MagicMock()
    paddle.predict.return_value = []
    monkeypatch.setattr(pdf_service, "_get_paddle_ocr", lambda lang: paddle)
    queued = request()
    assert "task_id" in queued
    assert finished_task(queued["task_id"]).status.value == "completed"
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
    task = finished_task(accepted["task_id"])
    assert task.status.value == "failed"
    assert task.result is None
    assert "model unavailable" in task.error
    paddle.assert_not_called()
    # A failed job frees its worker: the next request is served.
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: lambda *a, **kw: prediction())
    with fitz.open() as doc:
        doc.new_page(width=300)
        other = doc.tobytes()
    monkeypatch.setattr(download_service, "download_pdf", AsyncMock(return_value=other))
    accepted = client.post(
        "/text/ocr", content=body, headers=auth_headers("POST", "/text/ocr", body)
    ).json()
    assert finished_task(accepted["task_id"]).status.value == "completed"


def test_oversized_page_rejected_before_render_or_engine_load(monkeypatch):
    engine = MagicMock(side_effect=AssertionError("unexpected model load"))
    monkeypatch.setattr(rapid_ocr, "_get_engine", engine)
    result = rapid_ocr.ocr_pages(scan(width=5000, height=5000))
    assert result["pages"][0]["status"] == "failed"
    assert result["coverage"]["status"] == "unavailable"
    engine.assert_not_called()


def test_unsupported_language_fails_explicitly(monkeypatch):
    with pytest.raises(ValueError, match="language=en"):
        rapid_ocr.ocr_pages(scan(), language="fr")


def test_bad_model_hash_prevents_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "rapidocr_model_dir", str(tmp_path))
    monkeypatch.setattr(rapid_ocr.importlib.metadata, "version", lambda n: rapid_ocr.PACKAGES[n])
    (tmp_path / "PP-OCRv6_det_small.onnx").write_bytes(b"wrong weights")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        rapid_ocr._new_engine()


def test_invalid_geometry_is_not_silently_reported_as_success(monkeypatch):
    result = prediction()
    result.word_results[0][0][2][0][0] = float("nan")
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: lambda *a, **kw: result)
    response = rapid_ocr.ocr_pages(scan())
    assert response["coverage"]["status"] == "unavailable"
    assert response["pages"][0]["status"] == "partial"
    assert response["pages"][0]["discarded_words"] == 1
    assert response["pages"][0]["words"] == []


def test_pilot_refuses_multiple_workers():
    with pytest.raises(ValueError, match="WORKERS=1"):
        Settings(ocr_backend="rapid", workers=2, _env_file=None)


def test_empty_detector_sentinel_preserves_other_pages(monkeypatch):
    # The fake answers in call order; one worker keeps that order page-wise.
    monkeypatch.setattr(settings, "ocr_workers", 1)
    empty = SimpleNamespace(txts=None, boxes=None, word_results=[[[], [], []]])
    engine = MagicMock(side_effect=[prediction(), empty, prediction()])
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: engine)
    with fitz.open() as doc:
        for _ in range(3): doc.new_page()
        result = rapid_ocr.ocr_pages(doc.tobytes())
    assert [p["status"] for p in result["pages"]] == ["complete", "empty", "complete"]
    assert [len(p["words"]) for p in result["pages"]] == [1, 0, 1]
    assert result["pages"][1]["warnings"] == ["no_text_detected"]
    assert result["coverage"]["status"] == "partial"
    assert result["coverage"]["empty_pages"] == 1


def test_degenerate_word_is_excluded_without_losing_valid_word(monkeypatch):
    result = prediction()
    polygon = np.zeros((4, 2))
    result.word_results[0].append(("bad", .99, polygon))
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: lambda *a, **kw: result)
    response = rapid_ocr.ocr_pages(scan())
    assert len(response["pages"][0]["words"]) == 1
    assert response["pages"][0]["status"] == "partial"
    assert response["coverage"]["discarded_words"] == 1
    assert response["coverage"]["status"] == "partial"


def test_provider_failure_isolated_from_following_page(monkeypatch):
    # The fake answers in call order; one worker keeps that order page-wise.
    monkeypatch.setattr(settings, "ocr_workers", 1)
    engine = MagicMock(side_effect=[RuntimeError("private provider detail"), prediction()])
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: engine)
    with fitz.open() as doc:
        doc.new_page(); doc.new_page()
        result = rapid_ocr.ocr_pages(doc.tobytes())
    assert [p["status"] for p in result["pages"]] == ["failed", "complete"]
    assert result["pages"][0]["warnings"] == ["provider_error"]
    assert "private provider detail" not in str(result)
    assert result["total_words"] == 1


def test_partial_coverage_survives_async_and_cached_routes(client, auth_headers, monkeypatch):
    from app.services import download_service
    result = prediction()
    result.word_results[0].append(("bad", .9, np.zeros((4, 2))))
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: lambda *a, **kw: result)
    monkeypatch.setattr(download_service, "download_pdf", AsyncMock(return_value=scan()))
    body = json.dumps({"source_url": "https://example.com/scan.pdf"})
    first = client.post("/text/ocr", content=body, headers=auth_headers("POST", "/text/ocr", body)).json()
    async_data = finished_task(first["task_id"]).result
    cached = client.post("/text/ocr", content=body, headers=auth_headers("POST", "/text/ocr", body)).json()["data"]
    assert cached["coverage"] == async_data["coverage"]
    assert cached["pages"][0]["status"] == "partial"
    assert cached["pages"][0]["discarded_words"] == 1


def test_missing_line_geometry_marks_page_partial(monkeypatch):
    result = prediction()
    result.txts = ["first line", "second line"]
    result.word_results.append([])
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: lambda *a, **kw: result)
    response = rapid_ocr.ocr_pages(scan())
    assert response["coverage"]["status"] == "partial"
    assert "word_geometry_missing" in response["pages"][0]["warnings"]


def test_oversized_page_does_not_erase_previous_page(monkeypatch):
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: lambda *a, **kw: prediction())
    with fitz.open() as doc:
        doc.new_page(); doc.new_page(width=5000, height=5000)
        response = rapid_ocr.ocr_pages(doc.tobytes())
    assert [p["status"] for p in response["pages"]] == ["complete", "failed"]
    assert response["coverage"]["status"] == "partial"
