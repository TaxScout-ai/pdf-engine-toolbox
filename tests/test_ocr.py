"""Test the OCR endpoint (POST /text/ocr) wiring with a stand-in RapidOCR engine.

No model is loaded: ``rapid_ocr._get_engine`` returns a fake whose output has
RapidOCR's shape (``txts``, ``boxes``, ``word_results``). Paddle was removed
from the engine in TAX-6995; RapidOCR is the only backend.
"""

import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import fitz
import numpy as np
import pytest

from app.services import ocr_queue, rapid_ocr

LINES = [
    ("Form W-2", [[150, 150], [600, 150], [600, 225], [150, 225]]),
    ("Wage and Tax Statement", [[150, 270], [900, 270], [900, 345], [150, 345]]),
    ("Tax Year 2024", [[150, 390], [750, 390], [750, 465], [150, 465]]),
]


def _prediction(*_, **__):
    polygons = [np.array(p, dtype=float) for _, p in LINES]
    return SimpleNamespace(
        txts=tuple(t for t, _ in LINES),
        boxes=np.array(polygons),
        word_results=[[(t, 0.97, p)] for (t, _), p in zip(LINES, polygons, strict=True)],
    )


def _create_scanned_pdf():
    """A page with no selectable text, so OCR reads its image."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.draw_rect(fitz.Rect(50, 50, 200, 80), color=(0, 0, 0), fill=(0.9, 0.9, 0.9))
    return doc.tobytes()


def _drain(response):
    """Let the queued job finish while the test's mocks are still in place."""
    task_id = response.json().get("task_id")
    if task_id:
        ocr_queue.get_queue().wait(task_id)


def _completed_task_result(client, auth_headers, accepted: dict) -> dict:
    assert accepted["status"] == "pending"
    path = f"/tasks/{accepted['task_id']}"
    response = client.get(path, headers=auth_headers("GET", path, ""))
    assert response.status_code == 200
    task = response.json()["data"]
    assert task["status"] == "completed", task.get("error")
    return task["result"]


def _post(client, auth_headers, pdf_bytes, payload):
    body = json.dumps(payload)
    headers = auth_headers("POST", "/text/ocr", body)
    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=pdf_bytes,
    ):
        response = client.post("/text/ocr", content=body, headers=headers)
        _drain(response)
    return response


def test_ocr_returns_rich_json(client, auth_headers, monkeypatch):
    """OCR returns words, bboxes, confidence and a searchable base64 PDF."""
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: _prediction)
    response = _post(client, auth_headers, _create_scanned_pdf(), {
        "source_url": "https://example.com/scanned.pdf", "pages": [0],
        "language": "en", "dpi": 300,
    })
    assert response.status_code == 200
    ocr_data = _completed_task_result(client, auth_headers, response.json())

    page = ocr_data["pages"][0]
    assert page["page_index"] == 0
    assert [w["text"] for w in page["words"]] == [t for t, _ in LINES]
    bbox = page["words"][0]["bbox"]
    assert bbox["w"] > 0 and bbox["h"] > 0
    assert page["words"][0]["confidence"] > 0.9
    assert ocr_data["total_words"] == 3
    assert ocr_data["avg_confidence"] > 0.9
    assert base64.b64decode(ocr_data["pdf_base64"])[:5] == b"%PDF-"
    assert "Tax Year 2024" in page["full_text"]
    assert ocr_data["provenance"]["profile"] == rapid_ocr.PROFILE


def test_ocr_skips_pages_with_text(client, auth_headers, monkeypatch):
    """A page with a usable text layer is answered from it; no engine loads."""
    engine = MagicMock(side_effect=AssertionError("engine used on a text page"))
    monkeypatch.setattr(rapid_ocr, "_get_engine", engine)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text(
        fitz.Point(72, 72),
        "This is an existing text page with lots of content that should not need OCR.",
        fontsize=12,
    )
    response = _post(client, auth_headers, doc.tobytes(), {
        "source_url": "https://example.com/text.pdf", "pages": [0],
    })
    page_result = _completed_task_result(client, auth_headers, response.json())["pages"][0]
    assert page_result["text_source"] == "native_pdf"
    assert "existing text page" in page_result["full_text"]
    engine.assert_not_called()


def test_ocr_large_doc_becomes_background_task(client, auth_headers, monkeypatch):
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: _prediction)
    doc = fitz.open()
    for _ in range(10):
        doc.new_page(width=612, height=792)
    response = _post(client, auth_headers, doc.tobytes(), {
        "source_url": "https://example.com/large.pdf",
    })
    data = response.json()
    assert data["success"] is True
    assert data["status"] == "pending"
    assert "10 pages" in data["message"]


@pytest.mark.parametrize("language", ["fr", "ch"])
def test_ocr_other_languages_fail_the_task_with_the_reason(
    client, auth_headers, monkeypatch, language
):
    """Only English is read; another language fails its task, never silently."""
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: _prediction)
    response = _post(client, auth_headers, _create_scanned_pdf(), {
        "source_url": "https://example.com/doc.pdf", "pages": [0], "language": language,
    })
    path = f"/tasks/{response.json()['task_id']}"
    task = client.get(path, headers=auth_headers("GET", path, "")).json()["data"]
    assert task["status"] == "failed"
    assert "language=en" in task["error"]
