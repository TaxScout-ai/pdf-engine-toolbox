"""Test PaddleOCR endpoint (POST /text/ocr).

PaddleOCR is a heavy dependency (~150 MB models), so these tests mock
the PaddleOCR engine and verify the request/response wiring, response
format, and background task logic.

Tests target PaddleOCR 3.x (.predict() API, PP-OCRv5).
"""

import base64
import json
from unittest.mock import patch, AsyncMock, MagicMock

import fitz
import numpy as np


def _create_scanned_pdf():
    """Create a PDF with an image page (no selectable text) to trigger OCR."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    # Draw a filled rectangle to simulate a scanned page with some "content"
    rect = fitz.Rect(50, 50, 200, 80)
    page.draw_rect(rect, color=(0, 0, 0), fill=(0.9, 0.9, 0.9))
    return doc.tobytes()


def _mock_paddle_result():
    """Return a PaddleOCR 3.x .predict()-shaped result for one page.

    PaddleOCR 3.x returns an iterable of result objects.  Each result
    has a ``.json`` property returning a dict with rec_texts, rec_scores,
    dt_polys, etc.
    """
    result_obj = MagicMock()
    result_obj.json = {
        "rec_texts": ["Form W-2", "Wage and Tax Statement", "Tax Year 2024"],
        "rec_scores": [0.97, 0.95, 0.92],
        "dt_polys": [
            # 4-point polygons in image pixels
            [[50.0, 50.0], [200.0, 50.0], [200.0, 75.0], [50.0, 75.0]],
            [[50.0, 90.0], [300.0, 90.0], [300.0, 115.0], [50.0, 115.0]],
            [[50.0, 130.0], [250.0, 130.0], [250.0, 155.0], [50.0, 155.0]],
        ],
    }
    return [result_obj]


def _mock_paddle_empty_result():
    """Return an empty PaddleOCR 3.x result (no text detected)."""
    result_obj = MagicMock()
    result_obj.json = {
        "rec_texts": [],
        "rec_scores": [],
        "dt_polys": [],
    }
    return [result_obj]


def _completed_task_result(client, auth_headers, accepted: dict) -> dict:
    """Read the completed background result using the signed task endpoint."""
    assert accepted["status"] == "pending"
    path = f"/tasks/{accepted['task_id']}"
    response = client.get(path, headers=auth_headers("GET", path, ""))
    assert response.status_code == 200
    task = response.json()["data"]
    assert task["status"] == "completed"
    return task["result"]


def test_ocr_returns_rich_json(client, auth_headers):
    """OCR should return JSON with words, bboxes, confidence, and base64 PDF."""
    pdf_bytes = _create_scanned_pdf()

    body = json.dumps({
        "source_url": "https://example.com/scanned.pdf",
        "pages": [0],
        "language": "en",
        "dpi": 300,
    })
    headers = auth_headers("POST", "/text/ocr", body)

    mock_paddle = MagicMock()
    mock_paddle.predict.return_value = _mock_paddle_result()

    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=pdf_bytes,
    ), patch(
        "app.services.pdf_service._get_paddle_ocr",
        return_value=mock_paddle,
    ):
        response = client.post("/text/ocr", content=body, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    ocr_data = _completed_task_result(client, auth_headers, data)

    # Check page results
    assert len(ocr_data["pages"]) == 1
    page = ocr_data["pages"][0]
    assert page["page_index"] == 0
    assert len(page["words"]) == 3

    # Check word structure — bbox is {x, y, w, h} rectangle (not polygon)
    w0 = page["words"][0]
    assert w0["text"] == "Form W-2"
    bbox = w0["bbox"]
    assert "x" in bbox and "y" in bbox and "w" in bbox and "h" in bbox
    assert bbox["w"] > 0  # Non-zero width
    assert bbox["h"] > 0  # Non-zero height
    assert w0["confidence"] > 0.9

    # Check aggregates
    assert ocr_data["total_words"] == 3
    assert ocr_data["avg_confidence"] > 0.9

    # Check base64 PDF is present and decodable
    assert "pdf_base64" in ocr_data
    pdf_decoded = base64.b64decode(ocr_data["pdf_base64"])
    assert pdf_decoded[:5] == b"%PDF-"

    # Check full_text
    assert "Form W-2" in page["full_text"]
    assert "Tax Year 2024" in page["full_text"]


def test_ocr_skips_pages_with_text(client, auth_headers):
    """Pages that already have selectable text should be skipped (returned with empty words)."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    # This page has plenty of text already
    page.insert_text(fitz.Point(72, 72), "This is an existing text page with lots of content that should not need OCR processing at all.", fontsize=12)
    pdf_bytes = doc.tobytes()

    body = json.dumps({
        "source_url": "https://example.com/text.pdf",
        "pages": [0],
        "language": "en",
        "dpi": 300,
    })
    headers = auth_headers("POST", "/text/ocr", body)

    mock_paddle = MagicMock()
    # PaddleOCR should NOT be called for text pages

    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=pdf_bytes,
    ), patch(
        "app.services.pdf_service._get_paddle_ocr",
        return_value=mock_paddle,
    ):
        response = client.post("/text/ocr", content=body, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

    page_result = _completed_task_result(client, auth_headers, data)["pages"][0]
    # Should have no OCR words (page had existing text)
    assert len(page_result["words"]) == 0
    # But full_text should contain the existing text
    assert "existing text page" in page_result["full_text"]

    # PaddleOCR.predict should NOT have been called
    mock_paddle.predict.assert_not_called()


def test_ocr_large_doc_becomes_background_task(client, auth_headers):
    """Documents with > 5 pages should go to background task."""
    # Create a 10-page PDF
    doc = fitz.open()
    for _ in range(10):
        doc.new_page(width=612, height=792)
    pdf_bytes = doc.tobytes()

    body = json.dumps({
        "source_url": "https://example.com/large.pdf",
        "language": "en",
        "dpi": 300,
    })
    headers = auth_headers("POST", "/text/ocr", body)

    # Mock both download and OCR processing — TestClient runs background
    # tasks synchronously, so _run_ocr would actually load PaddleOCR models.
    mock_paddle = MagicMock()
    mock_paddle.predict.return_value = _mock_paddle_empty_result()

    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=pdf_bytes,
    ), patch(
        "app.services.pdf_service._get_paddle_ocr",
        return_value=mock_paddle,
    ):
        response = client.post("/text/ocr", content=body, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "task_id" in data
    assert data["status"] == "pending"
    assert "10 pages" in data["message"]


def test_ocr_default_language_is_en(client, auth_headers):
    """Default language should be 'en' (PaddleOCR format, not 'eng')."""
    pdf_bytes = _create_scanned_pdf()

    body = json.dumps({
        "source_url": "https://example.com/doc.pdf",
        "pages": [0],
    })
    headers = auth_headers("POST", "/text/ocr", body)

    mock_paddle = MagicMock()
    mock_paddle.predict.return_value = _mock_paddle_result()

    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=pdf_bytes,
    ), patch(
        "app.services.pdf_service._get_paddle_ocr",
        return_value=mock_paddle,
    ) as mock_get_paddle:
        response = client.post("/text/ocr", content=body, headers=headers)

    assert response.status_code == 200
    # Verify PaddleOCR was initialised with 'en'
    mock_get_paddle.assert_called_with("en")
