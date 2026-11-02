"""Test text extraction and search endpoints."""

import json
import hashlib
from unittest.mock import patch, AsyncMock


def test_text_search(client, auth_headers, sample_pdf_bytes):
    """Search should find text matches across pages."""
    body = json.dumps({
        "source_url": "https://example.com/test.pdf",
        "query": "1099-INT",
        "case_sensitive": False,
    })
    headers = auth_headers("POST", "/text/search", body)

    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=sample_pdf_bytes,
    ):
        response = client.post("/text/search", content=body, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["total_matches"] > 0
    assert data["data"]["matches"][0]["page_index"] >= 0


def test_text_extract(client, auth_headers, sample_pdf_bytes):
    """Extract should return text content for pages."""
    body = json.dumps({
        "source_url": "https://example.com/test.pdf",
        "pages": [0, 1],
        "include_positions": True,
    })
    headers = auth_headers("POST", "/text/extract", body)

    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=sample_pdf_bytes,
    ):
        response = client.post("/text/extract", content=body, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["data"]["pages"]) == 2
    assert "Page 1" in data["data"]["pages"][0]["text"]

    receipt = data["data"]["input_provenance"]
    assert receipt["input_sha256"] == hashlib.sha256(sample_pdf_bytes).hexdigest()
    assert receipt["effective_pdf_sha256"] == receipt["input_sha256"]
    assert receipt["page_count"] == 5  # total source, not two selected pages
