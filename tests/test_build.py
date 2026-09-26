"""Test build (commit) endpoint."""

import json
from unittest.mock import patch, AsyncMock

import fitz


def test_build_with_rotation(client, auth_headers, sample_pdf_bytes):
    """Build should assemble PDF with rotations applied."""
    body = json.dumps({
        "sources": [{
            "url": "https://example.com/test.pdf",
            "pages": [
                {"original_page": 0, "rotation": 90},
                {"original_page": 2, "rotation": 0},
                {"original_page": 4, "rotation": 180},
            ],
        }],
        "flatten_annotations": False,
        "compress": False,
    })
    headers = auth_headers("POST", "/build", body)

    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=sample_pdf_bytes,
    ):
        response = client.post("/build", content=body, headers=headers)

    assert response.status_code == 200
    result_doc = fitz.open(stream=response.content, filetype="pdf")
    assert len(result_doc) == 3
    assert result_doc[0].rotation == 90
    assert result_doc[1].rotation == 0
    assert result_doc[2].rotation == 180


def test_build_with_bookmarks(client, auth_headers, sample_pdf_bytes):
    """Build should add bookmarks to the assembled PDF."""
    body = json.dumps({
        "sources": [{
            "url": "https://example.com/test.pdf",
            "pages": [
                {"original_page": 0, "rotation": 0},
                {"original_page": 1, "rotation": 0},
                {"original_page": 2, "rotation": 0},
            ],
        }],
        "flatten_annotations": False,
        "compress": False,
        "bookmarks": [
            {"label": "W-2", "page": 0, "level": 1},
            {"label": "1099-INT", "page": 1, "level": 1},
            {"label": "1099-DIV", "page": 2, "level": 1},
        ],
    })
    headers = auth_headers("POST", "/build", body)

    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=sample_pdf_bytes,
    ):
        response = client.post("/build", content=body, headers=headers)

    assert response.status_code == 200
    result_doc = fitz.open(stream=response.content, filetype="pdf")
    toc = result_doc.get_toc()
    assert len(toc) == 3
    assert toc[0][1] == "W-2"
    assert toc[1][1] == "1099-INT"


def test_build_rotation_adds_to_the_page_own_rotation(client, auth_headers):
    """TAX-5655: a page that already carries /Rotate keeps it; the working-copy
    rotation is a delta on top, the way the toolbox shows the page."""
    src = fitz.open()
    src.new_page(width=612, height=792).set_rotation(90)
    src.new_page(width=612, height=792).set_rotation(90)
    src_bytes = src.tobytes()
    body = json.dumps({
        "sources": [{
            "url": "https://example.com/test.pdf",
            "pages": [
                {"original_page": 0, "rotation": 90},
                {"original_page": 1, "rotation": 0},
            ],
        }],
        "flatten_annotations": False,
        "compress": False,
    })
    headers = auth_headers("POST", "/build", body)

    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=src_bytes,
    ):
        response = client.post("/build", content=body, headers=headers)

    assert response.status_code == 200
    result_doc = fitz.open(stream=response.content, filetype="pdf")
    assert result_doc[0].rotation == 180
    assert result_doc[1].rotation == 90
