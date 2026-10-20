"""Test transform endpoints."""

import json
from unittest.mock import patch, AsyncMock

import fitz

from app.services import pdf_service


def test_compress_pdf(client, auth_headers, sample_pdf_bytes):
    """Compress should return a valid (possibly smaller) PDF."""
    body = json.dumps({
        "source_url": "https://example.com/test.pdf",
        "quality": "medium",
        "max_image_dpi": 150,
    })
    headers = auth_headers("POST", "/transform/compress", body)

    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=sample_pdf_bytes,
    ):
        response = client.post("/transform/compress", content=body, headers=headers)

    assert response.status_code == 200
    result_doc = fitz.open(stream=response.content, filetype="pdf")
    assert len(result_doc) == 5  # Same page count


def test_detect_page_orientation_endpoint(
    client, auth_headers, sample_pdf_bytes
):
    """Orientation endpoint returns page corrections without mutating bytes."""
    body = json.dumps({
        "source_url": "https://example.com/test.pdf",
        "scanned_only": True,
        "min_confidence": 0.9,
    })
    headers = auth_headers("POST", "/transform/orientation", body)
    detected = {
        "pages": [
            {
                "page": 0,
                "rotation": 90,
                "detected_orientation": 270,
                "confidence": 0.97,
            }
        ],
        "checked_page_count": 1,
        "skipped_page_count": 4,
    }

    with (
        patch(
            "app.services.download_service.download_pdf",
            new_callable=AsyncMock,
            return_value=sample_pdf_bytes,
        ),
        patch(
            "app.services.pdf_service.detect_page_orientations",
            return_value=detected,
        ) as detector,
    ):
        response = client.post(
            "/transform/orientation", content=body, headers=headers
        )

    assert response.status_code == 200
    assert response.json() == detected
    detector.assert_called_once_with(
        sample_pdf_bytes,
        None,
        scanned_only=True,
        min_confidence=0.9,
    )


def test_detect_page_orientations_filters_low_confidence(
    monkeypatch, sample_pdf_bytes
):
    """Only confident non-zero cardinal corrections leave the service.

    ``detected_orientation`` keeps the PaddleX classifier's meaning (TAX-6995):
    the page's orientation, whose clockwise correction is ``rotation``.
    """
    from app.services import page_orientation

    verdicts = [
        {"rotation": 90, "confidence": 0.97},
        {"rotation": 270, "confidence": 0.70},
        {"rotation": 0, "confidence": 0.99},
        {"rotation": 180, "confidence": 0.91},
        {"rotation": 0, "confidence": 0.0},
    ]
    seen = []

    def judge(source, index):
        seen.append(index)
        return verdicts[index]

    monkeypatch.setattr(page_orientation, "read_page", judge)

    result = pdf_service.detect_page_orientations(
        sample_pdf_bytes,
        scanned_only=False,
        min_confidence=0.85,
    )

    assert result == {
        "pages": [
            {
                "page": 0,
                "rotation": 90,
                "detected_orientation": 270,
                "confidence": 0.97,
            },
            {
                "page": 3,
                "rotation": 180,
                "detected_orientation": 180,
                "confidence": 0.91,
            },
        ],
        "checked_page_count": 5,
        "skipped_page_count": 0,
    }
    assert sorted(seen) == [0, 1, 2, 3, 4]


def test_flatten_annotations(client, auth_headers, sample_pdf_bytes):
    """Flatten should burn annotations into PDF."""
    body = json.dumps({
        "source_url": "https://example.com/test.pdf",
        "annotations": [
            {
                "page_number": 1,
                "type": "stamp",
                "x": 50.0,
                "y": 50.0,
                "stamp_type": "VERIFIED",
                "color": "#4CAF50",
            },
            {
                "page_number": 1,
                "type": "highlight",
                "x": 10.0,
                "y": 15.0,
                "width": 40.0,
                "height": 2.0,
                "color": "#FFEB3B",
            },
        ],
    })
    headers = auth_headers("POST", "/transform/flatten", body)

    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=sample_pdf_bytes,
    ):
        response = client.post("/transform/flatten", content=body, headers=headers)

    assert response.status_code == 200
    result_doc = fitz.open(stream=response.content, filetype="pdf")
    # The stamp text should now be part of the page content
    text = result_doc[0].get_text()
    assert "VERIFIED" in text
