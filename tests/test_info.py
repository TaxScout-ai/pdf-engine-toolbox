"""Test PDF info endpoint."""

import json
from unittest.mock import AsyncMock, patch

import fitz


def test_info_requires_auth(client):
    """Info endpoint should require authentication."""
    response = client.post("/info", json={"source_url": "https://example.com/test.pdf"})
    assert response.status_code == 401


def test_info_returns_page_data(client, auth_headers, sample_pdf_bytes):
    """Info endpoint should return page count and metadata."""
    body = json.dumps({"source_url": "https://example.com/test.pdf"})
    headers = auth_headers("POST", "/info", body)

    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=sample_pdf_bytes,
    ):
        response = client.post("/info", content=body, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["page_count"] == 5
    assert len(data["data"]["pages"]) == 5
    assert data["data"]["pages"][0]["index"] == 0
    assert data["data"]["pages"][0]["has_text"] is True
    assert data["data"]["is_encrypted"] is False
    assert data["data"]["requires_password"] is False
    assert data["data"]["authentication_level"] == "none"
    assert data["data"]["permissions"]["copy"] is True


def test_locked_info_returns_no_content_bearing_fields(
    client,
    auth_headers,
    sample_pdf_bytes,
):
    """Preflight must detect a password without reading pages or metadata."""
    source = fitz.open(stream=sample_pdf_bytes, filetype="pdf")
    encrypted_bytes = source.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="known-password",
        owner_pw="owner-password",
    )
    source.close()

    body = json.dumps({"source_url": "https://example.com/locked.pdf"})
    headers = auth_headers("POST", "/info", body)
    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=encrypted_bytes,
    ):
        response = client.post("/info", content=body, headers=headers)

    assert response.status_code == 200
    data = response.json()["data"]
    assert data == {
        "page_count": 0,
        "pages": [],
        "is_encrypted": True,
        "requires_password": True,
        "authentication_level": "none",
        "permissions": None,
        "has_digital_signatures": None,
        "signature_state": "unknown",
        "metadata": None,
    }


def test_owner_only_encryption_reports_permissions_without_requesting_password(
    client,
    auth_headers,
    sample_pdf_bytes,
):
    """An empty user password still requires permission-policy evaluation."""
    source = fitz.open(stream=sample_pdf_bytes, filetype="pdf")
    encrypted_bytes = source.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="",
        owner_pw="owner-password",
        permissions=fitz.PDF_PERM_PRINT,
    )
    source.close()

    body = json.dumps({"source_url": "https://example.com/owner-only.pdf"})
    headers = auth_headers("POST", "/info", body)
    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=encrypted_bytes,
    ):
        response = client.post("/info", content=body, headers=headers)

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["is_encrypted"] is True
    assert data["requires_password"] is False
    assert data["authentication_level"] == "user"
    assert data["permissions"]["copy"] is False
    assert data["permissions"]["print"] is True
