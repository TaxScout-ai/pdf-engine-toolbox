"""Test security endpoints: encrypt, authorize, decrypt, sanitize."""

import base64
import hashlib
import json
from unittest.mock import AsyncMock, patch

import fitz


def _encrypted_pdf(
    pdf_bytes: bytes,
    *,
    user_password: str = "known-password",
    owner_password: str = "owner-password",
    permissions: int = -1,
) -> bytes:
    source = fitz.open(stream=pdf_bytes, filetype="pdf")
    encrypted = source.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw=user_password,
        owner_pw=owner_password,
        permissions=permissions,
    )
    source.close()
    return encrypted


def _authorize_request(client, auth_headers, encrypted_bytes, password, *, attested=True):
    body = json.dumps(
        {
            "source_url": "https://example.com/protected.pdf",
            "password": password,
            "authority_attested": attested,
        }
    )
    headers = auth_headers(
        "POST",
        "/security/authorize-and-unlock",
        body,
        version=2,
    )
    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=encrypted_bytes,
    ):
        return client.post(
            "/security/authorize-and-unlock",
            content=body,
            headers=headers,
        )


def test_encrypt_pdf(client, auth_headers, sample_pdf_bytes):
    """Encrypt should return a password-protected PDF."""
    body = json.dumps(
        {
            "source_url": "https://example.com/test.pdf",
            "user_password": "user123",
            "owner_password": "owner456",
        }
    )
    headers = auth_headers("POST", "/security/encrypt", body)

    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=sample_pdf_bytes,
    ):
        response = client.post("/security/encrypt", content=body, headers=headers)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"

    # Verify the PDF is encrypted
    doc = fitz.open(stream=response.content, filetype="pdf")
    assert doc.is_encrypted


def test_decrypt_pdf(client, auth_headers, sample_pdf_bytes):
    """Decrypt should remove password protection."""
    # First encrypt the PDF
    doc = fitz.open(stream=sample_pdf_bytes, filetype="pdf")
    encrypted_bytes = doc.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="test123",
        owner_pw="owner456",
    )

    body = json.dumps(
        {
            "source_url": "https://example.com/test.pdf",
            "password": "test123",
        }
    )
    headers = auth_headers("POST", "/security/decrypt", body)

    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=encrypted_bytes,
    ):
        response = client.post("/security/decrypt", content=body, headers=headers)

    assert response.status_code == 200
    result_doc = fitz.open(stream=response.content, filetype="pdf")
    assert not result_doc.is_encrypted
    assert len(result_doc) == 5


def test_authorize_and_unlock_allows_known_user_password_with_copy_permission(
    client,
    auth_headers,
    sample_pdf_bytes,
):
    encrypted = _encrypted_pdf(
        sample_pdf_bytes,
        permissions=fitz.PDF_PERM_COPY | fitz.PDF_PERM_PRINT,
    )

    response = _authorize_request(
        client,
        auth_headers,
        encrypted,
        "known-password",
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store, private"
    assert response.headers["pragma"] == "no-cache"
    data = response.json()["data"]
    assert data["authentication_level"] == "user"
    assert data["permissions"] == {
        "copy": True,
        "modify": False,
        "annotate": False,
        "print": True,
    }
    unlocked = fitz.open(
        stream=base64.b64decode(data["pdf_base64"]),
        filetype="pdf",
    )
    assert unlocked.is_encrypted is False
    assert len(unlocked) == 5


def test_authorize_and_unlock_rejects_v1_auth(
    client,
    auth_headers,
    sample_pdf_bytes,
):
    body = json.dumps(
        {
            "source_url": "https://example.com/protected.pdf",
            "password": "known-password",
            "authority_attested": True,
        }
    )

    response = client.post(
        "/security/authorize-and-unlock",
        content=body,
        headers=auth_headers("POST", "/security/authorize-and-unlock", body),
    )

    assert response.status_code == 401


def test_authorize_and_unlock_rejects_replayed_v2_request(
    client,
    auth_headers,
    sample_pdf_bytes,
):
    encrypted = _encrypted_pdf(sample_pdf_bytes)
    body = json.dumps(
        {
            "source_url": "https://example.com/protected.pdf",
            "password": "known-password",
            "authority_attested": True,
        }
    )
    headers = auth_headers(
        "POST",
        "/security/authorize-and-unlock",
        body,
        version=2,
        nonce="ab" * 16,
    )

    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=encrypted,
    ):
        first = client.post(
            "/security/authorize-and-unlock",
            content=body,
            headers=headers,
        )
        replay = client.post(
            "/security/authorize-and-unlock",
            content=body,
            headers=headers,
        )

    assert first.status_code == 200
    assert replay.status_code == 401


def test_authorize_and_unlock_allows_owner_password_when_copy_is_restricted(
    client,
    auth_headers,
    sample_pdf_bytes,
):
    encrypted = _encrypted_pdf(
        sample_pdf_bytes,
        permissions=fitz.PDF_PERM_PRINT,
    )

    response = _authorize_request(
        client,
        auth_headers,
        encrypted,
        "owner-password",
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["authentication_level"] == "owner"
    assert data["permissions"]["copy"] is True


def test_authorize_and_unlock_denies_user_password_when_copy_is_restricted(
    client,
    auth_headers,
    sample_pdf_bytes,
):
    encrypted = _encrypted_pdf(
        sample_pdf_bytes,
        permissions=fitz.PDF_PERM_PRINT,
    )

    response = _authorize_request(
        client,
        auth_headers,
        encrypted,
        "known-password",
    )

    assert response.status_code == 403
    assert response.json()["error"] == {
        "code": "PDF_PASSWORD_AUTHORIZATION_FAILED",
        "message": "Password is incorrect or document access is not authorized",
    }


def test_authorize_and_unlock_uses_same_generic_error_for_failed_controls(
    client,
    auth_headers,
    sample_pdf_bytes,
):
    encrypted = _encrypted_pdf(sample_pdf_bytes)

    wrong_password = _authorize_request(
        client,
        auth_headers,
        encrypted,
        "wrong-password",
    )
    missing_attestation = _authorize_request(
        client,
        auth_headers,
        encrypted,
        "known-password",
        attested=False,
    )
    oversized_password = "🔐" * 65
    oversized = _authorize_request(
        client,
        auth_headers,
        encrypted,
        oversized_password,
    )

    assert wrong_password.status_code == 403
    assert missing_attestation.status_code == 403
    assert oversized.status_code == 403
    assert wrong_password.json()["error"] == missing_attestation.json()["error"]
    assert wrong_password.json()["error"] == oversized.json()["error"]
    assert oversized_password not in oversized.text


def test_authorize_and_unlock_rejects_unencrypted_document(
    client,
    auth_headers,
    sample_pdf_bytes,
):
    response = _authorize_request(
        client,
        auth_headers,
        sample_pdf_bytes,
        "known-password",
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "PDF_PASSWORD_NOT_REQUIRED"


def test_authorize_and_unlock_warns_for_signatures_without_mutating_original(
    client,
    auth_headers,
):
    source = fitz.open()
    page = source.new_page()
    signature = fitz.Widget()
    signature.field_name = "taxpayer-signature"
    signature.field_type = fitz.PDF_WIDGET_TYPE_SIGNATURE
    signature.rect = fitz.Rect(50, 50, 250, 100)
    page.add_widget(signature)
    encrypted = source.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="known-password",
        owner_pw="owner-password",
        permissions=fitz.PDF_PERM_COPY,
    )
    source.close()
    original_hash = hashlib.sha256(encrypted).hexdigest()

    response = _authorize_request(
        client,
        auth_headers,
        encrypted,
        "known-password",
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["has_digital_signatures"] is True
    assert data["signature_state"] == "signatures-may-be-invalidated"
    assert hashlib.sha256(encrypted).hexdigest() == original_hash


def test_sanitize_removes_metadata(client, auth_headers, sample_pdf_bytes):
    """Sanitize should clear document metadata."""
    # First set some metadata
    doc = fitz.open(stream=sample_pdf_bytes, filetype="pdf")
    doc.set_metadata({"title": "Secret Title", "author": "Secret Author"})
    pdf_with_metadata = doc.tobytes()

    body = json.dumps(
        {
            "source_url": "https://example.com/test.pdf",
            "remove_metadata": True,
            "remove_javascript": True,
        }
    )
    headers = auth_headers("POST", "/security/sanitize", body)

    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=pdf_with_metadata,
    ):
        response = client.post("/security/sanitize", content=body, headers=headers)

    assert response.status_code == 200
    result_doc = fitz.open(stream=response.content, filetype="pdf")
    assert result_doc.metadata["title"] == ""
    assert result_doc.metadata["author"] == ""
