"""Security endpoints: encrypt, authorize, decrypt, sanitize."""

import base64
import time

from fastapi import APIRouter, Depends, Response

from app.dependencies import require_auth
from app.models.requests import (
    AuthorizeAndUnlockRequest,
    DecryptRequest,
    EncryptRequest,
    SanitizeRequest,
)
from app.models.responses import AuthorizedUnlockData, AuthorizedUnlockResponse
from app.services import download_service, pdf_service

router = APIRouter(prefix="/security")


@router.post("/encrypt", dependencies=[Depends(require_auth)])
async def encrypt_pdf(request: EncryptRequest):
    """Encrypt a PDF with password protection. Returns encrypted PDF."""
    pdf_bytes = await download_service.download_pdf(request.source_url)
    result = pdf_service.encrypt_pdf(
        pdf_bytes, request.user_password, request.owner_password, request.permissions
    )
    return Response(content=result, media_type="application/pdf")


@router.post("/decrypt", dependencies=[Depends(require_auth)])
async def decrypt_pdf(request: DecryptRequest):
    """Decrypt a password-protected PDF. Returns unencrypted PDF."""
    pdf_bytes = await download_service.download_pdf(request.source_url)
    result = pdf_service.decrypt_pdf(pdf_bytes, request.password)
    return Response(content=result, media_type="application/pdf")


@router.post(
    "/authorize-and-unlock",
    response_model=AuthorizedUnlockResponse,
    dependencies=[Depends(require_auth)],
)
async def authorize_and_unlock_pdf(request: AuthorizeAndUnlockRequest, response: Response):
    """Authorize one known password and return a temporary in-memory derivative."""
    start = time.monotonic()
    response.headers["Cache-Control"] = "no-store, private"
    response.headers["Pragma"] = "no-cache"

    pdf_bytes = await download_service.download_pdf(request.source_url)
    result = pdf_service.authorize_and_unlock_pdf(
        pdf_bytes,
        request.password.get_secret_value(),
        authority_attested=request.authority_attested,
    )
    elapsed = (time.monotonic() - start) * 1000

    return AuthorizedUnlockResponse(
        success=True,
        data=AuthorizedUnlockData(
            authentication_level=result["authentication_level"],
            permissions=result["permissions"],
            has_digital_signatures=result["has_digital_signatures"],
            signature_state=result["signature_state"],
            pdf_base64=base64.b64encode(result["pdf_bytes"]).decode("ascii"),
        ),
        processing_time_ms=round(elapsed, 2),
    )


@router.post("/sanitize", dependencies=[Depends(require_auth)])
async def sanitize_document(request: SanitizeRequest):
    """Sanitize a PDF by removing metadata, JavaScript, links, and annotations."""
    pdf_bytes = await download_service.download_pdf(request.source_url)
    result = pdf_service.sanitize_document(
        pdf_bytes,
        request.remove_metadata,
        request.remove_javascript,
        request.remove_links,
        request.remove_annotations,
    )
    return Response(content=result, media_type="application/pdf")
