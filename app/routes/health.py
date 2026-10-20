"""Health check endpoint."""

import fitz
from fastapi import APIRouter

from app.build_identity import (
    LICENSE_IDENTIFIER,
    PROJECT_LICENSE_IDENTIFIER,
    corresponding_source_url,
    read_build_commit,
)
from app.config import settings
from app.models.responses import HealthResponse

router = APIRouter()


def capabilities() -> list[str]:
    """What this engine instance actually does, for callers that gate on it.

    * ``ocr-queue:persistent`` — /text/ocr jobs live in the persistent queue:
      they survive a restart, and a full queue answers 429 with Retry-After
      (seconds) instead of failing a task as busy.
    * ``identifier-redaction:rapid-ocr`` — /redact/identifiers reads page
      images with RapidOCR. Present only when OCR_BACKEND=rapid; under the
      Paddle rollback redaction reads with Paddle and this is absent.
    """
    caps = ["ocr-queue:persistent"]
    if settings.ocr_backend == "rapid":
        caps.append("identifier-redaction:rapid-ocr")
    return caps


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check - no authentication required."""
    build_commit = read_build_commit()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        build_commit=build_commit,
        pymupdf_version=fitz.version[0],
        license=LICENSE_IDENTIFIER,
        project_license=PROJECT_LICENSE_IDENTIFIER,
        source_code_url=corresponding_source_url(build_commit),
        capabilities=capabilities(),
    )
