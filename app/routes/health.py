"""Health check endpoint."""

import fitz
from fastapi import APIRouter

from app.build_identity import (
    LICENSE_IDENTIFIER,
    corresponding_source_url,
    read_build_commit,
)
from app.models.responses import HealthResponse

router = APIRouter()


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
        source_code_url=corresponding_source_url(build_commit),
    )
