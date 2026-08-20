"""PDF info endpoint."""

import time

from fastapi import APIRouter, Depends

from app.dependencies import require_auth
from app.models.requests import InfoRequest
from app.models.responses import PageInfo, PdfInfoData, PdfInfoResponse
from app.services import download_service, pdf_service

router = APIRouter()


@router.post("/info", response_model=PdfInfoResponse, dependencies=[Depends(require_auth)])
async def get_pdf_info(request: InfoRequest):
    """Get PDF metadata and page information."""
    start = time.monotonic()

    pdf_bytes = await download_service.download_pdf(request.source_url)
    info = pdf_service.get_info(pdf_bytes)

    elapsed = (time.monotonic() - start) * 1000

    return PdfInfoResponse(
        success=True,
        data=PdfInfoData(
            page_count=info["page_count"],
            pages=[PageInfo(**p) for p in info["pages"]],
            is_encrypted=info["is_encrypted"],
            requires_password=info["requires_password"],
            authentication_level=info["authentication_level"],
            permissions=info["permissions"],
            has_digital_signatures=info["has_digital_signatures"],
            signature_state=info["signature_state"],
            metadata=info["metadata"],
        ),
        processing_time_ms=round(elapsed, 2),
    )
