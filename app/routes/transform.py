"""Transform operation endpoints: orientation, deskew, compress, flatten, watermark."""

from fastapi import APIRouter, Depends, Response

from app.dependencies import require_auth
from app.models.requests import (
    CompressRequest,
    DeskewRequest,
    FlattenRequest,
    OrientationRequest,
    WatermarkRequest,
)
from app.services import download_service, pdf_service

router = APIRouter(prefix="/transform")


@router.post("/orientation", dependencies=[Depends(require_auth)])
async def detect_page_orientation(request: OrientationRequest):
    """Detect cardinal rotation on image-based pages without changing the PDF."""
    pdf_bytes = await download_service.download_pdf(request.source_url)
    return pdf_service.detect_page_orientations(
        pdf_bytes,
        request.pages,
        scanned_only=request.scanned_only,
        min_confidence=request.min_confidence,
    )


@router.post("/deskew", dependencies=[Depends(require_auth)])
async def deskew_pages(request: DeskewRequest):
    """Auto-straighten scanned pages."""
    pdf_bytes = await download_service.download_pdf(request.source_url)
    result = pdf_service.deskew_pages(pdf_bytes, request.pages)
    return Response(content=result, media_type="application/pdf")


@router.post("/compress", dependencies=[Depends(require_auth)])
async def compress_pdf(request: CompressRequest):
    """Compress and optimize PDF."""
    pdf_bytes = await download_service.download_pdf(request.source_url)
    result = pdf_service.compress_pdf(pdf_bytes, request.quality, request.max_image_dpi)
    return Response(content=result, media_type="application/pdf")


@router.post("/flatten", dependencies=[Depends(require_auth)])
async def flatten_annotations(request: FlattenRequest):
    """Burn annotations into PDF content."""
    pdf_bytes = await download_service.download_pdf(request.source_url)
    annotations = [a.model_dump() for a in request.annotations]
    result = pdf_service.flatten_annotations(pdf_bytes, annotations)
    return Response(content=result, media_type="application/pdf")


@router.post("/watermark", dependencies=[Depends(require_auth)])
async def add_watermark(request: WatermarkRequest):
    """Add a text watermark to pages."""
    pdf_bytes = await download_service.download_pdf(request.source_url)
    result = pdf_service.add_text_watermark(
        pdf_bytes,
        request.text,
        request.pages,
        request.font_size,
        request.color,
        request.opacity,
        request.rotation,
        request.user_name,
        request.date,
    )
    return Response(content=result, media_type="application/pdf")
