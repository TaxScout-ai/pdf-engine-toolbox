"""Conversion endpoints: images to PDF, office to PDF."""

import os
import time

from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from app.dependencies import require_auth
from app.models.requests import ImageToPdfRequest, OfficeToPdfRequest, PdfaRequest
from app.models.responses import TaskAcceptedResponse
from app.services import download_service, pdf_service, cache_service, task_service
from app.services.pdfa_service import PdfaConversionError, pdf_to_pdfa

router = APIRouter()

# Office files larger than this are processed as background tasks
_OFFICE_ASYNC_SIZE_THRESHOLD = 5 * 1024 * 1024  # 5 MB


@router.post("/convert/from-image", dependencies=[Depends(require_auth)])
async def images_to_pdf(request: ImageToPdfRequest):
    """Convert images (JPG, PNG, TIFF, HEIC) into a single PDF.

    Each image becomes one page. Supports iPhone HEIC photos via pillow-heif.
    """
    start = time.monotonic()

    # Download all images
    image_list = []
    for url in request.image_urls:
        img_bytes = await download_service.download_pdf(url)  # Works for any binary
        image_list.append(img_bytes)

    result = pdf_service.images_to_pdf(image_list, request.filenames)

    elapsed = (time.monotonic() - start) * 1000

    return Response(
        content=result,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=converted.pdf",
            "X-Processing-Time-Ms": f"{elapsed:.2f}",
        },
    )


async def _run_office_conversion(
    task_id: str, file_bytes: bytes, filename: str, src_hash: str,
):
    """Background worker for office-to-PDF conversion."""
    task_service.set_processing(task_id)
    try:
        result = pdf_service.office_to_pdf(file_bytes, filename)
        cache_service.put_cached(src_hash, "office_to_pdf", result, {"filename": filename})
        task_service.complete_task(task_id, {"pdf_size_bytes": len(result)})
    except Exception as e:
        task_service.fail_task(task_id, str(e))


@router.post("/convert/from-office", dependencies=[Depends(require_auth)])
async def office_to_pdf(request: OfficeToPdfRequest, background_tasks: BackgroundTasks):
    """Convert an office document (DOCX, XLSX, PPTX, etc.) to PDF via LibreOffice.

    Supports Word, Excel, PowerPoint, OpenDocument formats, RTF, CSV, and plain text.

    Small files are processed synchronously. Files larger than 5 MB are
    processed as a background task to avoid HTTP timeouts — returns a task_id
    that can be polled via ``GET /tasks/{task_id}``.
    """
    start = time.monotonic()

    file_bytes = await download_service.download_pdf(request.source_url)

    # Check cache (office conversion is expensive)
    src_hash = cache_service.content_hash(file_bytes)
    cache_params = {"filename": request.filename}
    cached = cache_service.get_cached(src_hash, "office_to_pdf", cache_params)
    if cached and isinstance(cached, bytes):
        base_name = os.path.splitext(request.filename)[0]
        elapsed = (time.monotonic() - start) * 1000
        return Response(
            content=cached,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{base_name}.pdf"',
                "X-Processing-Time-Ms": f"{elapsed:.2f}",
            },
        )

    # Large files → background task to avoid HTTP timeout
    if len(file_bytes) > _OFFICE_ASYNC_SIZE_THRESHOLD:
        task = task_service.create_task("office_to_pdf")
        background_tasks.add_task(
            _run_office_conversion, task.id, file_bytes, request.filename, src_hash,
        )
        size_mb = len(file_bytes) / (1024 * 1024)
        return TaskAcceptedResponse(
            success=True,
            task_id=task.id,
            status="pending",
            message=(
                f"Office conversion started for {request.filename} "
                f"({size_mb:.1f} MB). Poll GET /tasks/{task.id} for status."
            ),
        )

    # Small files → synchronous conversion
    result = pdf_service.office_to_pdf(file_bytes, request.filename)

    cache_service.put_cached(src_hash, "office_to_pdf", result, cache_params)

    base_name = os.path.splitext(request.filename)[0]
    elapsed = (time.monotonic() - start) * 1000

    return Response(
        content=result,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{base_name}.pdf"',
            "X-Processing-Time-Ms": f"{elapsed:.2f}",
        },
    )


@router.post("/convert/pdfa", dependencies=[Depends(require_auth)])
async def pdf_to_pdfa_route(request: PdfaRequest):
    """Convert a PDF to PDF/A-2b for archiving (TAX-5637)."""
    pdf_bytes = await download_service.download_pdf(request.source_url)
    try:
        # Ghostscript runs in a worker thread so it does not block the loop.
        result = await run_in_threadpool(pdf_to_pdfa, pdf_bytes)
    except PdfaConversionError as e:
        return JSONResponse(
            status_code=422,
            content={"success": False, "error": {"code": "PDFA_FAILED", "message": str(e)}},
        )
    return Response(content=result, media_type="application/pdf")
