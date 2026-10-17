"""Text extraction, search, bookmark, table, and OCR endpoints."""

import base64
import time

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.dependencies import require_auth
from app.models.requests import (
    BookmarksRequest,
    ExtractTablesRequest,
    OcrRequest,
    TextExtractRequest,
    TextSearchRequest,
)
from app.models.responses import (
    ExtractedTable,
    ExtractTablesData,
    ExtractTablesResponse,
    OcrData,
    OcrPageResult,
    OcrResponse,
    OcrWord,
    PageText,
    SearchMatch,
    TaskAcceptedResponse,
    TextBlock,
    TextExtractData,
    TextExtractResponse,
    TextSearchData,
    TextSearchResponse,
)
from app.services import cache_service, download_service, ocr_queue, pdf_service
from app.services.rapid_ocr import cache_profile

router = APIRouter(prefix="/text")


@router.post("/extract", response_model=TextExtractResponse, dependencies=[Depends(require_auth)])
async def extract_text(request: TextExtractRequest):
    """Extract text from PDF pages with optional position data."""
    start = time.monotonic()

    pdf_bytes = await download_service.download_pdf(request.source_url)
    pages_data = pdf_service.extract_text(pdf_bytes, request.pages, request.include_positions)

    elapsed = (time.monotonic() - start) * 1000

    pages = []
    for pd in pages_data:
        blocks = None
        if pd.get("blocks"):
            blocks = [TextBlock(**b) for b in pd["blocks"]]
        pages.append(
            PageText(
                page_index=pd["page_index"],
                text=pd["text"],
                blocks=blocks,
            )
        )

    return TextExtractResponse(
        success=True,
        data=TextExtractData(pages=pages),
        processing_time_ms=round(elapsed, 2),
    )


@router.post("/search", response_model=TextSearchResponse, dependencies=[Depends(require_auth)])
async def search_text(request: TextSearchRequest):
    """Search for text across all PDF pages."""
    start = time.monotonic()

    pdf_bytes = await download_service.download_pdf(request.source_url)
    result = pdf_service.search_text(pdf_bytes, request.query, request.case_sensitive)

    elapsed = (time.monotonic() - start) * 1000

    return TextSearchResponse(
        success=True,
        data=TextSearchData(
            total_matches=result["total_matches"],
            matches=[SearchMatch(**m) for m in result["matches"]],
        ),
        processing_time_ms=round(elapsed, 2),
    )


@router.post("/bookmarks", dependencies=[Depends(require_auth)])
async def add_bookmarks(request: BookmarksRequest):
    """Add bookmarks (table of contents) to a PDF."""
    pdf_bytes = await download_service.download_pdf(request.source_url)
    bookmarks = [b.model_dump() for b in request.bookmarks]
    result = pdf_service.add_bookmarks(pdf_bytes, bookmarks)
    return Response(content=result, media_type="application/pdf")


@router.post("/tables", response_model=ExtractTablesResponse, dependencies=[Depends(require_auth)])
async def extract_tables(request: ExtractTablesRequest):
    """Extract tabular data from PDF pages.

    Supports three strategies:
    - ``auto`` (default): PyMuPDF for digital pages, PP-Structure for scans.
    - ``pymupdf``: Fast rule-based extraction (digital PDFs only).
    - ``ppstructure``: Neural network table recognition (best for scanned
      1099s, W-2s, K-1s with faint or crooked lines).
    """
    start = time.monotonic()

    pdf_bytes = await download_service.download_pdf(request.source_url)
    # Off the event loop: PP-Structure inference can wait on the Paddle lock.
    tables = await run_in_threadpool(
        pdf_service.extract_tables,
        pdf_bytes,
        request.pages,
        strategy=request.strategy,
        language=request.language,
        dpi=request.dpi,
    )

    elapsed = (time.monotonic() - start) * 1000

    return ExtractTablesResponse(
        success=True,
        data=ExtractTablesData(
            tables=[ExtractedTable(**t) for t in tables],
            total_count=len(tables),
        ),
        processing_time_ms=round(elapsed, 2),
    )


def _build_ocr_response(result: dict, elapsed_ms: float) -> OcrResponse:
    """Convert the raw ocr_pages() dict into the Pydantic OcrResponse."""
    pdf_b64 = base64.b64encode(result["pdf_bytes"]).decode("ascii")

    pages = [
        OcrPageResult(
            page_index=p["page_index"],
            words=[OcrWord(**w) for w in p["words"]],
            full_text=p["full_text"],
            text_source=p.get("text_source"),
            visual_width=p.get("visual_width"),
            visual_height=p.get("visual_height"),
            text_layer_quality=p.get("text_layer_quality"),
            preprocessing=p.get("preprocessing"),
            status=p.get("status"), warnings=p.get("warnings"),
            discarded_words=p.get("discarded_words"),
        )
        for p in result["pages"]
    ]

    return OcrResponse(
        success=True,
        data=OcrData(
            pages=pages,
            total_words=result["total_words"],
            avg_confidence=result["avg_confidence"],
            pdf_base64=pdf_b64,
            provenance=result.get("provenance"),
            coverage=result.get("coverage"),
        ),
        processing_time_ms=round(elapsed_ms, 2),
    )


@router.post("/ocr", dependencies=[Depends(require_auth)])
async def ocr_pages(request: OcrRequest):
    """Run the configured OCR backend on scanned PDF pages.

    Returns rich JSON with per-word text, 4-point bounding boxes (in PDF
    points), and confidence scores.  Also includes the PDF with an invisible
    text layer (base64-encoded) so Ctrl+F search works in viewers.

    Queued as a persistent job — returns a ``task_id`` for polling via
    ``GET /tasks/{task_id}``. The job survives an engine restart. When the
    queue already holds ``OCR_QUEUE_MAX_JOBS`` unfinished jobs the request is
    refused with HTTP 429 and a ``Retry-After`` header (seconds); the caller
    must retry after that delay. Nothing else refuses a request as busy.
    """
    pdf_bytes = await download_service.download_pdf(request.source_url)
    backend = settings.ocr_backend

    # Check cache (OCR is expensive)
    src_hash = cache_service.content_hash(pdf_bytes)
    cache_params = {
        "pages": sorted(request.pages) if request.pages else None,
        "language": request.language,
        "dpi": request.dpi,
        **cache_profile(backend),
    }
    # Deduplicated from the queue's sealed ledger, never the plaintext file
    # cache: OCR results carry client text and the searchable PDF.
    cached = await run_in_threadpool(
        ocr_queue.get_queue().cached_result, src_hash, cache_params
    )
    if cached and isinstance(cached, dict):
        return OcrResponse(
            success=True,
            data=OcrData(**cached),
            processing_time_ms=0.0,
        )

    # Determine page count
    page_count = len(request.pages) if request.pages else 0
    if page_count == 0:
        import fitz

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_count = len(doc)

    # Always queue OCR — CPU inference can exceed HTTP client timeouts (120s)
    # even for single-page documents on constrained instances. The client
    # polls via GET /tasks/{task_id} with OCR_TIMEOUT_MS.
    try:
        task_id = await run_in_threadpool(
            ocr_queue.get_queue().submit_ocr,
            pdf_bytes,
            request.pages,
            request.language,
            request.dpi,
            backend,
            src_hash,
            cache_params,
        )
    except ocr_queue.QueueFullError as exc:
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(exc.retry_after)},
            content={
                "success": False,
                "error": {
                    "code": "OCR_QUEUE_FULL",
                    "message": f"OCR queue is full; retry after {exc.retry_after} seconds",
                },
            },
        )

    return TaskAcceptedResponse(
        success=True,
        task_id=task_id,
        status="pending",
        message=f"OCR started for {page_count} pages. Poll GET /tasks/{task_id} for status.",
    )
