"""Text extraction, search, bookmark, table, and OCR endpoints."""

import base64
import time

from fastapi import APIRouter, BackgroundTasks, Depends, Response

from app.dependencies import require_auth
from app.models.requests import (
    TextExtractRequest,
    TextSearchRequest,
    BookmarksRequest,
    ExtractTablesRequest,
    OcrRequest,
    VlReadRequest,
)
from app.models.responses import (
    VlReadResponse,
    VlReadData,
    TextExtractResponse,
    TextExtractData,
    PageText,
    TextBlock,
    TextSearchResponse,
    TextSearchData,
    SearchMatch,
    ExtractTablesResponse,
    ExtractTablesData,
    ExtractedTable,
    OcrResponse,
    OcrData,
    OcrPageResult,
    OcrWord,
    TaskAcceptedResponse,
)
from app.services import download_service, pdf_service, task_service, cache_service

router = APIRouter(prefix="/text")


@router.post(
    "/extract", response_model=TextExtractResponse, dependencies=[Depends(require_auth)]
)
async def extract_text(request: TextExtractRequest):
    """Extract text from PDF pages with optional position data."""
    start = time.monotonic()

    pdf_bytes = await download_service.download_pdf(request.source_url)
    pages_data = pdf_service.extract_text(
        pdf_bytes, request.pages, request.include_positions
    )

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


@router.post(
    "/search", response_model=TextSearchResponse, dependencies=[Depends(require_auth)]
)
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


@router.post(
    "/tables", response_model=ExtractTablesResponse, dependencies=[Depends(require_auth)]
)
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
    tables = pdf_service.extract_tables(
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
        ),
        processing_time_ms=round(elapsed_ms, 2),
    )


def _run_ocr(task_id: str, pdf_bytes: bytes, pages, language: str, dpi: int):
    """Background worker for OCR processing.

    IMPORTANT: Must be a regular ``def`` (not ``async def``) so Starlette
    runs it via ``run_in_threadpool()`` instead of ``await``-ing it on the
    event loop.  PaddleOCR inference is CPU-bound and blocks for minutes;
    an ``async def`` wrapper would freeze the event loop and prevent
    ``complete_task()`` from executing.
    """
    task_service.set_processing(task_id)
    try:
        result = pdf_service.ocr_pages(pdf_bytes, pages, language, dpi)
        pdf_b64 = base64.b64encode(result["pdf_bytes"]).decode("ascii")

        # Build the full result the client expects (pages + pdf_base64)
        ocr_data = {
            "pages": result["pages"],
            "total_words": result["total_words"],
            "avg_confidence": result["avg_confidence"],
            "pdf_base64": pdf_b64,
        }

        # Cache for future identical requests
        src_hash = cache_service.content_hash(pdf_bytes)
        cache_params = {
            "pages": sorted(pages) if pages else None,
            "language": language,
            "dpi": dpi,
        }
        cache_service.put_cached(src_hash, "ocr", ocr_data, cache_params)

        task_service.complete_task(task_id, ocr_data)
    except Exception as e:
        task_service.fail_task(task_id, str(e))


@router.post("/ocr", dependencies=[Depends(require_auth)])
async def ocr_pages(request: OcrRequest, background_tasks: BackgroundTasks):
    """Run PaddleOCR on scanned PDF pages.

    Returns rich JSON with per-word text, 4-point bounding boxes (in PDF
    points), and confidence scores.  Also includes the PDF with an invisible
    text layer (base64-encoded) so Ctrl+F search works in viewers.

    Uses ``use_angle_cls=True`` to handle rotated receipts/scans.

    For small documents (≤5 pages), processes synchronously.
    For larger documents, runs as a background task — returns a ``task_id``
    for polling via ``GET /tasks/{task_id}``.
    """
    pdf_bytes = await download_service.download_pdf(request.source_url)

    # Check cache (OCR is expensive)
    src_hash = cache_service.content_hash(pdf_bytes)
    cache_params = {
        "pages": sorted(request.pages) if request.pages else None,
        "language": request.language,
        "dpi": request.dpi,
    }
    cached = cache_service.get_cached(src_hash, "ocr", cache_params)
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

    # Always use background tasks for OCR — CPU inference with PP-OCRv5
    # server_det at 300 DPI can exceed HTTP client timeouts (120s) even for
    # single-page documents on constrained instances (t3a.medium / 2 vCPU).
    # The client polls via GET /tasks/{task_id} with OCR_TIMEOUT_MS.
    task = task_service.create_task("ocr")
    background_tasks.add_task(
        _run_ocr, task.id, pdf_bytes, request.pages, request.language, request.dpi,
    )

    return TaskAcceptedResponse(
        success=True,
        task_id=task.id,
        status="pending",
        message=f"OCR started for {page_count} pages. Poll GET /tasks/{task.id} for status.",
    )


def _run_vl_read(task_id: str, pdf_bytes: bytes, pages: list[int]) -> None:
    """Background worker for /text/read."""
    from app.services import vl_service

    try:
        data = vl_service.read_pages(pdf_bytes, pages)
        cache_service.put_cached(
            cache_service.content_hash(pdf_bytes), "vl_read", data,
            {"pages": sorted(pages)},
        )
        task_service.complete_task(task_id, data)
    except Exception as e:
        # Fails the task rather than completing it with nothing. An empty read
        # reported as success would be indistinguishable from "the document
        # does not say that" — the exact silence that let the paddlepaddle
        # 3.3.x OCR regression pass 82 tests.
        task_service.fail_task(task_id, str(e))


@router.post("/read", dependencies=[Depends(require_auth)])
async def vl_read_pages(request: VlReadRequest, background_tasks: BackgroundTasks):
    """Read pages with PaddleOCR-VL 1.6 — a second opinion, not an OCR replacement.

    Returns block-level text with **no word geometry**. When you need per-word
    boxes (viewer highlight, searchable PDF, evidence anchoring), use
    ``POST /text/ocr``; VL cannot serve those and never will.

    Use this where a read is *doubted*: a field that failed corroboration
    against the text layer, a confidence below threshold, a validator flag. VL
    reads ``3497`` where the standard recogniser reads ``349``.

    Always a background task. At ~25 s/page even two pages outrun a typical
    HTTP client timeout. Poll ``GET /tasks/{task_id}``.
    """
    pdf_bytes = await download_service.download_pdf(request.source_url)

    # VL is the most expensive thing this service does; never pay twice.
    src_hash = cache_service.content_hash(pdf_bytes)
    cache_params = {"pages": sorted(request.pages)}
    cached = cache_service.get_cached(src_hash, "vl_read", cache_params)
    if cached and isinstance(cached, dict):
        return VlReadResponse(
            success=True, data=VlReadData(**cached), processing_time_ms=0.0
        )

    task = task_service.create_task("vl_read")
    background_tasks.add_task(_run_vl_read, task.id, pdf_bytes, request.pages)

    return TaskAcceptedResponse(
        success=True,
        task_id=task.id,
        status="pending",
        message=(
            f"VL read started for {len(request.pages)} page(s) "
            f"(~25 s/page). Poll GET /tasks/{task.id}."
        ),
    )
