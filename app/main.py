"""PDF Engine Toolbox - FastAPI Application.

PyMuPDF-based PDF processing microservice.
Provides endpoints for page operations, transforms, redaction, text extraction,
thumbnails, and final PDF assembly.

Licensed under AGPL-3.0 (required by PyMuPDF dependency).
"""

import os
import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.build_identity import (
    LICENSE_IDENTIFIER,
    LICENSE_URL,
    corresponding_license_url,
    corresponding_source_url,
    UNVERIFIED_REVISIONS,
    read_build_commit,
    verify_build_identity,
)
from app.config import settings
from app.routes import (
    annotations,
    build,
    classify,
    convert,
    health,
    images,
    info,
    metadata,
    pages,
    redact,
    repair,
    security,
    source,
    forms,
    tasks,
    text,
    thumbnails,
    transform,
)
from app.utils.errors import PdfEngineError

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger()


def source_offer_headers() -> dict[str, str]:
    """Return the revision-pinned AGPL source headers for any response path."""
    build_commit = read_build_commit()
    source_url = corresponding_source_url(build_commit)
    license_url = corresponding_license_url(build_commit)
    return {
        "X-Source-Code": source_url,
        "Link": (
            f'<{source_url}>; rel="source", '
            f'<{license_url}>; rel="license"'
        ),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    log.info("pdf_engine_starting", log_level=settings.log_level)

    # Before serving anything: either we can name the revision we are running,
    # or we are explicitly allowed not to. Raising here aborts startup.
    revision = verify_build_identity()
    if revision in UNVERIFIED_REVISIONS:
        log.warning("pdf_engine_build_identity_unverified", build_commit=revision)
    else:
        log.info("pdf_engine_build_identity", build_commit=revision)

    # Missing or altered weights/packages abort startup before serving.
    from app.services.rapid_ocr import validate_runtime
    validate_runtime()

    legacy_backend = os.environ.get("OCR_BACKEND", "").strip().lower()
    if legacy_backend not in ("", "rapid"):
        # Paddle left the engine (TAX-6995); a leftover rollback pin must not
        # stop the service, but whoever set it has to know it does nothing.
        log.warning("ocr_backend_setting_ignored", ocr_backend=legacy_backend,
                    reason="Paddle was removed; RapidOCR serves every OCR operation")

    # Resume OCR jobs that were queued or running before the restart. Engines
    # load lazily, one per OCR worker, on the first page that needs a model.
    from app.services import ocr_queue
    ocr_queue.get_queue()

    yield
    log.info("pdf_engine_shutting_down")
    # Unfinished jobs stay in the ledger and resume on the next start.
    ocr_queue.shutdown()


app = FastAPI(
    title="PDF Engine Toolbox",
    description=(
        "PyMuPDF-based PDF processing microservice. Licensed under "
        f"{LICENSE_IDENTIFIER}. Corresponding Source: "
        "https://github.com/TaxScout-ai/pdf-engine-toolbox"
    ),
    version="1.0.0",
    license_info={"name": LICENSE_IDENTIFIER, "url": LICENSE_URL},
    lifespan=lifespan,
)


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(PdfEngineError)
async def pdf_engine_error_handler(request: Request, exc: PdfEngineError):
    """Handle known PDF engine errors."""
    return JSONResponse(
        status_code=exc.status_code,
        headers=source_offer_headers(),
        content={
            "success": False,
            "error": {"code": exc.code, "message": exc.message},
        },
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    log.error("unhandled_error", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        headers=source_offer_headers(),
        content={
            "success": False,
            "error": {"code": "INTERNAL_ERROR", "message": "An unexpected error occurred"},
        },
    )


# ============================================================================
# Middleware
# ============================================================================


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add timing and AGPL source-offer headers to every response."""
    start = time.monotonic()
    response = await call_next(request)
    elapsed = (time.monotonic() - start) * 1000
    response.headers["X-Processing-Time-Ms"] = f"{elapsed:.2f}"
    for name, value in source_offer_headers().items():
        response.headers[name] = value
    return response


# ============================================================================
# Routes
# ============================================================================

app.include_router(health.router, tags=["Health"])
app.include_router(source.router, tags=["Source"])
app.include_router(info.router, tags=["Info"])
app.include_router(pages.router, tags=["Pages"])
app.include_router(transform.router, tags=["Transform"])
app.include_router(redact.router, tags=["Redact"])
app.include_router(text.router, tags=["Text"])
app.include_router(thumbnails.router, tags=["Thumbnails"])
app.include_router(build.router, tags=["Build"])
app.include_router(images.router, tags=["Images"])
app.include_router(metadata.router, tags=["Metadata"])
app.include_router(security.router, tags=["Security"])
app.include_router(annotations.router, tags=["Annotations"])
app.include_router(repair.router, tags=["Repair"])
app.include_router(convert.router, tags=["Convert"])
app.include_router(classify.router, tags=["Classify"])
app.include_router(tasks.router, tags=["Tasks"])
app.include_router(forms.router, tags=["Forms"])
