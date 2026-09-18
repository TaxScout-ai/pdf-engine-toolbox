"""Redaction endpoints."""

import base64
import binascii
import time

from fastapi import APIRouter, BackgroundTasks, Depends, Response
from fastapi.responses import JSONResponse

from app.dependencies import require_auth
from app.models.requests import DetectPiiRequest, RedactIdentifiersRequest, RedactRequest
from app.models.responses import (
    DetectPiiData,
    DetectPiiResponse,
    PiiDetection,
)
from app.services import (
    cache_service,
    download_service,
    identifier_redaction,
    pdf_service,
    task_service,
)

router = APIRouter(prefix="/redact")


@router.post("", dependencies=[Depends(require_auth)])
async def apply_redactions(request: RedactRequest):
    """Apply true redactions - permanently remove content."""
    pdf_bytes = await download_service.download_pdf(request.source_url)
    redactions = [r.model_dump() for r in request.redactions]
    result = pdf_service.redact_content(pdf_bytes, redactions)
    return Response(content=result, media_type="application/pdf")


@router.post("/detect-pii", response_model=DetectPiiResponse, dependencies=[Depends(require_auth)])
async def detect_pii(request: DetectPiiRequest):
    """Detect PII patterns (SSN, EIN, phone, email) in PDF text."""
    start = time.monotonic()

    pdf_bytes = await download_service.download_pdf(request.source_url)
    detections = pdf_service.detect_pii(pdf_bytes, request.patterns)

    elapsed = (time.monotonic() - start) * 1000

    return DetectPiiResponse(
        success=True,
        data=DetectPiiData(
            detections=[PiiDetection(**d) for d in detections],
        ),
        processing_time_ms=round(elapsed, 2),
    )


_OPERATION = "redact_identifiers"


def _redaction_payload(result: dict) -> dict:
    return {
        "pdf_base64": base64.b64encode(result["pdf_bytes"]).decode("ascii"),
        "redactions": result["redactions"],
        "ocr_pages": result["ocr_pages"],
    }


def _run_identifier_redaction(task_id: str, data: bytes, media_type: str, source_hash: str):
    """Background worker: OCR is CPU-bound for minutes, so plain ``def``."""
    task_service.set_processing(task_id)
    try:
        payload = _redaction_payload(identifier_redaction.redact_identifiers(data, media_type))
        cache_service.put_cached(source_hash, _OPERATION, payload, {"media_type": media_type})
        task_service.complete_task(task_id, payload)
    except identifier_redaction.RedactionIncompleteError as e:
        task_service.fail_task(task_id, f"redaction_incomplete: {e}")
    except Exception as e:
        task_service.fail_task(task_id, str(e))


@router.post("/identifiers", dependencies=[Depends(require_auth)])
async def redact_identifiers(request: RedactIdentifiersRequest, background_tasks: BackgroundTasks):
    """Black out SSNs, ITINs and EINs and prove none is left (TAX-4858).

    Digital documents answer at once. A document with any page that needs OCR
    (a scan or a photo) is processed as a background task: poll /tasks/{id}.
    Results are cached by content hash, so the same bytes are redacted once.
    A document whose identifiers cannot all be removed is refused (422), never
    returned partially redacted.
    """
    try:
        data = base64.b64decode(request.content_base64, validate=True)
    except (binascii.Error, ValueError):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": {
                    "code": "INVALID_BASE64",
                    "message": "content_base64 is not valid base64",
                },
            },
        )
    source_hash = cache_service.content_hash(data)
    params = {"media_type": request.media_type}
    cached = cache_service.get_cached(source_hash, _OPERATION, params)
    if isinstance(cached, dict):
        return {"success": True, "data": cached, "processing_time_ms": 0.0}

    needs_ocr = request.media_type != "application/pdf" or any(
        identifier_redaction.needs_ocr(page) for page in pdf_service._open_pdf(data)
    )
    if needs_ocr and request.text_layer_only:
        # The caller redacts text-layer documents only and sends the rest as
        # before; minutes of OCR it will not use are not started (TAX-4858 C).
        return {"success": True, "status": "needs_ocr"}
    if needs_ocr:
        task = task_service.create_task(_OPERATION)
        background_tasks.add_task(
            _run_identifier_redaction, task.id, data, request.media_type, source_hash
        )
        return {"success": True, "task_id": task.id, "status": "pending"}

    start = time.monotonic()
    try:
        result = identifier_redaction.redact_identifiers(
            data, request.media_type, ocr=lambda *_: {}
        )
    except identifier_redaction.RedactionIncompleteError as e:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": {"code": "REDACTION_INCOMPLETE", "message": str(e)},
            },
        )
    payload = _redaction_payload(result)
    cache_service.put_cached(source_hash, _OPERATION, payload, params)
    return {
        "success": True,
        "data": payload,
        "processing_time_ms": round((time.monotonic() - start) * 1000, 2),
    }
