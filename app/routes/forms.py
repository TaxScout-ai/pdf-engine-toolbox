"""PDF form fields: list and fill (TAX-5565)."""

from fastapi import APIRouter, Depends, Response

from app.dependencies import require_auth
from app.models.requests import FormFieldsRequest, FormFillRequest
from app.services import download_service, pdf_service

router = APIRouter(prefix="/forms")


@router.post("/fields", dependencies=[Depends(require_auth)])
async def list_form_fields(request: FormFieldsRequest):
    """The fillable fields of a PDF form (signature fields excluded)."""
    pdf_bytes = await download_service.download_pdf(request.source_url)
    return {"success": True, "data": {"fields": pdf_service.list_form_fields(pdf_bytes)}}


@router.post("/fill", dependencies=[Depends(require_auth)])
async def fill_form(request: FormFillRequest):
    """Fill fields by name and return the PDF; optionally flatten the form."""
    pdf_bytes = await download_service.download_pdf(request.source_url)
    result, changed = pdf_service.fill_form_fields(
        pdf_bytes, request.values, flatten=request.flatten
    )
    return Response(
        content=result,
        media_type="application/pdf",
        headers={"X-Fields-Changed": str(changed)},
    )
