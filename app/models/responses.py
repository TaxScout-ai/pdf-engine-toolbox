"""Pydantic response models for all API endpoints."""

from typing import Literal

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Error detail in API response."""

    code: str
    message: str


class PageInfo(BaseModel):
    """Info about a single page."""

    index: int
    width: float
    height: float
    rotation: int
    has_text: bool


PdfAuthenticationLevel = Literal["none", "user", "owner", "user-and-owner"]
PdfSignatureState = Literal[
    "unknown",
    "none-detected",
    "signature-fields-present",
    "signatures-may-be-invalidated",
]


class PdfPermissionState(BaseModel):
    """Normalized extraction-relevant PDF permission flags."""

    copy_allowed: bool = Field(alias="copy", serialization_alias="copy")
    modify: bool
    annotate: bool
    print: bool


class PdfInfoData(BaseModel):
    """PDF metadata response data."""

    page_count: int
    pages: list[PageInfo]
    is_encrypted: bool
    requires_password: bool
    authentication_level: PdfAuthenticationLevel
    permissions: PdfPermissionState | None
    has_digital_signatures: bool | None
    signature_state: PdfSignatureState
    metadata: dict | None = None


class PdfInfoResponse(BaseModel):
    """Response for POST /info."""

    success: bool
    data: PdfInfoData | None = None
    error: ErrorDetail | None = None
    processing_time_ms: float | None = None


class AuthorizedUnlockData(BaseModel):
    """Authorized in-memory derivative plus normalized policy evidence."""

    authentication_level: PdfAuthenticationLevel
    permissions: PdfPermissionState
    has_digital_signatures: bool | None
    signature_state: PdfSignatureState
    pdf_base64: str


class AuthorizedUnlockResponse(BaseModel):
    """Response for POST /security/authorize-and-unlock."""

    success: bool
    data: AuthorizedUnlockData | None = None
    error: ErrorDetail | None = None
    processing_time_ms: float | None = None


class PiiDetection(BaseModel):
    """A detected PII instance."""

    page: int
    pattern: str
    text: str
    rect: dict


class DetectPiiData(BaseModel):
    """PII detection response data."""

    detections: list[PiiDetection]


class DetectPiiResponse(BaseModel):
    """Response for POST /redact/detect-pii."""

    success: bool
    data: DetectPiiData | None = None
    error: ErrorDetail | None = None
    processing_time_ms: float | None = None


class TextBlock(BaseModel):
    """A block of extracted text with position."""

    bbox: dict
    text: str
    font_size: float | None = None


class PageText(BaseModel):
    """Extracted text for a page."""

    page_index: int
    text: str
    blocks: list[TextBlock] | None = None


class TextExtractData(BaseModel):
    """Text extraction response data."""

    pages: list[PageText]


class TextExtractResponse(BaseModel):
    """Response for POST /text/extract."""

    success: bool
    data: TextExtractData | None = None
    error: ErrorDetail | None = None
    processing_time_ms: float | None = None


class SearchMatch(BaseModel):
    """A text search match."""

    page_index: int
    text: str
    rect: dict
    context: str


class TextSearchData(BaseModel):
    """Text search response data."""

    total_matches: int
    matches: list[SearchMatch]


class TextSearchResponse(BaseModel):
    """Response for POST /text/search."""

    success: bool
    data: TextSearchData | None = None
    error: ErrorDetail | None = None
    processing_time_ms: float | None = None


class ThumbnailItem(BaseModel):
    """A single page thumbnail."""

    page_index: int
    data_url: str
    width: int
    height: int


class ThumbnailsData(BaseModel):
    """Thumbnails response data."""

    thumbnails: list[ThumbnailItem]


class ThumbnailsResponse(BaseModel):
    """Response for POST /thumbnails."""

    success: bool
    data: ThumbnailsData | None = None
    error: ErrorDetail | None = None
    processing_time_ms: float | None = None


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    version: str
    pymupdf_version: str


# ============================================================================
# Image Extraction Responses
# ============================================================================


class ExtractedImage(BaseModel):
    """An extracted image from a PDF page."""

    page_index: int
    image_index: int
    width: int
    height: int
    colorspace: int
    bpc: int
    size_bytes: int
    data_url: str


class ExtractImagesData(BaseModel):
    """Image extraction response data."""

    images: list[ExtractedImage]
    total_count: int


class ExtractImagesResponse(BaseModel):
    """Response for POST /images/extract."""

    success: bool
    data: ExtractImagesData | None = None
    error: ErrorDetail | None = None
    processing_time_ms: float | None = None


class RenderedPage(BaseModel):
    """A rendered page image."""

    page_index: int
    width: int
    height: int
    dpi: int
    data_url: str


class PageToImageResponse(BaseModel):
    """Response for POST /images/render."""

    success: bool
    data: RenderedPage | None = None
    error: ErrorDetail | None = None
    processing_time_ms: float | None = None


# ============================================================================
# Metadata Responses
# ============================================================================


class MetadataResponse(BaseModel):
    """Response for POST /metadata/get."""

    success: bool
    data: dict | None = None
    error: ErrorDetail | None = None
    processing_time_ms: float | None = None


# ============================================================================
# Annotations Responses
# ============================================================================


class AnnotationInfo(BaseModel):
    """An annotation read from a PDF page."""

    page_index: int
    type: str
    type_code: int
    rect: dict
    content: str | None = None
    author: str | None = None
    subject: str | None = None
    stroke_color: list[float] | None = None
    fill_color: list[float] | None = None


class ReadAnnotationsData(BaseModel):
    """Annotations reading response data."""

    annotations: list[AnnotationInfo]
    total_count: int


class ReadAnnotationsResponse(BaseModel):
    """Response for POST /annotations/read."""

    success: bool
    data: ReadAnnotationsData | None = None
    error: ErrorDetail | None = None
    processing_time_ms: float | None = None


# ============================================================================
# Table Extraction Responses
# ============================================================================


class ExtractedTable(BaseModel):
    """A table extracted from a PDF page."""

    page_index: int
    table_index: int
    bbox: dict
    rows: list[list[str | None]]
    row_count: int
    col_count: int


class ExtractTablesData(BaseModel):
    """Table extraction response data."""

    tables: list[ExtractedTable]
    total_count: int


class ExtractTablesResponse(BaseModel):
    """Response for POST /text/tables."""

    success: bool
    data: ExtractTablesData | None = None
    error: ErrorDetail | None = None
    processing_time_ms: float | None = None


# ============================================================================
# Blank Page Detection
# ============================================================================


class BlankPageInfo(BaseModel):
    page_index: int
    is_blank: bool
    ink_ratio: float


class DetectBlankData(BaseModel):
    pages: list[BlankPageInfo]
    blank_count: int
    total_count: int


class DetectBlankResponse(BaseModel):
    """Response for POST /pages/detect-blank."""

    success: bool
    data: DetectBlankData | None = None
    error: ErrorDetail | None = None
    processing_time_ms: float | None = None


# ============================================================================
# Simple Classification
# ============================================================================


class ClassifyData(BaseModel):
    suggested_label: str
    confidence: float
    matched_keyword: str | None = None
    tax_year: str | None = None
    page_text_preview: str = ""


class ClassifyResponse(BaseModel):
    """Response for POST /classify/simple."""

    success: bool
    data: ClassifyData | None = None
    error: ErrorDetail | None = None
    processing_time_ms: float | None = None


# ============================================================================
# OCR (PaddleOCR) Responses
# ============================================================================


class OcrBbox(BaseModel):
    """Axis-aligned bounding box in PDF points."""

    x: float
    y: float
    w: float
    h: float


class OcrWord(BaseModel):
    """A single word detected by PaddleOCR."""

    text: str
    bbox: OcrBbox  # {x, y, w, h} in PDF points
    confidence: float


class OcrPageResult(BaseModel):
    """OCR results for a single page."""

    page_index: int
    words: list[OcrWord]
    full_text: str


class OcrData(BaseModel):
    """Rich OCR response data with per-word bounding boxes and confidence."""

    pages: list[OcrPageResult]
    total_words: int
    avg_confidence: float
    pdf_base64: str  # Base64-encoded PDF with invisible text layer for Ctrl+F


class OcrResponse(BaseModel):
    """Response for POST /text/ocr."""

    success: bool
    data: OcrData | None = None
    error: ErrorDetail | None = None
    processing_time_ms: float | None = None


# ============================================================================
# Background Task
# ============================================================================


class TaskAcceptedResponse(BaseModel):
    """Response when a request is accepted for background processing."""

    success: bool
    task_id: str
    status: str = "pending"
    message: str = "Processing started. Poll GET /tasks/{task_id} for status."
