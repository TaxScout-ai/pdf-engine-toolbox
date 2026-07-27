"""Pydantic request models for all API endpoints."""

from pydantic import BaseModel, Field


# ============================================================================
# Common Types
# ============================================================================


class PageRange(BaseModel):
    """A range of page indices (0-indexed, inclusive)."""

    start: int = Field(ge=0, description="Start page index (0-indexed)")
    end: int = Field(ge=0, description="End page index (0-indexed, inclusive)")


class PageSpec(BaseModel):
    """Specification for a single page in a build operation."""

    original_page: int = Field(ge=0, description="0-indexed page number from source")
    rotation: int = Field(default=0, description="Rotation in degrees (0, 90, 180, 270)")


class MergeSource(BaseModel):
    """A source PDF with page selections for merge/build operations."""

    url: str = Field(description="Presigned S3 GET URL for the source PDF")
    page_ranges: list[PageRange] | None = Field(
        default=None, description="Page ranges to include (all pages if omitted)"
    )
    rotations: dict[str, int] | None = Field(
        default=None, description="Page index -> rotation degrees mapping"
    )
    pages: list[PageSpec] | None = Field(
        default=None, description="Explicit page specifications (for build endpoint)"
    )


class AnnotationData(BaseModel):
    """Annotation to be flattened into PDF content."""

    page_number: int = Field(ge=1, description="1-indexed page number")
    type: str = Field(description="Annotation type: stamp, highlight, rectangle, etc.")
    x: float = Field(description="X position as percentage of page width (0-100)")
    y: float = Field(description="Y position as percentage of page height (0-100)")
    width: float | None = Field(default=None, description="Width as percentage")
    height: float | None = Field(default=None, description="Height as percentage")
    text: str | None = Field(default=None, description="Text content")
    stamp_type: str | None = Field(default=None, description="Stamp type label")
    color: str | None = Field(default="#DC2626", description="Color as hex string")
    end_x: float | None = Field(default=None, description="End X for arrows/lines")
    end_y: float | None = Field(default=None, description="End Y for arrows/lines")
    path: list[dict] | None = Field(default=None, description="Path points for freehand")


class RedactionRect(BaseModel):
    """A rectangle area to redact."""

    x0: float
    y0: float
    x1: float
    y1: float


class RedactionSpec(BaseModel):
    """Redaction specification for a page."""

    page: int = Field(ge=0, description="0-indexed page number")
    rects: list[RedactionRect] | None = Field(default=None, description="Rectangles to redact")
    text_pattern: str | None = Field(
        default=None, description="Regex pattern to find and redact"
    )
    fill_color: list[float] | None = Field(
        default=None, description="Fill color as [R, G, B] 0-1 values"
    )


class BookmarkEntry(BaseModel):
    """A bookmark/TOC entry."""

    label: str = Field(description="Bookmark label text")
    page: int = Field(ge=0, description="0-indexed target page")
    level: int = Field(default=1, ge=1, description="Nesting level (1 = top level)")


# ============================================================================
# Endpoint Request Models
# ============================================================================


class InfoRequest(BaseModel):
    """Request for POST /info."""

    source_url: str = Field(description="Presigned S3 GET URL for the PDF")


class RotateRequest(BaseModel):
    """Request for POST /pages/rotate."""

    source_url: str
    pages: list[int] = Field(description="0-indexed page numbers to rotate")
    degrees: int = Field(description="Rotation degrees: 90, 180, 270, -90")


class SplitRequest(BaseModel):
    """Request for POST /pages/split."""

    source_url: str
    page_ranges: list[PageRange]
    rotations: dict[str, int] | None = Field(
        default=None, description="Page index -> rotation degrees"
    )


class MergeRequest(BaseModel):
    """Request for POST /pages/merge."""

    sources: list[MergeSource]


class ReorderRequest(BaseModel):
    """Request for POST /pages/reorder."""

    source_url: str
    new_order: list[int] = Field(description="New page order as list of 0-indexed page numbers")


class DeletePagesRequest(BaseModel):
    """Request for POST /pages/delete."""

    source_url: str
    pages_to_delete: list[int] = Field(description="0-indexed page numbers to delete")


class DeskewRequest(BaseModel):
    """Request for POST /transform/deskew."""

    source_url: str
    pages: list[int] | None = Field(default=None, description="Pages to deskew (all if omitted)")


class OrientationRequest(BaseModel):
    """Request for POST /transform/orientation."""

    source_url: str
    pages: list[int] | None = Field(
        default=None, description="0-indexed pages to inspect (all if omitted)"
    )
    scanned_only: bool = Field(
        default=True,
        description="Skip pages with a usable native text layer",
    )
    min_confidence: float = Field(
        default=0.85,
        ge=0,
        le=1,
        description="Minimum classifier confidence required to return a rotation",
    )


class CompressRequest(BaseModel):
    """Request for POST /transform/compress."""

    source_url: str
    quality: str = Field(default="medium", description="Quality: low, medium, high")
    max_image_dpi: int = Field(default=150, description="Max DPI for images")


class FlattenRequest(BaseModel):
    """Request for POST /transform/flatten."""

    source_url: str
    annotations: list[AnnotationData]


class RedactRequest(BaseModel):
    """Request for POST /redact."""

    source_url: str
    redactions: list[RedactionSpec]


class DetectPiiRequest(BaseModel):
    """Request for POST /redact/detect-pii."""

    source_url: str
    patterns: list[str] = Field(
        default=["ssn", "ein"],
        description="PII patterns to detect: ssn, ein, phone, email",
    )


class TextExtractRequest(BaseModel):
    """Request for POST /text/extract."""

    source_url: str
    pages: list[int] | None = Field(default=None, description="Pages to extract (all if omitted)")
    include_positions: bool = Field(default=True, description="Include text position data")


class TextSearchRequest(BaseModel):
    """Request for POST /text/search."""

    source_url: str
    query: str = Field(min_length=1, description="Search query text")
    case_sensitive: bool = Field(default=False)


class BookmarksRequest(BaseModel):
    """Request for POST /text/bookmarks."""

    source_url: str
    bookmarks: list[BookmarkEntry]


class ThumbnailsRequest(BaseModel):
    """Request for POST /thumbnails."""

    source_url: str
    pages: list[int] | None = Field(default=None, description="Pages (all if omitted)")
    width: int = Field(default=200, ge=50, le=800, description="Thumbnail width in pixels")
    format: str = Field(default="webp", description="Image format: png, webp, jpeg")
    quality: int = Field(default=80, ge=1, le=100, description="Image quality for webp/jpeg")


class BuildRequest(BaseModel):
    """Request for POST /build (commit)."""

    sources: list[MergeSource]
    annotations: list[AnnotationData] | None = Field(default=None)
    flatten_annotations: bool = Field(default=False)
    compress: bool = Field(default=False)
    bookmarks: list[BookmarkEntry] | None = Field(default=None)


# ============================================================================
# Image Operations
# ============================================================================


class ExtractImagesRequest(BaseModel):
    """Request for POST /images/extract."""

    source_url: str
    pages: list[int] | None = Field(default=None, description="Pages to extract from (all if omitted)")
    min_width: int = Field(default=50, ge=1, description="Minimum image width to include")
    min_height: int = Field(default=50, ge=1, description="Minimum image height to include")
    format: str = Field(default="png", description="Output format: png, jpeg")


class PageToImageRequest(BaseModel):
    """Request for POST /images/render."""

    source_url: str
    page: int = Field(ge=0, description="0-indexed page number")
    dpi: int = Field(default=150, ge=72, le=600, description="Resolution in DPI")
    format: str = Field(default="png", description="Output format: png, jpeg, webp")
    quality: int = Field(default=85, ge=1, le=100, description="Image quality for webp/jpeg")


# ============================================================================
# Metadata Operations
# ============================================================================


class SetMetadataRequest(BaseModel):
    """Request for POST /metadata/set."""

    source_url: str
    title: str | None = Field(default=None)
    author: str | None = Field(default=None)
    subject: str | None = Field(default=None)
    keywords: str | None = Field(default=None)
    creator: str | None = Field(default=None)
    producer: str | None = Field(default=None)


class GetMetadataRequest(BaseModel):
    """Request for POST /metadata/get."""

    source_url: str


# ============================================================================
# Page Crop
# ============================================================================


class CropMargins(BaseModel):
    """Margins for cropping in PDF points."""

    top: float = Field(default=0, ge=0)
    right: float = Field(default=0, ge=0)
    bottom: float = Field(default=0, ge=0)
    left: float = Field(default=0, ge=0)


class CropBox(BaseModel):
    """Explicit crop box in PDF points."""

    x0: float
    y0: float
    x1: float
    y1: float


class CropPagesRequest(BaseModel):
    """Request for POST /pages/crop."""

    source_url: str
    pages: list[int] | None = Field(default=None, description="Pages to crop (all if omitted)")
    margins: CropMargins | None = Field(default=None, description="Inset from each edge")
    crop_box: CropBox | None = Field(default=None, description="Explicit crop box")


# ============================================================================
# Watermark
# ============================================================================


class WatermarkRequest(BaseModel):
    """Request for POST /transform/watermark."""

    source_url: str
    text: str = Field(min_length=1, description="Watermark text with optional {user_name} and {date} placeholders")
    pages: list[int] | None = Field(default=None, description="Pages to watermark (all if omitted)")
    font_size: float = Field(default=60.0, ge=8, le=200)
    color: str = Field(default="#CCCCCC", description="Hex color")
    opacity: float = Field(default=0.3, ge=0, le=1)
    rotation: float = Field(default=-45.0, description="Rotation angle in degrees")
    user_name: str | None = Field(default=None, description="Value for {user_name} placeholder")
    date: str | None = Field(default=None, description="Value for {date} placeholder (defaults to today)")


# ============================================================================
# Annotations (Read)
# ============================================================================


class ReadAnnotationsRequest(BaseModel):
    """Request for POST /annotations/read."""

    source_url: str
    pages: list[int] | None = Field(default=None, description="Pages to read (all if omitted)")


# ============================================================================
# Security
# ============================================================================


class EncryptRequest(BaseModel):
    """Request for POST /security/encrypt."""

    source_url: str
    user_password: str = Field(default="", description="Password to open PDF (empty = no view password)")
    owner_password: str = Field(min_length=1, description="Owner password for full permissions")
    permissions: int = Field(
        default=-1,
        description="Permission flags bitmask (-1 = all permissions)",
    )


class DecryptRequest(BaseModel):
    """Request for POST /security/decrypt."""

    source_url: str
    password: str = Field(min_length=1, description="Password to unlock the PDF")


class SanitizeRequest(BaseModel):
    """Request for POST /security/sanitize."""

    source_url: str
    remove_metadata: bool = Field(default=True, description="Clear document metadata")
    remove_javascript: bool = Field(default=True, description="Remove embedded JavaScript")
    remove_links: bool = Field(default=False, description="Remove all hyperlinks")
    remove_annotations: bool = Field(default=False, description="Remove all annotations")


# ============================================================================
# Table Extraction
# ============================================================================


class ExtractTablesRequest(BaseModel):
    """Request for POST /text/tables."""

    source_url: str
    pages: list[int] | None = Field(default=None, description="Pages to extract from (all if omitted)")
    strategy: str = Field(
        default="auto",
        description=(
            "Table detection strategy: "
            "'auto' (PyMuPDF for digital, PP-Structure for scans), "
            "'pymupdf' (fast, digital only), "
            "'ppstructure' (neural network, best for scanned forms)"
        ),
    )
    language: str = Field(default="en", description="PaddleOCR language code for PP-Structure")
    dpi: int = Field(default=300, ge=150, le=600, description="Rendering DPI for PP-Structure")


# ============================================================================
# Page Labels
# ============================================================================


class PageLabelRule(BaseModel):
    """A page labeling rule."""

    start_page: int = Field(ge=0, description="0-indexed page where rule begins")
    prefix: str = Field(default="", description="Text prefix")
    style: str = Field(
        default="D",
        description="D=decimal, r=roman lower, R=roman upper, a=alpha lower, A=alpha upper",
    )
    first_page_num: int = Field(default=1, ge=1, description="Starting number")


class SetPageLabelsRequest(BaseModel):
    """Request for POST /pages/labels."""

    source_url: str
    labels: list[PageLabelRule]


# ============================================================================
# Repair
# ============================================================================


class RepairRequest(BaseModel):
    """Request for POST /repair."""

    source_url: str


# ============================================================================
# Image to PDF
# ============================================================================


class ImageToPdfRequest(BaseModel):
    """Request for POST /convert/from-image.

    Accepts image URLs (presigned S3) and optional filenames for format detection.
    """

    image_urls: list[str] = Field(min_length=1, description="Presigned URLs for images")
    filenames: list[str] | None = Field(default=None, description="Original filenames (for HEIC detection)")


# ============================================================================
# Office to PDF
# ============================================================================


class OfficeToPdfRequest(BaseModel):
    """Request for POST /convert/from-office.

    Accepts a presigned URL to an office document (DOCX, XLSX, PPTX, etc.)
    and converts it to PDF via LibreOffice.
    """

    source_url: str = Field(description="Presigned URL for the office document")
    filename: str = Field(description="Original filename with extension (e.g. 'report.xlsx')")


# ============================================================================
# Blank Page Detection
# ============================================================================


class DetectBlankRequest(BaseModel):
    """Request for POST /pages/detect-blank."""

    source_url: str
    ink_threshold: float = Field(
        default=0.01, ge=0.0, le=0.5,
        description="Max ink ratio to consider a page blank (0.01 = 1%)",
    )


# ============================================================================
# Simple Classification
# ============================================================================


class ClassifyRequest(BaseModel):
    """Request for POST /classify/simple."""

    source_url: str


# ============================================================================
# OCR
# ============================================================================


class OcrRequest(BaseModel):
    """Request for POST /text/ocr."""

    source_url: str
    pages: list[int] | None = Field(default=None, description="Pages to OCR (all if omitted)")
    language: str = Field(default="en", description="PaddleOCR language code (en, fr, de, ch, etc.)")
    dpi: int = Field(default=300, ge=150, le=600, description="Rendering DPI for OCR")
