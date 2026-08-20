"""Core PDF operations using PyMuPDF (fitz)."""

import base64
import datetime
import io
import math
import re

import cv2
import fitz  # PyMuPDF
import numpy as np
import structlog
from PIL import Image

from app.utils.errors import (
    PageOutOfRangeError,
    PdfCorruptError,
    PdfPasswordAuthorizationError,
    PdfPasswordNotRequiredError,
)

log = structlog.get_logger()


def _open_pdf(pdf_bytes: bytes) -> fitz.Document:
    """Open a PDF from bytes, raising PdfCorruptError on failure."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return doc
    except Exception as e:
        raise PdfCorruptError(f"Cannot open PDF: {e}")


def _validate_pages(doc: fitz.Document, pages: list[int]) -> None:
    """Validate that all page indices are within range."""
    total = len(doc)
    for p in pages:
        if p < 0 or p >= total:
            raise PageOutOfRangeError(p, total)


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert hex color string to (r, g, b) tuple with 0-1 values."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return (r / 255.0, g / 255.0, b / 255.0)
    return (0, 0, 0)


# ============================================================================
# Info
# ============================================================================

_AUTHENTICATION_LEVELS = {
    2: "user",
    4: "owner",
    6: "user-and-owner",
}


def _normalized_permissions(doc: fitz.Document) -> dict[str, bool]:
    """Translate PyMuPDF's bitmask into explicit policy inputs."""
    permissions = doc.permissions
    return {
        "copy": bool(permissions & fitz.PDF_PERM_COPY),
        "modify": bool(permissions & fitz.PDF_PERM_MODIFY),
        "annotate": bool(permissions & fitz.PDF_PERM_ANNOTATE),
        "print": bool(permissions & fitz.PDF_PERM_PRINT),
    }


def _digital_signature_state(doc: fitz.Document) -> tuple[bool | None, str]:
    """Return conservative signature evidence without modifying the document."""
    signature_flags = doc.get_sigflags()
    if signature_flags == 3:
        return True, "signatures-may-be-invalidated"
    if signature_flags == 1:
        return True, "signature-fields-present"

    try:
        for page in doc:
            widgets = page.widgets()
            if widgets and any(
                widget.field_type == fitz.PDF_WIDGET_TYPE_SIGNATURE
                for widget in widgets
            ):
                return True, "signature-fields-present"
    except (RuntimeError, ValueError):
        return None, "unknown"

    if signature_flags == -1:
        return None, "unknown"
    return False, "none-detected"


def _has_encryption(doc: fitz.Document) -> bool:
    """Detect both user-password and owner-only encryption."""
    metadata = doc.metadata or {}
    return bool(doc.needs_pass or metadata.get("encryption"))


def get_info(pdf_bytes: bytes) -> dict:
    """Get policy-safe PDF information without reading locked content."""
    doc = _open_pdf(pdf_bytes)
    try:
        requires_password = bool(doc.needs_pass)
        if requires_password:
            return {
                "page_count": 0,
                "pages": [],
                "is_encrypted": True,
                "requires_password": True,
                "authentication_level": "none",
                "permissions": None,
                "has_digital_signatures": None,
                "signature_state": "unknown",
                "metadata": None,
            }

        is_encrypted = _has_encryption(doc)
        authentication_level = "none"
        if is_encrypted:
            authentication_level = _AUTHENTICATION_LEVELS.get(
                doc.authenticate(""),
                "none",
            )

        pages = []
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text().strip()
            pages.append(
                {
                    "index": i,
                    "width": round(page.rect.width, 2),
                    "height": round(page.rect.height, 2),
                    "rotation": page.rotation,
                    "has_text": bool(text),
                }
            )

        has_signatures, signature_state = _digital_signature_state(doc)
        return {
            "page_count": len(doc),
            "pages": pages,
            "is_encrypted": is_encrypted,
            "requires_password": False,
            "authentication_level": authentication_level,
            "permissions": _normalized_permissions(doc),
            "has_digital_signatures": has_signatures,
            "signature_state": signature_state,
            "metadata": doc.metadata or {},
        }
    finally:
        doc.close()


def authorize_and_unlock_pdf(
    pdf_bytes: bytes,
    password: str,
    *,
    authority_attested: bool,
) -> dict:
    """Authenticate one known password and return an in-memory derivative.

    This function never guesses or retries passwords. User-level access is
    allowed only when the PDF explicitly permits copying/extraction.
    """
    try:
        password_bytes = password.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise PdfPasswordAuthorizationError() from exc
    if not authority_attested or not 1 <= len(password_bytes) <= 256:
        raise PdfPasswordAuthorizationError()

    doc = _open_pdf(pdf_bytes)
    try:
        if not _has_encryption(doc):
            raise PdfPasswordNotRequiredError()

        authentication_level = _AUTHENTICATION_LEVELS.get(doc.authenticate(password))
        if authentication_level is None:
            raise PdfPasswordAuthorizationError()

        permissions = _normalized_permissions(doc)
        owner_authenticated = authentication_level in {"owner", "user-and-owner"}
        if not owner_authenticated and not permissions["copy"]:
            raise PdfPasswordAuthorizationError()

        has_signatures, signature_state = _digital_signature_state(doc)
        unlocked_pdf = doc.tobytes(
            garbage=4,
            deflate=True,
            encryption=fitz.PDF_ENCRYPT_NONE,
        )
        return {
            "authentication_level": authentication_level,
            "permissions": permissions,
            "has_digital_signatures": has_signatures,
            "signature_state": signature_state,
            "pdf_bytes": unlocked_pdf,
        }
    finally:
        doc.close()


# ============================================================================
# Page Operations
# ============================================================================


def rotate_pages(pdf_bytes: bytes, pages: list[int], degrees: int) -> bytes:
    """Rotate specified pages by given degrees."""
    doc = _open_pdf(pdf_bytes)
    _validate_pages(doc, pages)

    for page_idx in pages:
        page = doc[page_idx]
        current = page.rotation
        page.set_rotation((current + degrees) % 360)

    return doc.tobytes(garbage=4, deflate=True)


def split_pages(
    pdf_bytes: bytes,
    page_indices: list[int],
    rotations: dict[int, int] | None = None,
) -> bytes:
    """Extract specified pages into a new PDF."""
    src = _open_pdf(pdf_bytes)
    _validate_pages(src, page_indices)

    dst = fitz.open()
    for idx in page_indices:
        dst.insert_pdf(src, from_page=idx, to_page=idx)
        if rotations and idx in rotations:
            dst[-1].set_rotation(rotations[idx])

    return dst.tobytes(garbage=4, deflate=True)


def merge_pdfs(sources: list[dict]) -> bytes:
    """Merge pages from multiple source PDFs.

    Args:
        sources: List of dicts with keys:
            - bytes: PDF content as bytes
            - page_ranges: Optional list of {start, end} dicts
            - rotations: Optional dict of page_index -> degrees
            - pages: Optional list of {original_page, rotation} dicts
    """
    dst = fitz.open()

    for source in sources:
        src_bytes = source["bytes"]
        src = _open_pdf(src_bytes)

        if source.get("pages"):
            # Explicit page list (used by build endpoint)
            for page_spec in source["pages"]:
                page_idx = page_spec["original_page"]
                rotation = page_spec.get("rotation", 0)

                if page_idx < 0 or page_idx >= len(src):
                    raise PageOutOfRangeError(page_idx, len(src))

                dst.insert_pdf(src, from_page=page_idx, to_page=page_idx)
                if rotation:
                    dst[-1].set_rotation(rotation)

        elif source.get("page_ranges"):
            # Page ranges
            rotations = source.get("rotations", {})
            for pr in source["page_ranges"]:
                start = pr["start"]
                end = pr["end"]
                for idx in range(start, end + 1):
                    if idx < 0 or idx >= len(src):
                        raise PageOutOfRangeError(idx, len(src))
                    dst.insert_pdf(src, from_page=idx, to_page=idx)
                    rot = rotations.get(str(idx), 0)
                    if rot:
                        dst[-1].set_rotation(rot)
        else:
            # All pages
            dst.insert_pdf(src)

    return dst.tobytes(garbage=4, deflate=True)


def reorder_pages(pdf_bytes: bytes, new_order: list[int]) -> bytes:
    """Reorder pages according to new_order list."""
    src = _open_pdf(pdf_bytes)
    _validate_pages(src, new_order)

    if sorted(new_order) != list(range(len(src))):
        raise PdfCorruptError(
            f"new_order must be a permutation of pages 0..{len(src) - 1}"
        )

    dst = fitz.open()
    for idx in new_order:
        dst.insert_pdf(src, from_page=idx, to_page=idx)

    return dst.tobytes(garbage=4, deflate=True)


def delete_pages(pdf_bytes: bytes, pages_to_delete: list[int]) -> bytes:
    """Remove specified pages from PDF."""
    doc = _open_pdf(pdf_bytes)
    _validate_pages(doc, pages_to_delete)

    # Delete in reverse order to maintain indices
    for idx in sorted(pages_to_delete, reverse=True):
        doc.delete_page(idx)

    if len(doc) == 0:
        raise PdfCorruptError("Cannot delete all pages from PDF")

    return doc.tobytes(garbage=4, deflate=True)


# ============================================================================
# Transform Operations
# ============================================================================


def _detect_skew_angle(pix: fitz.Pixmap) -> float:
    """Detect the skew angle of a page image using OpenCV.

    Uses Canny edge detection + Hough line transform to find
    the dominant line angle, which corresponds to text line skew.

    Returns angle in degrees (positive = clockwise skew).
    """
    # Convert PyMuPDF pixmap to numpy array for OpenCV
    img_bytes = pix.tobytes("png")
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return 0.0

    # Edge detection
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    # Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100, minLineLength=img.shape[1] // 4, maxLineGap=10,
    )

    if lines is None or len(lines) == 0:
        return 0.0

    # Calculate angles of all detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # Only consider near-horizontal lines (text lines)
        if abs(angle) < 15:
            angles.append(angle)

    if not angles:
        return 0.0

    # Return median angle (robust against outliers)
    angles.sort()
    return angles[len(angles) // 2]


def deskew_pages(pdf_bytes: bytes, pages: list[int] | None = None) -> bytes:
    """Auto-straighten scanned pages using OpenCV for angle detection.

    For each target page:
      1. Render page as high-res image.
      2. Use OpenCV (Canny + Hough) to detect the skew angle.
      3. If skew > 0.5 degrees, re-render the deskewed page with PyMuPDF
         by replacing the page content with a rotated image.
    """
    doc = _open_pdf(pdf_bytes)

    target_pages = pages if pages is not None else list(range(len(doc)))
    if pages:
        _validate_pages(doc, pages)

    for page_idx in target_pages:
        page = doc[page_idx]

        # High-res render for accurate angle detection
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)

        angle = _detect_skew_angle(pix)

        if abs(angle) < 0.5:
            # Skew too small to bother correcting
            continue

        log.info("deskew_detected", page=page_idx, angle=round(angle, 2))

        # Deskew: render page as image, rotate, replace page
        # Use higher resolution for quality
        hi_mat = fitz.Matrix(3, 3)
        hi_pix = page.get_pixmap(matrix=hi_mat)
        img_bytes = hi_pix.tobytes("png")

        # Rotate with OpenCV (white background fill)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img, rotation_matrix, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )

        # Encode back to PNG
        _, rotated_png = cv2.imencode(".png", rotated)

        # Replace page content with the deskewed image
        page_rect = page.rect
        page.clean_contents()
        # Remove existing content and insert rotated image
        page.insert_image(page_rect, stream=rotated_png.tobytes())

    return doc.tobytes(garbage=4, deflate=True)


_doc_orientation_model = None


def _get_doc_orientation_model():
    """Return the cached PaddleX cardinal document-orientation classifier."""
    global _doc_orientation_model
    if _doc_orientation_model is None:
        try:
            from paddlex import create_model
        except ImportError as exc:
            raise PdfCorruptError(
                "Document orientation detection requires paddlex"
            ) from exc
        _doc_orientation_model = create_model(
            model_name="PP-LCNet_x1_0_doc_ori"
        )
    return _doc_orientation_model


def detect_page_orientations(
    pdf_bytes: bytes,
    pages: list[int] | None = None,
    *,
    scanned_only: bool = True,
    min_confidence: float = 0.85,
) -> dict:
    """Detect 90/180/270-degree raster rotation before routing and OCR.

    PDF rotation metadata does not describe scans whose pixels were captured
    sideways. The lightweight PaddleX classifier reads the rendered page and
    returns the clockwise correction required to make its text upright.
    Native-text pages are skipped by default because their orientation is
    already represented by the PDF content/rotation model.
    """
    doc = _open_pdf(pdf_bytes)
    target_pages = pages if pages is not None else list(range(len(doc)))
    if pages:
        _validate_pages(doc, pages)

    candidate_pages: list[int] = []
    images: list[np.ndarray] = []
    for page_idx in target_pages:
        page = doc[page_idx]
        if scanned_only and len(page.get_text("text").strip()) >= 20:
            continue

        # 108 DPI is enough for the cardinal classifier and keeps the
        # pre-router control cheap even for long brokerage statements.
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        candidate_pages.append(page_idx)
        images.append(image)

    if not images:
        return {
            "pages": [],
            "checked_page_count": 0,
            "skipped_page_count": len(target_pages),
        }

    model = _get_doc_orientation_model()
    predictions = list(model.predict(images, batch_size=min(8, len(images))))
    rotations: list[dict] = []
    for page_idx, prediction in zip(candidate_pages, predictions, strict=True):
        payload = prediction.json if hasattr(prediction, "json") else prediction
        if callable(payload):
            payload = payload()
        if isinstance(payload, dict):
            payload = payload.get("res", payload)
        if not isinstance(payload, dict):
            continue

        labels = payload.get("label_names") or []
        scores = payload.get("scores") or []
        try:
            rotation = int(labels[0])
            confidence = float(scores[0])
        except (IndexError, TypeError, ValueError):
            continue

        if rotation not in (90, 180, 270) or confidence < min_confidence:
            continue
        correction = (360 - rotation) % 360
        rotations.append(
            {
                "page": page_idx,
                "rotation": correction,
                "detected_orientation": rotation,
                "confidence": confidence,
            }
        )

    return {
        "pages": rotations,
        "checked_page_count": len(candidate_pages),
        "skipped_page_count": len(target_pages) - len(candidate_pages),
    }


def compress_pdf(
    pdf_bytes: bytes,
    quality: str = "medium",
    max_image_dpi: int = 150,
) -> bytes:
    """Compress PDF by optimizing images and cleaning structure.

    Args:
        pdf_bytes: Source PDF
        quality: low, medium, or high
        max_image_dpi: Maximum DPI for raster images
    """
    doc = _open_pdf(pdf_bytes)

    # PyMuPDF save options for compression
    # garbage=4: Maximum garbage collection (removes unused objects)
    # deflate=True: Compress streams
    # clean=True: Clean and optimize content streams
    return doc.tobytes(
        garbage=4,
        deflate=True,
        clean=True,
    )


def flatten_annotations(pdf_bytes: bytes, annotations: list[dict]) -> bytes:
    """Burn annotations into PDF content as permanent marks.

    Converts percentage-based coordinates to PDF points and draws
    using PyMuPDF drawing primitives.
    """
    doc = _open_pdf(pdf_bytes)

    for annot in annotations:
        page_num = annot["page_number"] - 1  # Convert 1-indexed to 0-indexed
        if page_num < 0 or page_num >= len(doc):
            continue

        page = doc[page_num]
        rect = page.rect

        # Convert percentage coordinates to PDF points
        x = annot["x"] / 100.0 * rect.width
        y = annot["y"] / 100.0 * rect.height

        annot_type = annot.get("type", "")
        color = _hex_to_rgb(annot.get("color", "#DC2626"))

        if annot_type == "stamp":
            stamp_text = annot.get("stamp_type", annot.get("text", ""))
            if stamp_text:
                page.insert_text(
                    fitz.Point(x, y + 12),  # Offset for baseline
                    stamp_text,
                    fontsize=11,
                    color=color,
                    fontname="helv",
                )

        elif annot_type == "highlight":
            w = (annot.get("width", 0) / 100.0 * rect.width) if annot.get("width") else 50
            h = (annot.get("height", 0) / 100.0 * rect.height) if annot.get("height") else 12
            highlight_rect = fitz.Rect(x, y, x + w, y + h)
            # Semi-transparent highlight
            page.draw_rect(highlight_rect, color=color, fill=color, fill_opacity=0.3)

        elif annot_type == "rectangle":
            w = (annot.get("width", 0) / 100.0 * rect.width) if annot.get("width") else 50
            h = (annot.get("height", 0) / 100.0 * rect.height) if annot.get("height") else 30
            draw_rect = fitz.Rect(x, y, x + w, y + h)
            page.draw_rect(draw_rect, color=color, width=1.5)

        elif annot_type == "circle":
            w = (annot.get("width", 0) / 100.0 * rect.width) if annot.get("width") else 30
            h = (annot.get("height", 0) / 100.0 * rect.height) if annot.get("height") else 30
            center = fitz.Point(x + w / 2, y + h / 2)
            oval_rect = fitz.Rect(x, y, x + w, y + h)
            page.draw_oval(oval_rect, color=color, width=1.5)

        elif annot_type in ("arrow", "line"):
            end_x = (annot.get("end_x", 0) / 100.0 * rect.width) if annot.get("end_x") else x + 50
            end_y = (annot.get("end_y", 0) / 100.0 * rect.height) if annot.get("end_y") else y
            page.draw_line(fitz.Point(x, y), fitz.Point(end_x, end_y), color=color, width=1.5)

        elif annot_type == "sticky_note":
            text = annot.get("text", "")
            if text:
                # Draw a small note indicator + text
                note_rect = fitz.Rect(x, y, x + 100, y + 40)
                page.draw_rect(note_rect, color=(1, 1, 0.7), fill=(1, 1, 0.7))
                page.draw_rect(note_rect, color=color, width=0.5)
                page.insert_textbox(
                    note_rect,
                    text,
                    fontsize=8,
                    color=(0, 0, 0),
                    fontname="helv",
                )

        elif annot_type == "freehand":
            path_points = annot.get("path", [])
            if len(path_points) >= 2:
                for i in range(len(path_points) - 1):
                    p1 = path_points[i]
                    p2 = path_points[i + 1]
                    x1 = p1["x"] / 100.0 * rect.width
                    y1 = p1["y"] / 100.0 * rect.height
                    x2 = p2["x"] / 100.0 * rect.width
                    y2 = p2["y"] / 100.0 * rect.height
                    page.draw_line(
                        fitz.Point(x1, y1), fitz.Point(x2, y2), color=color, width=1.5
                    )

        elif annot_type == "checkmark":
            # Draw a checkmark symbol
            page.insert_text(
                fitz.Point(x, y + 14),
                "\u2713",  # Unicode checkmark
                fontsize=16,
                color=(0, 0.6, 0),
                fontname="helv",
            )

        elif annot_type == "x_mark":
            page.insert_text(
                fitz.Point(x, y + 14),
                "\u2717",  # Unicode X mark
                fontsize=16,
                color=color,
                fontname="helv",
            )

        elif annot_type == "date_stamp":
            import datetime

            date_text = annot.get("text", datetime.date.today().isoformat())
            page.insert_text(
                fitz.Point(x, y + 10),
                date_text,
                fontsize=9,
                color=color,
                fontname="helv",
            )

    return doc.tobytes(garbage=4, deflate=True)


# ============================================================================
# Redaction
# ============================================================================


def redact_content(pdf_bytes: bytes, redactions: list[dict]) -> bytes:
    """Apply true redactions - permanently remove content under redacted areas."""
    doc = _open_pdf(pdf_bytes)

    for redaction in redactions:
        page_idx = redaction["page"]
        if page_idx < 0 or page_idx >= len(doc):
            continue

        page = doc[page_idx]
        fill = redaction.get("fill_color", [0, 0, 0])
        fill_tuple = tuple(float(c) for c in fill) if fill else (0, 0, 0)

        # Add redaction annotations for rectangles
        for rect_data in redaction.get("rects", []):
            rect = fitz.Rect(rect_data["x0"], rect_data["y0"], rect_data["x1"], rect_data["y1"])
            page.add_redact_annot(rect, fill=fill_tuple)

        # Add redaction annotations for text patterns
        text_pattern = redaction.get("text_pattern")
        if text_pattern:
            text_instances = page.search_for(text_pattern)
            for inst in text_instances:
                page.add_redact_annot(inst, fill=fill_tuple)

        # Apply all redactions on this page (permanently removes content)
        page.apply_redactions()

    return doc.tobytes(garbage=4, deflate=True)


# PII regex patterns
PII_PATTERNS = {
    "ssn": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
    "ein": r"\b\d{2}[-\s]?\d{7}\b",
    "phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
}


def detect_pii(pdf_bytes: bytes, patterns: list[str]) -> list[dict]:
    """Detect PII patterns in PDF text.

    Returns list of detections with page, pattern type, text, and bounding rect.
    """
    doc = _open_pdf(pdf_bytes)
    detections = []

    for i in range(len(doc)):
        page = doc[i]
        page_text = page.get_text()

        for pattern_name in patterns:
            regex = PII_PATTERNS.get(pattern_name)
            if not regex:
                continue

            for match in re.finditer(regex, page_text):
                matched_text = match.group()

                # Mask sensitive parts for display
                if pattern_name == "ssn":
                    display_text = f"***-**-{matched_text[-4:]}"
                elif pattern_name == "ein":
                    display_text = f"**-***{matched_text[-4:]}"
                else:
                    display_text = matched_text

                # Find bounding rectangles for the match
                rects = page.search_for(matched_text)
                for rect in rects:
                    detections.append(
                        {
                            "page": i,
                            "pattern": pattern_name,
                            "text": display_text,
                            "rect": {
                                "x0": round(rect.x0, 2),
                                "y0": round(rect.y0, 2),
                                "x1": round(rect.x1, 2),
                                "y1": round(rect.y1, 2),
                            },
                        }
                    )

    return detections


# ============================================================================
# Text Operations
# ============================================================================


def extract_text(
    pdf_bytes: bytes,
    pages: list[int] | None = None,
    include_positions: bool = True,
) -> list[dict]:
    """Extract text from PDF pages.

    Returns list of page text data with optional position information.
    """
    doc = _open_pdf(pdf_bytes)
    target_pages = pages if pages is not None else list(range(len(doc)))
    if pages:
        _validate_pages(doc, pages)

    result = []
    for page_idx in target_pages:
        page = doc[page_idx]
        full_text = page.get_text()

        page_data = {
            "page_index": page_idx,
            "text": full_text,
        }

        if include_positions:
            # Get structured text with positions
            text_dict = page.get_text("dict")
            blocks = []
            for block in text_dict.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    block_text = ""
                    max_font_size = 0
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            block_text += span.get("text", "")
                            max_font_size = max(max_font_size, span.get("size", 0))
                        block_text += "\n"

                    if block_text.strip():
                        bbox = block["bbox"]
                        blocks.append(
                            {
                                "bbox": {
                                    "x0": round(bbox[0], 2),
                                    "y0": round(bbox[1], 2),
                                    "x1": round(bbox[2], 2),
                                    "y1": round(bbox[3], 2),
                                },
                                "text": block_text.strip(),
                                "font_size": round(max_font_size, 1),
                            }
                        )

            page_data["blocks"] = blocks

        result.append(page_data)

    return result


def search_text(pdf_bytes: bytes, query: str, case_sensitive: bool = False) -> dict:
    """Search for text across all pages.

    Returns total matches count and list of match locations.
    """
    doc = _open_pdf(pdf_bytes)
    matches = []

    flags = 0 if case_sensitive else fitz.TEXT_PRESERVE_WHITESPACE

    for i in range(len(doc)):
        page = doc[i]

        # search_for returns list of Rect objects
        rects = page.search_for(query)

        for rect in rects:
            # Get surrounding context
            context = ""
            blocks = page.get_text("blocks")
            for block in blocks:
                block_rect = fitz.Rect(block[:4])
                if block_rect.intersects(rect):
                    text = block[4] if len(block) > 4 else ""
                    # Get a snippet around the match
                    context = text[:150].strip()
                    break

            matches.append(
                {
                    "page_index": i,
                    "text": query,
                    "rect": {
                        "x0": round(rect.x0, 2),
                        "y0": round(rect.y0, 2),
                        "x1": round(rect.x1, 2),
                        "y1": round(rect.y1, 2),
                    },
                    "context": context,
                }
            )

    return {
        "total_matches": len(matches),
        "matches": matches,
    }


def add_bookmarks(pdf_bytes: bytes, bookmarks: list[dict]) -> bytes:
    """Add bookmarks (table of contents) to a PDF.

    Args:
        pdf_bytes: Source PDF
        bookmarks: List of {label, page, level} dicts
    """
    doc = _open_pdf(pdf_bytes)

    # Build TOC list for PyMuPDF: [level, title, page_number (1-indexed)]
    toc = []
    for bm in bookmarks:
        level = bm.get("level", 1)
        label = bm["label"]
        page = bm["page"] + 1  # Convert 0-indexed to 1-indexed
        toc.append([level, label, page])

    doc.set_toc(toc)
    return doc.tobytes(garbage=4, deflate=True)


# ============================================================================
# Thumbnails
# ============================================================================


def _pixmap_to_format(pix: fitz.Pixmap, fmt: str, quality: int = 80) -> tuple[bytes, str]:
    """Convert a PyMuPDF Pixmap to the requested image format.

    Supports png, jpeg, and webp (via Pillow for WebP).
    Returns (image_bytes, mime_type).
    """
    if fmt == "webp":
        # PyMuPDF doesn't support WebP natively - use Pillow
        pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        buf = io.BytesIO()
        pil_img.save(buf, format="WEBP", quality=quality)
        return buf.getvalue(), "image/webp"
    elif fmt == "jpeg":
        return pix.tobytes("jpeg"), "image/jpeg"
    else:
        return pix.tobytes("png"), "image/png"


def generate_thumbnails(
    pdf_bytes: bytes,
    pages: list[int] | None = None,
    width: int = 200,
    fmt: str = "webp",
    quality: int = 80,
) -> list[dict]:
    """Generate page thumbnails as base64 data URLs.

    Args:
        pdf_bytes: Source PDF
        pages: Page indices (all if None)
        width: Target width in pixels
        fmt: Image format (png, jpeg, webp)
        quality: Image quality 1-100 (for jpeg/webp)
    """
    doc = _open_pdf(pdf_bytes)
    target_pages = pages if pages is not None else list(range(len(doc)))
    if pages:
        _validate_pages(doc, pages)

    thumbnails = []
    for page_idx in target_pages:
        page = doc[page_idx]

        # Calculate scale to achieve target width
        page_width = page.rect.width
        scale = width / page_width if page_width > 0 else 1
        mat = fitz.Matrix(scale, scale)

        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes, mime = _pixmap_to_format(pix, fmt, quality)

        data_url = f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"

        thumbnails.append(
            {
                "page_index": page_idx,
                "data_url": data_url,
                "width": pix.width,
                "height": pix.height,
            }
        )

    return thumbnails


# ============================================================================
# Build (Commit)
# ============================================================================


def build_pdf(
    sources: list[dict],
    annotations: list[dict] | None = None,
    flatten: bool = False,
    compress: bool = False,
    bookmarks: list[dict] | None = None,
) -> bytes:
    """Build final PDF from multiple sources with all transformations.

    This is the "commit" operation that assembles the final document.
    """
    # First, merge all sources
    result_bytes = merge_pdfs(sources)

    # Flatten annotations if requested
    if flatten and annotations:
        result_bytes = flatten_annotations(result_bytes, annotations)

    # Add bookmarks if provided
    if bookmarks:
        result_bytes = add_bookmarks(result_bytes, bookmarks)

    # Compress if requested
    if compress:
        result_bytes = compress_pdf(result_bytes)

    return result_bytes


# ============================================================================
# Image Operations
# ============================================================================


def extract_images(
    pdf_bytes: bytes,
    pages: list[int] | None = None,
    min_width: int = 50,
    min_height: int = 50,
    fmt: str = "png",
) -> list[dict]:
    """Extract images from PDF pages.

    Returns list of image data with base64 encoding and metadata.

    Args:
        pdf_bytes: Source PDF
        pages: Page indices to extract from (all if None)
        min_width: Minimum image width to include
        min_height: Minimum image height to include
        fmt: Output format (png, jpeg)
    """
    doc = _open_pdf(pdf_bytes)
    target_pages = pages if pages is not None else list(range(len(doc)))
    if pages:
        _validate_pages(doc, pages)

    images = []
    for page_idx in target_pages:
        page = doc[page_idx]
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue

            if not base_image:
                continue

            width = base_image.get("width", 0)
            height = base_image.get("height", 0)

            if width < min_width or height < min_height:
                continue

            img_bytes = base_image["image"]
            img_ext = base_image.get("ext", "png")

            # Convert to requested format if needed
            if fmt == "jpeg" and img_ext != "jpeg":
                # Use PyMuPDF pixmap for conversion
                pix = fitz.Pixmap(img_bytes)
                if pix.alpha:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img_bytes = pix.tobytes("jpeg")
                mime = "image/jpeg"
            elif fmt == "png" and img_ext != "png":
                pix = fitz.Pixmap(img_bytes)
                img_bytes = pix.tobytes("png")
                mime = "image/png"
            else:
                mime = f"image/{img_ext}"

            data_url = f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"

            images.append({
                "page_index": page_idx,
                "image_index": img_idx,
                "width": width,
                "height": height,
                "colorspace": base_image.get("colorspace", 0),
                "bpc": base_image.get("bpc", 8),
                "size_bytes": len(base_image["image"]),
                "data_url": data_url,
            })

    return images


def convert_page_to_image(
    pdf_bytes: bytes,
    page: int,
    dpi: int = 150,
    fmt: str = "png",
    quality: int = 85,
) -> dict:
    """Render a single page as a high-resolution image.

    Args:
        pdf_bytes: Source PDF
        page: 0-indexed page number
        dpi: Resolution (72-600)
        fmt: Output format (png, jpeg, webp)
        quality: Image quality 1-100 (for jpeg/webp)
    """
    doc = _open_pdf(pdf_bytes)
    _validate_pages(doc, [page])

    pg = doc[page]
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = pg.get_pixmap(matrix=mat, alpha=False)

    img_bytes, mime = _pixmap_to_format(pix, fmt, quality)

    data_url = f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"

    return {
        "page_index": page,
        "width": pix.width,
        "height": pix.height,
        "dpi": dpi,
        "data_url": data_url,
    }


# ============================================================================
# Metadata Operations
# ============================================================================


def set_metadata(pdf_bytes: bytes, metadata: dict) -> bytes:
    """Set or update PDF metadata fields.

    Args:
        pdf_bytes: Source PDF
        metadata: Dict with optional keys: title, author, subject, keywords, creator, producer
    """
    doc = _open_pdf(pdf_bytes)

    current = doc.metadata or {}
    # Merge with existing, only override provided keys
    for key in ("title", "author", "subject", "keywords", "creator", "producer"):
        if key in metadata:
            current[key] = metadata[key]

    doc.set_metadata(current)
    return doc.tobytes(garbage=4, deflate=True)


def get_metadata(pdf_bytes: bytes) -> dict:
    """Get all PDF metadata fields."""
    doc = _open_pdf(pdf_bytes)
    return doc.metadata or {}


# ============================================================================
# Page Crop
# ============================================================================


def crop_pages(
    pdf_bytes: bytes,
    pages: list[int] | None = None,
    margins: dict | None = None,
    crop_box: dict | None = None,
) -> bytes:
    """Crop pages by setting margins or an explicit crop box.

    Args:
        pdf_bytes: Source PDF
        pages: Pages to crop (all if None)
        margins: Inset from each edge in points: {top, right, bottom, left}
        crop_box: Explicit crop box: {x0, y0, x1, y1} in PDF points
    """
    doc = _open_pdf(pdf_bytes)
    target_pages = pages if pages is not None else list(range(len(doc)))
    if pages:
        _validate_pages(doc, pages)

    for page_idx in target_pages:
        page = doc[page_idx]
        rect = page.rect

        if crop_box:
            new_rect = fitz.Rect(
                crop_box["x0"], crop_box["y0"],
                crop_box["x1"], crop_box["y1"],
            )
        elif margins:
            new_rect = fitz.Rect(
                rect.x0 + margins.get("left", 0),
                rect.y0 + margins.get("top", 0),
                rect.x1 - margins.get("right", 0),
                rect.y1 - margins.get("bottom", 0),
            )
        else:
            continue

        page.set_cropbox(new_rect)

    return doc.tobytes(garbage=4, deflate=True)


# ============================================================================
# Watermark
# ============================================================================


def add_text_watermark(
    pdf_bytes: bytes,
    text: str,
    pages: list[int] | None = None,
    font_size: float = 60.0,
    color: str = "#CCCCCC",
    opacity: float = 0.3,
    rotation: float = -45.0,
    user_name: str | None = None,
    date_str: str | None = None,
) -> bytes:
    """Add a text watermark to pages.

    Renders the watermark text diagonally across the center of each page.
    Supports dynamic variables: {user_name} and {date} are replaced in the text.

    Args:
        pdf_bytes: Source PDF
        text: Watermark text with optional placeholders {user_name} and {date}
              Examples: "DRAFT", "REVIEWED by {user_name} on {date}"
        pages: Pages to watermark (all if None)
        font_size: Font size in points
        color: Hex color string
        opacity: Opacity 0-1
        rotation: Rotation angle in degrees (negative = counterclockwise)
        user_name: Value to substitute for {user_name} placeholder
        date_str: Value to substitute for {date} placeholder (defaults to today)
    """
    # Substitute dynamic variables
    resolved_date = date_str or datetime.date.today().strftime("%b %d, %Y")
    text = text.replace("{user_name}", user_name or "").replace("{date}", resolved_date)
    doc = _open_pdf(pdf_bytes)
    target_pages = pages if pages is not None else list(range(len(doc)))
    if pages:
        _validate_pages(doc, pages)

    rgb = _hex_to_rgb(color)

    for page_idx in target_pages:
        page = doc[page_idx]
        rect = page.rect

        # Center of page
        cx = rect.width / 2
        cy = rect.height / 2

        # Create a text writer for the watermark
        tw = fitz.TextWriter(page.rect)
        font = fitz.Font("helv")

        # Measure text width to center it
        text_width = font.text_length(text, fontsize=font_size)

        # Insert text centered
        start_x = cx - text_width / 2
        start_y = cy + font_size / 3  # Approximate vertical centering

        tw.append(fitz.Point(start_x, start_y), text, font=font, fontsize=font_size)

        # Apply with rotation and opacity
        tw.write_text(page, color=rgb, opacity=opacity, morph=(fitz.Point(cx, cy), fitz.Matrix(rotation)))

    return doc.tobytes(garbage=4, deflate=True)


# ============================================================================
# Annotations (Read)
# ============================================================================


def read_annotations(pdf_bytes: bytes, pages: list[int] | None = None) -> list[dict]:
    """Read existing PDF annotations from pages.

    Returns list of annotation data including type, position, and content.
    """
    doc = _open_pdf(pdf_bytes)
    target_pages = pages if pages is not None else list(range(len(doc)))
    if pages:
        _validate_pages(doc, pages)

    results = []
    for page_idx in target_pages:
        page = doc[page_idx]
        annot = page.first_annot

        while annot:
            annot_data = {
                "page_index": page_idx,
                "type": annot.type[1],  # String name like "Highlight", "Text", etc.
                "type_code": annot.type[0],  # Numeric type code
                "rect": {
                    "x0": round(annot.rect.x0, 2),
                    "y0": round(annot.rect.y0, 2),
                    "x1": round(annot.rect.x1, 2),
                    "y1": round(annot.rect.y1, 2),
                },
            }

            # Add content/info if available
            info = annot.info
            if info.get("content"):
                annot_data["content"] = info["content"]
            if info.get("title"):
                annot_data["author"] = info["title"]
            if info.get("subject"):
                annot_data["subject"] = info["subject"]

            # Colors
            colors = annot.colors
            if colors.get("stroke"):
                annot_data["stroke_color"] = list(colors["stroke"])
            if colors.get("fill"):
                annot_data["fill_color"] = list(colors["fill"])

            results.append(annot_data)
            annot = annot.next

    return results


# ============================================================================
# Security: Encrypt / Decrypt / Sanitize
# ============================================================================


def encrypt_pdf(
    pdf_bytes: bytes,
    user_password: str = "",
    owner_password: str = "",
    permissions: int = -1,
) -> bytes:
    """Encrypt a PDF with password protection.

    Args:
        pdf_bytes: Source PDF
        user_password: Password to open the PDF (empty = no password to view)
        owner_password: Password for full permissions (required)
        permissions: Permission flags bitmask (PyMuPDF constants).
            Default -1 = all permissions. Common flags:
            fitz.PDF_PERM_PRINT = allow printing
            fitz.PDF_PERM_MODIFY = allow modification
            fitz.PDF_PERM_COPY = allow copying
            fitz.PDF_PERM_ANNOTATE = allow annotations
    """
    doc = _open_pdf(pdf_bytes)

    if not owner_password:
        raise PdfCorruptError("Owner password is required for encryption")

    return doc.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw=user_password,
        owner_pw=owner_password,
        permissions=permissions,
        garbage=4,
        deflate=True,
    )


def decrypt_pdf(pdf_bytes: bytes, password: str) -> bytes:
    """Decrypt a password-protected PDF.

    Args:
        pdf_bytes: Encrypted PDF
        password: Password to unlock (tries as both user and owner password)
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise PdfCorruptError(f"Cannot open PDF: {e}")

    if doc.is_encrypted:
        if not doc.authenticate(password):
            raise PdfCorruptError("Incorrect password")

    # Save without encryption
    return doc.tobytes(garbage=4, deflate=True)


def sanitize_document(
    pdf_bytes: bytes,
    remove_metadata: bool = True,
    remove_javascript: bool = True,
    remove_links: bool = False,
    remove_annotations: bool = False,
) -> bytes:
    """Sanitize a PDF by removing potentially sensitive or dangerous elements.

    Args:
        pdf_bytes: Source PDF
        remove_metadata: Clear all document metadata
        remove_javascript: Remove embedded JavaScript
        remove_links: Remove all hyperlinks
        remove_annotations: Remove all annotations
    """
    doc = _open_pdf(pdf_bytes)

    if remove_metadata:
        doc.set_metadata({
            "title": "",
            "author": "",
            "subject": "",
            "keywords": "",
            "creator": "",
            "producer": "",
        })

    if remove_javascript:
        # Remove JavaScript from the document catalog
        doc.scrub(
            attached_files=False,
            clean_pages=False,
            embedded_files=False,
            hidden_text=False,
            javascript=True,
            metadata=False,
            redactions=False,
            remove_links=False,
            reset_fields=False,
            reset_responses=False,
            thumbnails=False,
            xml_metadata=False,
        )

    for i in range(len(doc)):
        page = doc[i]

        if remove_links:
            # Remove all link annotations
            links = page.get_links()
            for link in reversed(links):
                page.delete_link(link)

        if remove_annotations:
            # Remove all annotations
            annot = page.first_annot
            while annot:
                next_annot = annot.next
                page.delete_annot(annot)
                annot = next_annot

    return doc.tobytes(garbage=4, deflate=True, clean=True)


# ============================================================================
# Table Extraction
# ============================================================================


# PPStructureV3 singleton – separate from PaddleOCR text recognition.
_pp_structure_instances: dict[str, object] = {}


def _get_pp_structure(lang: str = "en"):
    """Return a cached PPStructureV3 instance for the given language.

    PPStructureV3 uses neural networks to detect table structure from images,
    which works far better than rule-based approaches on scanned documents
    (faint lines, skewed rows, merged cells).
    """
    if lang not in _pp_structure_instances:
        try:
            from paddleocr import PPStructureV3
        except ImportError:
            raise PdfCorruptError(
                "Table OCR requires paddleocr>=3.0 with PPStructureV3 support. "
                "pip install paddlepaddle paddleocr"
            )
        _pp_structure_instances[lang] = PPStructureV3(
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_seal_recognition=False,
            use_formula_recognition=False,
            use_chart_recognition=False,
            use_table_recognition=True,
            lang=lang,
        )
    return _pp_structure_instances[lang]


def _parse_html_table(html: str) -> list[list[str]]:
    """Parse PP-Structure's HTML table output into rows of cell strings.

    Handles colspan by inserting empty strings for spanned columns.
    """
    from html.parser import HTMLParser

    class _Parser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows: list[list[str]] = []
            self._row: list[str] = []
            self._cell = ""
            self._in_cell = False
            self._colspan = 1

        def handle_starttag(self, tag, attrs):
            if tag in ("td", "th"):
                self._in_cell = True
                self._cell = ""
                self._colspan = 1
                for name, val in attrs:
                    if name == "colspan":
                        try:
                            self._colspan = int(val)
                        except (ValueError, TypeError):
                            pass

        def handle_endtag(self, tag):
            if tag in ("td", "th"):
                self._in_cell = False
                self._row.append(self._cell.strip())
                for _ in range(self._colspan - 1):
                    self._row.append("")
            elif tag == "tr":
                if self._row:
                    self.rows.append(self._row)
                    self._row = []

        def handle_data(self, data):
            if self._in_cell:
                self._cell += data

    parser = _Parser()
    parser.feed(html)
    # Flush any remaining row
    if parser._row:
        parser.rows.append(parser._row)
    return parser.rows


def _extract_tables_pymupdf(doc, page_idx: int) -> list[dict]:
    """Extract tables from a page using PyMuPDF's built-in detection."""
    page = doc[page_idx]
    try:
        tabs = page.find_tables()
    except AttributeError:
        log.warning("table_detection_unavailable", page=page_idx)
        return []

    results = []
    for tab_idx, table in enumerate(tabs):
        rows = table.extract()
        bbox = table.bbox
        results.append({
            "page_index": page_idx,
            "table_index": tab_idx,
            "bbox": {
                "x0": round(bbox[0], 2),
                "y0": round(bbox[1], 2),
                "x1": round(bbox[2], 2),
                "y1": round(bbox[3], 2),
            },
            "rows": rows,
            "row_count": len(rows),
            "col_count": len(rows[0]) if rows else 0,
        })
    return results


def _extract_tables_ppstructure(
    doc, page_idx: int, language: str = "en", dpi: int = 300,
) -> list[dict]:
    """Extract tables from a scanned page using PPStructureV3 neural network."""
    engine = _get_pp_structure(language)
    page = doc[page_idx]

    # Render page as image for PPStructureV3
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3
    )

    try:
        output = engine.predict(input=img_array)
    except Exception as e:
        log.warning("ppstructure_failed", page=page_idx, error=str(e))
        return []

    scale = 72.0 / dpi
    results = []
    tab_idx = 0

    for res in output:
        raw = res.json if hasattr(res, "json") else res
        # PaddleOCR 3.4.0 nests results under a "res" key
        data = raw.get("res", raw) if isinstance(raw, dict) else raw

        # PPStructureV3 returns layout regions + table results.
        # Extract table HTML from layout_det_res boxes + table_res_list.
        table_items = _find_tables_in_ppstructurev3(data)

        for html, bbox in table_items:
            rows = _parse_html_table(html)
            if not rows:
                continue

            results.append({
                "page_index": page_idx,
                "table_index": tab_idx,
                "bbox": {
                    "x0": round(bbox[0] * scale, 2),
                    "y0": round(bbox[1] * scale, 2),
                    "x1": round(bbox[2] * scale, 2),
                    "y1": round(bbox[3] * scale, 2),
                },
                "rows": rows,
                "row_count": len(rows),
                "col_count": len(rows[0]) if rows else 0,
            })
            tab_idx += 1

    return results


def _find_tables_in_ppstructurev3(data: dict) -> list[tuple[str, list]]:
    """Extract (html, bbox) pairs from a PPStructureV3 result dict.

    PPStructureV3 output structure varies across versions. This function
    tries multiple known access patterns and logs the actual keys when
    no tables are found, aiding debugging.
    """
    tables: list[tuple[str, list]] = []

    # Pattern 1: table_res_list with pred_html (most common in 3.x)
    table_res_list = data.get("table_res_list", [])
    layout_boxes = data.get("layout_det_res", {}).get("boxes", [])
    table_boxes = [b for b in layout_boxes if b.get("label") == "table"]

    for i, table_res in enumerate(table_res_list):
        html = (
            table_res.get("pred_html", "")
            or table_res.get("html", "")
            or table_res.get("structure_str", "")
        )
        if not html:
            continue

        # Bbox from matching layout detection box, or from table result
        if i < len(table_boxes):
            bbox = table_boxes[i].get("coordinate", [0, 0, 0, 0])
        else:
            bbox = table_res.get("bbox", table_res.get("coordinate", [0, 0, 0, 0]))

        tables.append((html, bbox))

    # Pattern 2: Fallback — older PPStructureV3 with parsing_res_list
    if not tables:
        for item in data.get("parsing_res_list", []):
            label = item.get("block_label", item.get("type", ""))
            if label != "table":
                continue
            html = item.get("block_content", item.get("res", {}).get("html", ""))
            bbox = item.get("block_bbox", item.get("bbox", [0, 0, 0, 0]))
            if html:
                tables.append((html, bbox))

    if not tables:
        log.debug(
            "ppstructurev3_no_tables_found",
            result_keys=list(data.keys()) if isinstance(data, dict) else str(type(data)),
        )

    return tables


def extract_tables(
    pdf_bytes: bytes,
    pages: list[int] | None = None,
    strategy: str = "auto",
    language: str = "en",
    dpi: int = 300,
) -> list[dict]:
    """Extract tabular data from PDF pages.

    Strategies:
      - ``"auto"`` (default): Use PyMuPDF for digital pages.  For scanned
        pages (< 50 chars of extractable text) or when PyMuPDF finds no
        tables, fall back to PP-Structure.
      - ``"pymupdf"``: Always use PyMuPDF (fast, good for digital PDFs).
      - ``"ppstructure"``: Always use PPStructureV3 neural network (best for
        scanned 1099s, W-2s, K-1s with faint/crooked lines).

    Args:
        pdf_bytes: Source PDF
        pages: Pages to extract from (all if None)
        strategy: "auto", "pymupdf", or "ppstructure"
        language: PaddleOCR language code for PP-Structure
        dpi: Rendering DPI for PP-Structure image input
    """
    doc = _open_pdf(pdf_bytes)
    target_pages = pages if pages is not None else list(range(len(doc)))
    if pages:
        _validate_pages(doc, pages)

    results = []
    for page_idx in target_pages:
        page = doc[page_idx]

        if strategy == "pymupdf":
            results.extend(_extract_tables_pymupdf(doc, page_idx))
            continue

        if strategy == "ppstructure":
            results.extend(
                _extract_tables_ppstructure(doc, page_idx, language, dpi)
            )
            continue

        # strategy == "auto": decide per page
        has_text = len(page.get_text().strip()) > 50

        if has_text:
            # Digital page — try PyMuPDF first (fast)
            page_tables = _extract_tables_pymupdf(doc, page_idx)
            if page_tables:
                results.extend(page_tables)
                continue
            # PyMuPDF found nothing → try PP-Structure as fallback
            log.info("pymupdf_no_tables_fallback_ppstructure", page=page_idx)

        # Scanned page or PyMuPDF found nothing → PP-Structure
        results.extend(
            _extract_tables_ppstructure(doc, page_idx, language, dpi)
        )

    return results


# ============================================================================
# Page Labels
# ============================================================================


def set_page_labels(
    pdf_bytes: bytes,
    labels: list[dict],
) -> bytes:
    """Set custom page labels (e.g., roman numerals for front matter).

    Args:
        pdf_bytes: Source PDF
        labels: List of label rules, each with:
            - start_page: 0-indexed page where this rule begins
            - prefix: Text prefix (e.g., "A-")
            - style: "D" (decimal), "r" (roman lower), "R" (roman upper),
                     "a" (alpha lower), "A" (alpha upper)
            - first_page_num: Starting number (default 1)
    """
    doc = _open_pdf(pdf_bytes)

    # Build page labels array for PyMuPDF
    # Format: list of dicts with startpage, prefix, style, firstpagenum
    label_rules = []
    for label in labels:
        rule = {
            "startpage": label["start_page"],
            "prefix": label.get("prefix", ""),
            "style": label.get("style", "D"),
            "firstpagenum": label.get("first_page_num", 1),
        }
        label_rules.append(rule)

    doc.set_page_labels(label_rules)
    return doc.tobytes(garbage=4, deflate=True)


# ============================================================================
# Repair
# ============================================================================


def repair_pdf(pdf_bytes: bytes) -> bytes:
    """Repair a corrupted PDF by re-saving with garbage collection.

    PyMuPDF can often recover from XREF table errors, broken object streams,
    and other structural damage by simply opening and re-saving the document.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        # Try opening with repair flag
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise PdfCorruptError(f"PDF is too damaged to repair: {e}")

    return doc.tobytes(garbage=4, deflate=True, clean=True)


# ============================================================================
# Image to PDF
# ============================================================================


def images_to_pdf(image_list: list[bytes], filenames: list[str] | None = None) -> bytes:
    """Convert a list of images (JPG, PNG, TIFF, HEIC) into a single PDF.

    Each image becomes one page sized to fit the image at 72 DPI.

    Args:
        image_list: List of raw image bytes
        filenames: Optional filenames (used to detect HEIC format)
    """
    doc = fitz.open()  # New empty PDF

    for idx, img_bytes in enumerate(image_list):
        fname = (filenames[idx] if filenames and idx < len(filenames) else "").lower()

        # Handle HEIC (iPhone) format via pillow-heif
        if fname.endswith((".heic", ".heif")):
            try:
                import pillow_heif
                heif_file = pillow_heif.read_heif(img_bytes)
                pil_img = heif_file.to_pillow()
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                img_bytes = buf.getvalue()
            except ImportError:
                raise PdfCorruptError("HEIC support requires pillow-heif package")
            except Exception as e:
                raise PdfCorruptError(f"Failed to convert HEIC image: {e}")

        try:
            # Open the image to get dimensions
            pil_img = Image.open(io.BytesIO(img_bytes))
            w, h = pil_img.size

            # Create a page matching the image dimensions (at 72 DPI)
            page = doc.new_page(width=w, height=h)
            rect = fitz.Rect(0, 0, w, h)

            # Insert the image
            page.insert_image(rect, stream=img_bytes)
        except Exception as e:
            raise PdfCorruptError(f"Failed to process image {idx}: {e}")

    if len(doc) == 0:
        raise PdfCorruptError("No valid images provided")

    return doc.tobytes(garbage=4, deflate=True)


# ============================================================================
# Office to PDF (LibreOffice)
# ============================================================================


_SPREADSHEET_EXTENSIONS = {".xlsx", ".xls", ".csv", ".tsv", ".ods"}


def _prepare_spreadsheet_for_pdf(input_path: str, ext: str, tmpdir: str) -> str:
    """Pre-process a spreadsheet to set fit-to-width page scaling.

    For .xlsx: Modifies the workbook's page setup so all columns fit on one
    page width (rows can span multiple pages). Sets landscape orientation.

    For .csv/.tsv: Creates a proper .xlsx workbook with fit-to-width settings,
    which LibreOffice then converts more reliably than raw CSV.

    For .xls/.ods: Returns as-is (LibreOffice uses calc_pdf_Export filter).

    Returns:
        Path to the processed file (may differ from input if format changed).
    """
    import csv
    import os

    try:
        import openpyxl
        from openpyxl.worksheet.properties import PageSetupProperties
    except ImportError:
        return input_path

    if ext == ".xlsx":
        try:
            wb = openpyxl.load_workbook(input_path)
            for ws in wb.worksheets:
                ws.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
                ws.page_setup.fitToWidth = 1
                ws.page_setup.fitToHeight = 0
                ws.page_setup.orientation = "landscape"
            wb.save(input_path)
        except Exception:
            pass  # If openpyxl can't handle it, let LibreOffice try as-is
        return input_path

    if ext in (".csv", ".tsv"):
        delimiter = "\t" if ext == ".tsv" else ","
        try:
            with open(input_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f, delimiter=delimiter)
                rows = list(reader)

            wb = openpyxl.Workbook()
            ws = wb.active
            for row in rows:
                ws.append(row)

            ws.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
            ws.page_setup.fitToWidth = 1
            ws.page_setup.fitToHeight = 0
            ws.page_setup.orientation = "landscape"

            xlsx_name = os.path.splitext(os.path.basename(input_path))[0] + ".xlsx"
            xlsx_path = os.path.join(tmpdir, xlsx_name)
            wb.save(xlsx_path)
            return xlsx_path
        except Exception:
            return input_path  # Fall back to raw CSV conversion

    # .xls and .ods: return as-is, calc_pdf_Export handles them
    return input_path


def office_to_pdf(file_bytes: bytes, filename: str) -> bytes:
    """Convert an office document (DOCX, XLSX, PPTX, etc.) to PDF using LibreOffice.

    LibreOffice runs in headless mode as a subprocess. The input file is
    written to a temp directory, converted, and the resulting PDF is read back.

    For spreadsheet formats (.xlsx, .xls, .csv, .tsv, .ods), the file is
    pre-processed to set fit-to-width page scaling so that all columns fit
    on the page width (landscape). This prevents wide reports from being
    split across multiple page widths.

    Args:
        file_bytes: Raw bytes of the office document
        filename: Original filename with extension (used to determine format)

    Returns:
        PDF bytes

    Raises:
        PdfCorruptError: If conversion fails
    """
    import os
    import subprocess
    import tempfile

    ext = os.path.splitext(filename)[1].lower()
    allowed_extensions = {
        ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
        ".odt", ".ods", ".odp", ".rtf", ".csv", ".tsv", ".txt",
    }
    if ext not in allowed_extensions:
        raise PdfCorruptError(
            f"Unsupported file format '{ext}'. "
            f"Supported: {', '.join(sorted(allowed_extensions))}"
        )

    is_spreadsheet = ext in _SPREADSHEET_EXTENSIONS

    with tempfile.TemporaryDirectory(prefix="lo_convert_") as tmpdir:
        # Write the input file
        input_path = os.path.join(tmpdir, filename)
        with open(input_path, "wb") as f:
            f.write(file_bytes)

        # For spreadsheets: set fit-to-width page scaling before conversion
        if is_spreadsheet:
            input_path = _prepare_spreadsheet_for_pdf(input_path, ext, tmpdir)

        # Use calc_pdf_Export filter for spreadsheets (better page layout)
        convert_to = "pdf:calc_pdf_Export" if is_spreadsheet else "pdf"

        # Run LibreOffice headless conversion
        try:
            result = subprocess.run(
                [
                    "libreoffice",
                    "--headless",
                    "--norestore",
                    "--convert-to", convert_to,
                    "--outdir", tmpdir,
                    input_path,
                ],
                capture_output=True,
                timeout=120,
                env={
                    **os.environ,
                    # Isolated user profile to allow parallel conversions
                    "HOME": tmpdir,
                },
            )
        except subprocess.TimeoutExpired:
            raise PdfCorruptError("LibreOffice conversion timed out (120s limit)")
        except FileNotFoundError:
            raise PdfCorruptError(
                "LibreOffice is not installed. Install libreoffice-core, "
                "libreoffice-writer, libreoffice-calc."
            )

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[:500]
            raise PdfCorruptError(f"LibreOffice conversion failed: {stderr}")

        # Find the output PDF (derive name from the file we actually converted)
        converted_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(tmpdir, f"{converted_name}.pdf")

        if not os.path.exists(output_path):
            # Sometimes LibreOffice names the output differently
            pdf_files = [f for f in os.listdir(tmpdir) if f.endswith(".pdf")]
            if not pdf_files:
                raise PdfCorruptError("LibreOffice produced no PDF output")
            output_path = os.path.join(tmpdir, pdf_files[0])

        with open(output_path, "rb") as f:
            return f.read()


# ============================================================================
# Blank Page Detection
# ============================================================================


def detect_blank_pages(
    pdf_bytes: bytes,
    ink_threshold: float = 0.01,
) -> list[dict]:
    """Detect blank or nearly-blank pages in a PDF.

    Renders each page as a small grayscale image and calculates the
    ratio of "ink" pixels (non-white) to total pixels. Pages below
    the threshold are considered blank.

    Args:
        pdf_bytes: Source PDF
        ink_threshold: Max ink ratio to be considered blank (0.01 = 1%)

    Returns:
        List of dicts: [{page_index, is_blank, ink_ratio}]
    """
    doc = _open_pdf(pdf_bytes)
    results = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        # Render small grayscale image for speed
        mat = fitz.Matrix(0.5, 0.5)  # Low resolution is fine
        pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csGRAY)

        # Convert to numpy array
        img_bytes = pix.samples
        arr = np.frombuffer(img_bytes, dtype=np.uint8)

        # Count non-white pixels (threshold: pixel value < 240)
        total_pixels = len(arr)
        ink_pixels = int(np.sum(arr < 240))
        ink_ratio = ink_pixels / total_pixels if total_pixels > 0 else 0

        results.append({
            "page_index": page_idx,
            "is_blank": ink_ratio < ink_threshold,
            "ink_ratio": round(ink_ratio, 6),
        })

    return results


# ============================================================================
# Simple Document Classification
# ============================================================================


# Tax form keyword patterns: (keyword_on_first_page, suggested_label)
_TAX_FORM_PATTERNS = [
    ("Wage and Tax Statement", "W-2"),
    ("Form W-2", "W-2"),
    ("Interest Income", "1099-INT"),
    ("Form 1099-INT", "1099-INT"),
    ("Dividend Income", "1099-DIV"),
    ("Form 1099-DIV", "1099-DIV"),
    ("Miscellaneous Income", "1099-MISC"),
    ("Form 1099-MISC", "1099-MISC"),
    ("Nonemployee Compensation", "1099-NEC"),
    ("Form 1099-NEC", "1099-NEC"),
    ("Form 1099-B", "1099-B"),
    ("Proceeds From Broker", "1099-B"),
    ("Form 1099-R", "1099-R"),
    ("Distributions From Pensions", "1099-R"),
    ("Form 1099-S", "1099-S"),
    ("Real Estate Transactions", "1099-S"),
    ("Form 1099-G", "1099-G"),
    ("Government Payments", "1099-G"),
    ("Form 1099-K", "1099-K"),
    ("Payment Card", "1099-K"),
    ("Form 1099-SSA", "SSA-1099"),
    ("Social Security Benefit", "SSA-1099"),
    ("Schedule K-1", "K-1"),
    ("Partner's Share", "K-1"),
    ("Shareholder's Share", "K-1"),
    ("Beneficiary's Share", "K-1"),
    ("Form 1098", "1098"),
    ("Mortgage Interest Statement", "1098"),
    ("Form 1098-T", "1098-T"),
    ("Tuition Statement", "1098-T"),
    ("Form 1095-A", "1095-A"),
    ("Health Insurance Marketplace", "1095-A"),
    ("Form 1095-B", "1095-B"),
    ("Form 1095-C", "1095-C"),
    ("U.S. Individual Income Tax Return", "1040"),
    ("Form 1040", "1040"),
    ("Form 1065", "1065"),
    ("U.S. Return of Partnership Income", "1065"),
    ("Form 1120", "1120"),
    ("U.S. Corporation Income Tax Return", "1120"),
    ("Form 1120-S", "1120-S"),
    ("Form 990", "990"),
    ("Form 941", "941"),
    ("Employer's Quarterly Federal Tax", "941"),
    ("Form 940", "940"),
    ("Form 8949", "8949"),
    ("Sales and Other Dispositions", "8949"),
    ("Schedule A", "Schedule A"),
    ("Itemized Deductions", "Schedule A"),
    ("Schedule B", "Schedule B"),
    ("Schedule C", "Schedule C"),
    ("Profit or Loss From Business", "Schedule C"),
    ("Schedule D", "Schedule D"),
    ("Capital Gains and Losses", "Schedule D"),
    ("Schedule E", "Schedule E"),
    ("Supplemental Income and Loss", "Schedule E"),
    ("Schedule SE", "Schedule SE"),
    ("Self-Employment Tax", "Schedule SE"),
    ("Passport", "Passport"),
    ("Driver License", "ID"),
    ("Driver's License", "ID"),
    ("Property Tax", "Property Tax"),
    ("Receipt", "Receipt"),
    ("Invoice", "Invoice"),
    ("Bank Statement", "Bank Statement"),
]


def _extract_tax_year(text: str) -> str | None:
    """Extract the tax year from document text.

    Searches for explicit year references near tax keywords first,
    then falls back to the most frequently occurring 4-digit year.

    Args:
        text: Raw text from the first 1-2 pages of the document.

    Returns:
        A 4-digit year string (e.g., "2024") or None if not found.
    """
    import re
    from collections import Counter

    # Focus on first ~3000 chars (covers header + form fields)
    snippet = text[:3000]

    # Priority 1: explicit tax-year references
    explicit_patterns = [
        r"(?:Tax\s+Year|Taxable\s+Year|Calendar\s+Year)\s*[:.\-]?\s*(20[1-3]\d)",
        r"(20[1-3]\d)\s+(?:Tax\s+Year|Taxable\s+Year|Calendar\s+Year)",
        r"(?:for\s+(?:the\s+)?(?:tax\s+|calendar\s+)?year)\s+.*?(20[1-3]\d)",
        r"(?:for\s+(?:the\s+)?(?:year|period)\s+(?:ending|ended|beginning))\s+.*?(20[1-3]\d)",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, snippet, re.IGNORECASE)
        if match:
            return match.group(1)

    # Priority 2: year in parentheses after a form reference  e.g. "Form W-2 (2024)"
    form_year = re.search(
        r"(?:Form\s+[\w-]+)\s*\(?(20[1-3]\d)\)?", snippet, re.IGNORECASE,
    )
    if form_year:
        return form_year.group(1)

    # Priority 3: date patterns – "December 31, 2024" or "01/31/2024"
    date_patterns = [
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(20[1-3]\d)",
        r"\d{1,2}/\d{1,2}/(20[1-3]\d)",
    ]
    for pattern in date_patterns:
        match = re.search(pattern, snippet, re.IGNORECASE)
        if match:
            return match.group(1)

    # Fallback: most common 4-digit year in the text
    all_years = re.findall(r"\b(20[1-3]\d)\b", snippet)
    if all_years:
        year_counts = Counter(all_years)
        return year_counts.most_common(1)[0][0]

    return None


def classify_document(pdf_bytes: bytes) -> dict:
    """Simple keyword-based document classification.

    Reads text from the first page and searches for known tax form keywords.
    Also attempts to extract the tax year from the document.

    Returns:
        {suggested_label, confidence, matched_keyword, tax_year, page_text_preview}
    """
    doc = _open_pdf(pdf_bytes)
    if len(doc) == 0:
        return {
            "suggested_label": "Unknown",
            "confidence": 0.0,
            "matched_keyword": None,
            "tax_year": None,
            "page_text_preview": "",
        }

    # Extract text from first page (and optionally second for multi-page forms)
    first_page_text = doc[0].get_text().strip()
    combined_text = first_page_text
    if len(doc) > 1:
        combined_text += " " + doc[1].get_text().strip()

    text_upper = combined_text.upper()
    tax_year = _extract_tax_year(combined_text)

    for keyword, label in _TAX_FORM_PATTERNS:
        if keyword.upper() in text_upper:
            return {
                "suggested_label": label,
                "confidence": 0.9,
                "matched_keyword": keyword,
                "tax_year": tax_year,
                "page_text_preview": first_page_text[:200],
            }

    return {
        "suggested_label": "Unknown",
        "confidence": 0.0,
        "matched_keyword": None,
        "tax_year": tax_year,
        "page_text_preview": first_page_text[:200],
    }


# ============================================================================
# OCR (Optical Character Recognition) – PaddleOCR engine
# ============================================================================


# Module-level singleton – PaddleOCR model loading is expensive (~2 s),
# so we initialise once per process/language.
_paddle_ocr_instances: dict[str, object] = {}


def _get_paddle_ocr(lang: str = "en"):
    """Return a cached PaddleOCR instance for the given language.

    Uses **PP-OCRv5** models for maximum accuracy.  PP-OCRv5 delivers
    significant accuracy improvements over v4, especially on dense tax
    forms and receipts with small text and complex layouts.
    """
    if lang not in _paddle_ocr_instances:
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise PdfCorruptError(
                "OCR requires paddleocr>=3.0 and paddlepaddle>=3.0 packages. "
                "pip install paddlepaddle paddleocr"
            )
        _paddle_ocr_instances[lang] = PaddleOCR(
            ocr_version="PP-OCRv5",
            lang=lang,
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )
    return _paddle_ocr_instances[lang]


def ocr_pages(
    pdf_bytes: bytes,
    pages: list[int] | None = None,
    language: str = "en",
    dpi: int = 300,
) -> dict:
    """Run PaddleOCR on scanned pages and return rich results.

    For pages that have no extractable text (scans/images), this:
      1. Renders the page as a high-res image.
      2. Runs PaddleOCR PP-OCRv5 (detection + orientation + recognition).
      3. Overlays invisible text at the correct positions so that
         Ctrl+F search works in PDF viewers.
      4. Returns per-word text, bounding rectangles, and confidence scores.

    Args:
        pdf_bytes: Source PDF
        pages: Page indices to OCR (all if None)
        language: PaddleOCR language code (e.g., "en", "fr", "de", "ch")
        dpi: Rendering resolution for OCR (higher = more accurate, slower)

    Returns:
        {
            "pages": [
                {
                    "page_index": 0,
                    "words": [
                        {"text": "...", "bbox": {"x": 50.0, "y": 50.0, "w": 150.0, "h": 25.0}, "confidence": 0.97}
                    ],
                    "full_text": "..."
                }
            ],
            "total_words": 150,
            "avg_confidence": 0.94,
            "pdf_bytes": b"..."   # PDF with invisible text layer
        }
    """
    paddle = _get_paddle_ocr(language)

    doc = _open_pdf(pdf_bytes)
    target_pages = pages if pages is not None else list(range(len(doc)))
    if pages:
        _validate_pages(doc, pages)

    all_page_results: list[dict] = []
    total_words = 0
    confidence_sum = 0.0

    for page_idx in target_pages:
        page = doc[page_idx]

        # Skip pages that already have good quality digital text.
        # Scanned PDFs often contain garbled embedded fonts that produce
        # nonsensical characters (e.g. "CqCY,Cr)COPaD-").  These pass
        # simple character-level checks because the chars are valid ASCII,
        # but the "words" are extremely short (avg ~1.8 chars vs ~4-5 for
        # real English text).  We use average token length as the heuristic.
        existing_text = page.get_text().strip()
        if len(existing_text) > 50:
            tokens = existing_text.split()
            avg_token_len = (
                sum(len(t) for t in tokens) / len(tokens) if tokens else 0
            )
            if avg_token_len >= 3.0:
                # Real text — skip OCR
                all_page_results.append({
                    "page_index": page_idx,
                    "words": [],
                    "full_text": existing_text,
                })
                continue
            else:
                log.info(
                    "ocr_garbage_text_detected",
                    page=page_idx,
                    text_len=len(existing_text),
                    avg_token_len=round(avg_token_len, 2),
                )

        # Render page at high DPI for OCR
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert to numpy array for PaddleOCR
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )

        # Run PaddleOCR PP-OCRv5 (detection + orientation + recognition)
        try:
            ocr_output = paddle.predict(input=img_array)
        except Exception as e:
            log.warning("ocr_failed", page=page_idx, error=str(e))
            all_page_results.append({
                "page_index": page_idx,
                "words": [],
                "full_text": "",
            })
            continue

        # PaddleOCR 3.x .predict() returns an iterable of result objects.
        # Each result has .json with: rec_texts, rec_scores, dt_polys, rec_boxes
        page_words: list[dict] = []
        page_texts: list[str] = []
        # Scale factor from rendered image pixels back to PDF points
        scale = 72.0 / dpi

        for res in ocr_output:
            raw = res.json if hasattr(res, "json") else res
            # PaddleOCR 3.4.0 nests results under a "res" key
            data = raw.get("res", raw) if isinstance(raw, dict) else raw
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            polys = data.get("dt_polys", [])

            for i, text in enumerate(texts):
                confidence = scores[i] if i < len(scores) else 0.0

                if not text.strip() or confidence < 0.3:
                    continue

                # dt_polys[i]: 4-point polygon in image pixels
                bbox_points = polys[i] if i < len(polys) else [[0, 0]] * 4
                # Convert 4-point polygon to axis-aligned {x, y, w, h} rectangle
                # in PDF points for frontend compatibility
                xs = [pt[0] * scale for pt in bbox_points]
                ys = [pt[1] * scale for pt in bbox_points]
                bx = min(xs)
                by = min(ys)
                bw = max(xs) - bx
                bh = max(ys) - by

                page_words.append({
                    "text": text.strip(),
                    "bbox": {
                        "x": round(bx, 2),
                        "y": round(by, 2),
                        "w": round(bw, 2),
                        "h": round(bh, 2),
                    },
                    "confidence": round(confidence, 4),
                })
                page_texts.append(text.strip())

                total_words += 1
                confidence_sum += confidence

                # Insert invisible text at the correct position in the PDF
                x = bbox_points[0][0] * scale
                y = bbox_points[0][1] * scale
                h = abs(bbox_points[3][1] - bbox_points[0][1]) * scale
                font_size = max(h * 0.8, 4)

                page.insert_text(
                    fitz.Point(x, y + h * 0.8),
                    text.strip(),
                    fontsize=font_size,
                    fontname="helv",
                    render_mode=3,  # Invisible text
                )

        all_page_results.append({
            "page_index": page_idx,
            "words": page_words,
            "full_text": " ".join(page_texts),
        })

    avg_confidence = round(confidence_sum / total_words, 4) if total_words > 0 else 0.0

    return {
        "pages": all_page_results,
        "total_words": total_words,
        "avg_confidence": avg_confidence,
        "pdf_bytes": doc.tobytes(garbage=4, deflate=True),
    }
