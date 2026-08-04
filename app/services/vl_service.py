"""PaddleOCR-VL 1.6 — the second opinion on what a scan actually says.

── What this is for ────────────────────────────────────────────────────────

Not a replacement for `/text/ocr`. VL emits no word-level geometry in either
mode (probed via `layout_det_res` and `use_ocr_for_image_block`), and three
shipped features are built on per-word boxes: the viewer highlight, the
searchable PDF, and the coordinates evidence is anchored to. PaddleOCR keeps
that job.

What VL does better is read. On the drivers-licence case it returns `3497`
where the standard recogniser returns `349` — a dropped digit in a street
number, the same class of defect that reaches CPAs as a correction.

So this is the arm you call when a read is *doubted*: a field that failed
corroboration against the text layer, a confidence below threshold, a figure a
validator flagged. Calling it on whole documents is the wrong shape — see cost.

── Cost, and why the caller must be selective ──────────────────────────────

25.1 s/page on CPU. That is per page, measured on this hardware, and it is why
every entry point here takes an explicit page list rather than defaulting to
the whole document.

(An older note says 933 s/doc. It is withdrawn: it timed a crash, because VL
was failing at import when it was measured. Do not size capacity from it.)

── Why the weights are converted ───────────────────────────────────────────

The published weights are bfloat16. A CPU without AVX512-BF16 aborts on load,
which is this machine and most cheap instances. The 620 tensors are converted
to fp32 once, offline, into ``PaddleOCR-VL-1.6-fp32``. This is not a memory
problem — that diagnosis was made once and was wrong; the box has 61 GB.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

#: Where the fp32-converted weights live. Overridable so an image can ship them
#: somewhere else without a code change.
VL_MODEL_DIR = os.environ.get(
    "VL_MODEL_DIR", "/models/PaddleOCR-VL-1.6-fp32"
)

#: Pages per request. VL is ~25 s/page, so a caller that asks for a 40-page
#: return is asking for seventeen minutes; refusing is kinder than accepting.
MAX_VL_PAGES = int(os.environ.get("VL_MAX_PAGES", "8"))

_pipeline = None


class VlUnavailable(RuntimeError):
    """VL was asked for and cannot be served.

    Raised rather than returning empty output. An empty OCR result reported as
    success is exactly how the paddlepaddle 3.3.x regression stayed invisible
    through 82 passing tests, and the same shape of silence in a *reading*
    service would be read as "the document does not say that".
    """


def _load_pipeline():
    """Import and construct lazily, exactly like the paddle imports elsewhere.

    The whole point of the lite image is that it can be built without this
    stack; a module-level import here would put it back.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    if not os.path.isdir(VL_MODEL_DIR):
        raise VlUnavailable(
            f"VL weights not found at {VL_MODEL_DIR}. They are the fp32 "
            "conversion, not the published bfloat16 release — a bf16 load "
            "aborts on a CPU without AVX512-BF16."
        )

    try:
        from paddleocr import PaddleOCRVL
    except ImportError as exc:  # pragma: no cover - depends on image
        raise VlUnavailable(
            "PaddleOCR-VL requires the full image; the lite build ships no "
            "recognition stack."
        ) from exc

    _pipeline = PaddleOCRVL(vl_rec_model_dir=VL_MODEL_DIR)
    logger.info("vl_service: pipeline ready (%s)", VL_MODEL_DIR)
    return _pipeline


def read_pages(pdf_bytes: bytes, pages: list[int]) -> dict[str, Any]:
    """Read the named pages with VL.

    :param pages: 1-based page numbers. Explicit and required — there is no
        "all pages" convenience, because at 25 s/page that convenience is how a
        caller accidentally spends a quarter of an hour.
    :returns: ``{"pages": [{"page": int, "blocks": [{"label", "text"}]}]}``

    Block-level only, and named so at the boundary: callers that need word
    boxes must use ``/text/ocr``. Returning a shape that merely *looks* like
    the OCR response would invite silently wiring VL into the viewer highlight,
    where it would produce nothing to highlight.
    """
    if not pages:
        raise ValueError("pages must be a non-empty list of 1-based page numbers")
    if len(pages) > MAX_VL_PAGES:
        raise ValueError(
            f"{len(pages)} pages requested, limit is {MAX_VL_PAGES}. "
            f"VL runs ~25 s/page; ask for the pages in doubt, not the document."
        )

    import fitz

    pipeline = _load_pipeline()
    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    out: list[dict[str, Any]] = []
    for page_number in pages:
        if page_number < 1 or page_number > document.page_count:
            raise ValueError(
                f"page {page_number} out of range (document has "
                f"{document.page_count})"
            )

        # One page at a time: a caller asking for page 7 of a 40-page return
        # should pay for one page, not for the pipeline walking to it.
        single = fitz.open()
        single.insert_pdf(document, from_page=page_number - 1, to_page=page_number - 1)
        page_bytes = single.tobytes()

        blocks: list[dict[str, Any]] = []
        for result in pipeline.predict(page_bytes):
            res = (result.json if hasattr(result, "json") else {}).get("res", {})
            for block in res.get("parsing_res_list") or []:
                text = (block.get("block_content") or "").strip()
                if text:
                    blocks.append(
                        {"label": block.get("block_label"), "text": text}
                    )
            break

        out.append({"page": page_number, "blocks": blocks})

    return {"pages": out, "model": "PaddleOCR-VL-1.6", "geometry": "block"}
