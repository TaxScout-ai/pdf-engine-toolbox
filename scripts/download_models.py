"""Pre-download PaddleOCR PP-OCRv5 models during Docker build.

This avoids ~300 MB download on first request at runtime.
Runs with || true semantics — failure here is non-fatal.

NOTE: Only downloads the core OCR models (_get_paddle_ocr), NOT PPStructureV3.
PPStructureV3 loads PP-Chart2Table (~1.5 GB transformer) which causes OOM
on t3a.medium (4 GB) during both build and runtime. PPStructure models
load lazily on first classify/table request if ever needed.
"""
import sys
from pathlib import Path

if "--rapid-only" in sys.argv:
    # Docker copies the standalone manifest beside this build script.
    from rapid_ocr_models import download_models
    download_models(Path("/opt/rapidocr/models"))
    raise SystemExit(0)

from paddleocr import PaddleOCR

PaddleOCR(
    ocr_version="PP-OCRv5",
    lang="en",
    device="cpu",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
)
