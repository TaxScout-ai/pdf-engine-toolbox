"""Download the pinned ONNX weights during the Docker build (TAX-6995).

RapidOCR (PP-OCRv6 Small detection/recognition, mobile line classifier) and
RapidTable (SLANet-plus) weights are fetched from their versioned upstream
locations and kept only when their SHA-256 matches
``app/services/rapid_ocr_models.py``. Any failure fails the build: the
service never downloads a model at runtime.

Paddle and its models were removed from the engine in TAX-6995.
"""
from pathlib import Path

# Docker copies the standalone manifest beside this build script.
from rapid_ocr_models import download_models

download_models(Path("/opt/rapidocr/models"))
