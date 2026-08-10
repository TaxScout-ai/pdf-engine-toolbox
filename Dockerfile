FROM python:3.12-slim

WORKDIR /app

# Install system dependencies:
# - curl: for health checks
# - libgomp1: OpenMP runtime required by PaddlePaddle
# - libheif: for HEIC image support (iPhone photos)
# - libgl1: required by opencv-python-headless
# - libreoffice-core + writer + calc + impress: for office-to-PDF conversion
# - fonts-liberation: standard fonts for LibreOffice rendering
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
        libheif-dev \
        libgl1 \
        libreoffice-core \
        libreoffice-writer \
        libreoffice-calc \
        libreoffice-impress \
        fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download PaddleOCR PP-OCRv5 models + PPStructureV3 table models
# during build (avoids ~300 MB download at runtime).
# Script is used instead of inline python -c to avoid Coolify ARG injection bug
# with multi-line backslash-continued RUN commands.
COPY scripts/download_models.py /tmp/download_models.py
RUN python /tmp/download_models.py || echo "Model pre-download failed; models will download at runtime"

# Copy application code
COPY app/ ./app/

# Create non-root user and cache/temp directories
# Copy PaddleX models if they were pre-downloaded
RUN useradd -m -r appuser && \
    mkdir -p /app/cache /tmp/libreoffice && \
    (cp -r /root/.paddlex /home/appuser/.paddlex 2>/dev/null || true) && \
    chown -R appuser:appuser /app /tmp/libreoffice /home/appuser

USER appuser

# Expose port
# BLAS thread count. Without this the recognition stack runs single-threaded and
# PaddleOCR-VL takes ~23x longer — 578 s against 25 s for the same page on a
# 28-core box. Paddle prints "OMP_NUM_THREADS set to 8, not 1" at startup,
# which reads like a warning and is actually the line confirming it is usable.
#
# A default rather than a requirement: override it to match the cores the
# container is actually given, or the threads oversubscribe and fight.
ENV OMP_NUM_THREADS=8

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn
CMD uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WORKERS:-1}
