ARG SOURCE_COMMIT

# Fail the build unless every source/build input copied into the runtime image
# is byte-for-byte identical to the public Git revision advertised by it.
FROM python:3.12-slim AS source-verifier
ARG SOURCE_COMMIT
WORKDIR /source-context
COPY app/ ./app/
COPY scripts/download_models.py scripts/verify_source_revision.py ./scripts/
COPY Dockerfile requirements.txt pyproject.toml LICENSE NOTICE.md README.md third-party-sources.json ./
RUN python scripts/verify_source_revision.py \
        "$SOURCE_COMMIT" \
        --root /source-context \
        --marker /source-commit.verified

FROM python:3.12-slim
ARG SOURCE_COMMIT

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
COPY LICENSE NOTICE.md README.md third-party-sources.json ./
RUN find /app/app -type d -name __pycache__ -prune -exec rm -r -- {} + && \
    find /app/app -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

# The verifier stage downloads the advertised public archive and compares all
# runtime/build inputs before emitting this marker. Referencing the stage here
# makes that proof a required dependency of the final image under BuildKit.
COPY --from=source-verifier /source-commit.verified /app/build-commit
RUN test "$(cat /app/build-commit)" = "$SOURCE_COMMIT" && chmod 0444 /app/build-commit
LABEL org.opencontainers.image.source="https://github.com/TaxScout-ai/pdf-engine-toolbox"
LABEL org.opencontainers.image.licenses="AGPL-3.0-only"
LABEL org.opencontainers.image.revision=$SOURCE_COMMIT

# Create non-root user and cache/temp directories
# Copy PaddleX models if they were pre-downloaded
RUN useradd -m -r appuser && \
    mkdir -p /app/cache /tmp/libreoffice && \
    (cp -r /root/.paddlex /home/appuser/.paddlex 2>/dev/null || true) && \
    chown -R appuser:appuser /app/cache /tmp/libreoffice /home/appuser

USER appuser

ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn. The explicit exec preserves signal delivery while allowing
# WORKERS to remain runtime-configurable.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers \"${WORKERS:-1}\""]
