"""Application configuration from environment variables."""

from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Authentication
    pdf_engine_secret: str = "change-me"

    # Server
    log_level: str = "info"
    max_upload_size_mb: int = 100
    workers: int = 1
    request_timeout_seconds: int = 120

    # Outbound source downloads. Requests fail closed unless the hostname is
    # listed explicitly (comma-separated, no schemes or paths). Production
    # should list only the S3 virtual-host endpoints used by TaxScout.
    pdf_source_allowed_hosts: str = ""

    # AGPL Corresponding Source. Built images turn this on so that an absent or
    # corrupt /app/build-commit refuses to start, instead of quietly advertising
    # the mutable main branch as the source of the code actually running.
    require_build_identity: bool = False

    # HMAC auth
    max_timestamp_drift_ms: int = 5 * 60 * 1000  # 5 minutes
    sensitive_auth_max_timestamp_drift_ms: int = 60 * 1000
    sensitive_nonce_cache_max_entries: int = 10_000
    sensitive_nonce_db_path: str = "/data/pdf-engine-sensitive-nonces.sqlite3"

    # OCR text detection runs on the page scaled so its longer side is at most
    # this many pixels; recognition still reads each line from the full-DPI
    # render. Unbounded detection on a 300-DPI letter page peaked at 5.7 GB and
    # was OOM-killed on the 4 GiB production host; 1280 peaks at ~1.55 GB, is
    # 3.5x faster and read 496/496 identifiers in small-print stress scans
    # (TAX-4858). 0 restores the unbounded PaddleOCR default.
    ocr_det_limit_side_len: int = 1280

    # Primary OCR for /text/ocr and identifier redaction. OCR_BACKEND=paddle is
    # the rollback path for both. Table extraction and orientation keep Paddle
    # models, which load lazily on first use of those endpoints.
    ocr_backend: Literal["paddle", "rapid"] = "rapid"
    rapidocr_model_dir: str = "/opt/rapidocr/models"

    # OCR job queue (TAX-6876). Every /text/ocr request is a persistent job:
    # its bytes are spilled under OCR_QUEUE_DIR and its state is kept in a
    # SQLite file there, so a restart resumes queued and in-flight work.
    # Keep this on the persistent /data volume.
    ocr_queue_dir: str = "/data/ocr-queue"
    # Concurrent OCR inferences. 0 = one per available CPU (cgroup quota aware).
    ocr_workers: int = 0
    # ONNX Runtime intra-op threads per worker session (and OpenCV threads).
    # 0 = available CPUs // OCR_WORKERS, at least 1, so workers x threads ~ cores.
    ocr_intra_op_threads: int = 0
    # Documents interleaved page by page at once; later ones wait FIFO.
    # 0 = max(2, OCR_WORKERS).
    ocr_max_active_jobs: int = 0
    # Unfinished OCR jobs accepted before /text/ocr answers 429 + Retry-After.
    ocr_queue_max_jobs: int = 500
    # Completed and failed OCR results stay readable this long.
    ocr_result_ttl_seconds: int = 3600

    @model_validator(mode="after")
    def single_process(self):
        # The OCR queue schedules and recovers jobs inside one process; two
        # uvicorn processes would both resume the same jobs after a restart.
        if self.workers != 1:
            raise ValueError(
                "The OCR job queue requires WORKERS=1; use OCR_WORKERS for parallel OCR"
            )
        return self

    # Cache
    cache_enabled: bool = True
    cache_dir: str = "/tmp/pdf_engine_cache"
    cache_max_size_mb: int = 500  # Max total cache size
    cache_ttl_seconds: int = 3600  # 1 hour default TTL

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
