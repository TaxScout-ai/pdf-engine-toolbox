"""Application configuration from environment variables."""

from typing import Literal

from pydantic import Field, model_validator
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

    # RapidOCR is the only model runtime (TAX-6995): /text/ocr, identifier
    # redaction, /transform/orientation and table recognition on page images
    # all use the pinned ONNX weights in this directory. OCR_BACKEND and
    # OCR_DET_LIMIT_SIDE_LEN belonged to the removed Paddle path and are
    # ignored (a warning is logged at startup when OCR_BACKEND is still set).
    rapidocr_model_dir: str = "/opt/rapidocr/models"
    # Optional, offline table structure fallback. Disabled until this model
    # build passes the financial-table evaluation for the deployment.
    table_fallback: Literal["none", "docling"] = "none"
    docling_model_dir: str = "/opt/docling/models"
    table_fallback_max_pages: int = Field(default=4, ge=0, le=50)

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
    # Key for OCR inputs, results and page checkpoints at rest (AES-256-GCM):
    # 32 bytes in URL-safe base64, or any passphrase (HKDF-derived). Empty =
    # derived from PDF_ENGINE_SECRET. Changing it makes queued jobs unreadable
    # (they fail with an authentication error).
    ocr_queue_key: str = ""
    # How often expired jobs, unreferenced files and orphan rows are swept.
    ocr_sweep_interval_seconds: float = 300
    # Synchronous OCR (identifier redaction) fails closed after this long.
    # Matches the app's redaction budget (OCR_TIMEOUT_MS default, 20 min).
    ocr_sync_timeout_seconds: float = 1200

    @model_validator(mode="after")
    def single_process(self):
        # The OCR queue schedules and recovers jobs inside one process; two
        # uvicorn processes would both resume the same jobs after a restart.
        if self.workers != 1:
            raise ValueError(
                f"WORKERS={self.workers} is not supported: WORKERS must be 1. The OCR job "
                "queue is scheduled and recovered by a single uvicorn process; two would "
                "resume the same jobs after a restart. Set WORKERS=1 and use OCR_WORKERS "
                "for parallel OCR."
            )
        return self

    # Cache
    cache_enabled: bool = True
    cache_dir: str = "/tmp/pdf_engine_cache"
    cache_max_size_mb: int = 500  # Max total cache size
    cache_ttl_seconds: int = 3600  # 1 hour default TTL

    # "ignore": a .env written for an older engine (OCR_BACKEND=paddle) must
    # not stop the service from starting.
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
