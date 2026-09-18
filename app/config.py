"""Application configuration from environment variables."""

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

    # Cache
    cache_enabled: bool = True
    cache_dir: str = "/tmp/pdf_engine_cache"
    cache_max_size_mb: int = 500  # Max total cache size
    cache_ttl_seconds: int = 3600  # 1 hour default TTL

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
