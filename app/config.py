"""Application configuration from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Authentication
    pdf_engine_secret: str = "change-me"

    # Server
    log_level: str = "info"
    max_upload_size_mb: int = 100
    workers: int = 2
    request_timeout_seconds: int = 120

    # Outbound source downloads. Requests fail closed unless the hostname is
    # listed explicitly (comma-separated, no schemes or paths). Production
    # should list only the S3 virtual-host endpoints used by TaxScout.
    pdf_source_allowed_hosts: str = ""

    # HMAC auth
    max_timestamp_drift_ms: int = 5 * 60 * 1000  # 5 minutes

    # Cache
    cache_enabled: bool = True
    cache_dir: str = "/tmp/pdf_engine_cache"
    cache_max_size_mb: int = 500  # Max total cache size
    cache_ttl_seconds: int = 3600  # 1 hour default TTL

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
