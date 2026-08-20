"""HMAC-SHA256 request authentication."""

import hashlib
import hmac
import re
import sqlite3
import time
from pathlib import Path

from fastapi import Request

from app.config import settings
from app.utils.errors import AuthenticationError

_NONCE_PATTERN = re.compile(r"^[0-9a-fA-F]{32,64}$")


def _claim_sensitive_nonce(nonce: str, now_ms: int, expires_at_ms: int) -> None:
    """Atomically claim a nonce in the restart- and replica-shared ledger."""
    path = Path(settings.sensitive_nonce_db_path)
    nonce_hash = hashlib.sha256(nonce.encode("ascii")).hexdigest()
    connection: sqlite3.Connection | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=5, isolation_level=None)
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sensitive_hmac_nonces (
                nonce_hash TEXT PRIMARY KEY,
                expires_at_ms INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            "DELETE FROM sensitive_hmac_nonces WHERE expires_at_ms < ?",
            (now_ms,),
        )
        count = connection.execute("SELECT COUNT(*) FROM sensitive_hmac_nonces").fetchone()[0]
        if count >= settings.sensitive_nonce_cache_max_entries:
            raise AuthenticationError("Sensitive request replay cache is full")
        try:
            connection.execute(
                "INSERT INTO sensitive_hmac_nonces (nonce_hash, expires_at_ms) VALUES (?, ?)",
                (nonce_hash, expires_at_ms),
            )
        except sqlite3.IntegrityError as error:
            raise AuthenticationError("Request replay detected") from error
        connection.execute("COMMIT")
    except AuthenticationError:
        if connection is not None and connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    except (OSError, sqlite3.Error) as error:
        if connection is not None and connection.in_transaction:
            connection.execute("ROLLBACK")
        raise AuthenticationError("Sensitive request replay store unavailable") from error
    finally:
        if connection is not None:
            connection.close()


async def verify_hmac(request: Request, *, require_v2: bool = False) -> None:
    """Verify HMAC-SHA256 signature on incoming request.

    Expected headers:
        X-Timestamp: milliseconds since epoch
        X-Signature: hex(HMAC-SHA256(secret, timestamp:method:path:body_hash))
    """
    timestamp = request.headers.get("X-Timestamp")
    signature = request.headers.get("X-Signature")
    auth_version = request.headers.get("X-Auth-Version")
    nonce = request.headers.get("X-Nonce")

    if not timestamp or not signature:
        raise AuthenticationError("Missing X-Timestamp or X-Signature headers")
    if require_v2:
        if settings.workers != 1:
            raise AuthenticationError("Sensitive HMAC v2 requires a single replay-cache worker")
        if auth_version != "2" or not nonce or not _NONCE_PATTERN.fullmatch(nonce):
            raise AuthenticationError("Missing or invalid HMAC v2 headers")

    # Check timestamp freshness
    try:
        ts = int(timestamp)
    except ValueError:
        raise AuthenticationError("Invalid timestamp format")

    now_ms = int(time.time() * 1000)
    max_drift = (
        settings.sensitive_auth_max_timestamp_drift_ms
        if require_v2
        else settings.max_timestamp_drift_ms
    )
    if abs(now_ms - ts) > max_drift:
        raise AuthenticationError("Request timestamp expired")

    # Read body and compute hash
    body = await request.body()
    body_hash = hashlib.sha256(body).hexdigest()

    # Build the message string
    if require_v2:
        message = f"v2:{timestamp}:{nonce}:{request.method}:{request.url.path}:{body_hash}"
    else:
        message = f"{timestamp}:{request.method}:{request.url.path}:{body_hash}"

    # Compute expected signature
    expected = hmac.new(
        settings.pdf_engine_secret.encode(),
        message.encode(),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(signature, expected):
        raise AuthenticationError("Invalid signature")

    if require_v2:
        # The cache entry must cover the request's full acceptance window. A
        # future-dated request accepted near +drift remains replayable until
        # signed timestamp + drift, not merely first-seen time + drift.
        _claim_sensitive_nonce(
            nonce,
            now_ms,
            ts + settings.sensitive_auth_max_timestamp_drift_ms,
        )


async def verify_sensitive_hmac(request: Request) -> None:
    """Require replay-resistant HMAC v2 for a password-bearing request."""
    await verify_hmac(request, require_v2=True)
