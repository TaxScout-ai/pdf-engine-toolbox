"""HMAC-SHA256 request authentication."""

import asyncio
import hashlib
import hmac
import re
import time

from fastapi import Request

from app.config import settings
from app.utils.errors import AuthenticationError

_NONCE_PATTERN = re.compile(r"^[0-9a-fA-F]{32,64}$")
_nonce_lock = asyncio.Lock()
_used_sensitive_nonces: dict[str, int] = {}


async def _claim_sensitive_nonce(nonce: str, now_ms: int) -> None:
    async with _nonce_lock:
        expired = [
            value for value, expires_at in _used_sensitive_nonces.items() if expires_at <= now_ms
        ]
        for value in expired:
            _used_sensitive_nonces.pop(value, None)

        if nonce in _used_sensitive_nonces:
            raise AuthenticationError("Request replay detected")
        if len(_used_sensitive_nonces) >= settings.sensitive_nonce_cache_max_entries:
            raise AuthenticationError("Sensitive request replay cache is full")

        _used_sensitive_nonces[nonce] = now_ms + settings.sensitive_auth_max_timestamp_drift_ms


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
        await _claim_sensitive_nonce(nonce, now_ms)


async def verify_sensitive_hmac(request: Request) -> None:
    """Require replay-resistant HMAC v2 for a password-bearing request."""
    await verify_hmac(request, require_v2=True)
