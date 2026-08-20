"""FastAPI dependency injection."""

from fastapi import Request

from app.auth.hmac_auth import verify_hmac, verify_sensitive_hmac


async def require_auth(request: Request) -> None:
    """Dependency that enforces HMAC authentication on a route."""
    await verify_hmac(request)


async def require_sensitive_auth(request: Request) -> None:
    """Require replay-resistant authentication for password-bearing routes."""
    await verify_sensitive_hmac(request)
