"""Service for downloading binaries from approved presigned S3 URLs."""

import asyncio
import ipaddress
import re
import socket
from urllib.parse import parse_qs, urlsplit

import httpx
import structlog

from app.config import settings
from app.utils.errors import DownloadFailedError, SourceUrlRejectedError

log = structlog.get_logger()

_REQUIRED_PRESIGN_QUERY_KEYS = {
    "x-amz-algorithm",
    "x-amz-credential",
    "x-amz-signature",
}

# Keep the request target within an origin-form HTTP target. In particular,
# `//attacker.example/path` is forbidden: httpx resolves that network-path
# reference against base_url by replacing the approved authority.
#
# AWS presigned S3 URLs use RFC 3986 unreserved/sub-delim characters plus
# percent-encoded octets. A raw backslash, whitespace, fragment, malformed
# percent escape, or second leading slash is never required.
_SAFE_REQUEST_TARGET = re.compile(
    r"/(?!/)(?:[A-Za-z0-9._~!$&'()*+,;=:@/-]|%[0-9A-Fa-f]{2})*"
    r"(?:\?(?:[A-Za-z0-9._~!$&'()*+,;=:@/?-]|%[0-9A-Fa-f]{2})*)?"
)


def _configured_allowed_hosts() -> frozenset[str]:
    """Return canonical exact hostnames configured by the operator."""
    hosts = {
        host.strip().lower()
        for host in settings.pdf_source_allowed_hosts.split(",")
        if host.strip()
    }
    if not hosts:
        raise SourceUrlRejectedError("Source URL allowlist is not configured")

    for host in hosts:
        if (
            ":" in host
            or "/" in host
            or "@" in host
            or host.endswith(".")
            or host != host.encode("idna").decode("ascii")
        ):
            raise SourceUrlRejectedError("Source URL allowlist is invalid")
    return frozenset(hosts)


def _validate_source_url(url: str) -> tuple[str, str]:
    """Validate and split an approved presigned URL into host and request target.

    The returned host is selected from operator configuration rather than copied
    from the request. That keeps the HTTP client's network authority independent
    from caller-controlled input.
    """
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise SourceUrlRejectedError() from exc

    if parsed.scheme != "https":
        raise SourceUrlRejectedError("Source URL must use HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise SourceUrlRejectedError("Source URL userinfo is not permitted")
    if parsed.fragment:
        raise SourceUrlRejectedError("Source URL fragments are not permitted")
    if port not in (None, 443):
        raise SourceUrlRejectedError("Source URL port is not permitted")

    request_host = parsed.hostname
    if not request_host or request_host.endswith("."):
        raise SourceUrlRejectedError()
    try:
        request_host = request_host.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise SourceUrlRejectedError() from exc

    # Map the untrusted hostname to the exact canonical value loaded from
    # configuration. Do not use suffix matching: evil.example can otherwise
    # impersonate an allowed bucket hostname.
    allowed_hosts = {host: host for host in _configured_allowed_hosts()}
    canonical_host = allowed_hosts.get(request_host)
    if canonical_host is None:
        raise SourceUrlRejectedError("Source URL host is not permitted")

    query_keys = {key.lower() for key in parse_qs(parsed.query, keep_blank_values=True)}
    if not _REQUIRED_PRESIGN_QUERY_KEYS.issubset(query_keys):
        raise SourceUrlRejectedError("Source URL must be an AWS presigned URL")

    request_target = parsed.path or "/"
    if not request_target.startswith("/"):
        raise SourceUrlRejectedError()
    if parsed.query:
        request_target = f"{request_target}?{parsed.query}"
    if not _SAFE_REQUEST_TARGET.fullmatch(request_target):
        raise SourceUrlRejectedError("Source URL request target is not permitted")
    return canonical_host, request_target


def _is_public_address(raw_address: str) -> bool:
    try:
        address = ipaddress.ip_address(raw_address)
    except ValueError:
        return False
    return (
        address.is_global
        and not address.is_private
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_reserved
        and not address.is_unspecified
        and not getattr(address, "is_site_local", False)
    )


async def _resolve_public_addresses(host: str) -> tuple[str, ...]:
    """Resolve once and return only validated public unicast addresses."""
    try:
        records = await asyncio.get_running_loop().getaddrinfo(
            host,
            443,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise DownloadFailedError("Source URL hostname could not be resolved") from exc

    addresses = tuple(dict.fromkeys(record[4][0] for record in records if record[4]))
    if not addresses or not all(_is_public_address(address) for address in addresses):
        raise SourceUrlRejectedError("Source URL resolved to a non-public address")
    return addresses


def _create_http_client(address: str) -> httpx.AsyncClient:
    """Create a no-redirect client whose TCP authority is a validated IP."""
    parsed_address = ipaddress.ip_address(address)
    url_host = f"[{parsed_address}]" if parsed_address.version == 6 else str(parsed_address)
    return httpx.AsyncClient(
        base_url=f"https://{url_host}",
        timeout=httpx.Timeout(settings.request_timeout_seconds),
        follow_redirects=False,
    )


async def download_pdf(url: str) -> bytes:
    """Download a PDF from a presigned S3 URL.

    Args:
        url: Presigned S3 GET URL

    Returns:
        PDF file content as bytes

    Raises:
        DownloadFailedError: If download fails
    """
    host, request_target = _validate_source_url(url)
    addresses = await _resolve_public_addresses(host)
    pinned_address = addresses[0]
    max_bytes = settings.max_upload_size_mb * 1024 * 1024

    try:
        async with _create_http_client(pinned_address) as client:
            async with client.stream(
                "GET",
                request_target,
                headers={"Host": host},
                extensions={"sni_hostname": host},
            ) as response:
                if response.is_redirect:
                    raise DownloadFailedError("Redirects are not permitted for source URLs")
                response.raise_for_status()

                content_length = response.headers.get("content-length")
                if content_length is not None:
                    try:
                        declared_size = int(content_length)
                    except ValueError as exc:
                        raise DownloadFailedError(
                            "Source returned an invalid Content-Length"
                        ) from exc
                    if declared_size < 0 or declared_size > max_bytes:
                        raise DownloadFailedError("Source file exceeds the maximum size")

                chunks: list[bytes] = []
                total_bytes = 0
                async for chunk in response.aiter_bytes():
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        raise DownloadFailedError("Source file exceeds the maximum size")
                    chunks.append(chunk)

                content = b"".join(chunks)
                log.info("pdf_downloaded", size_bytes=len(content))
                return content

    except httpx.TimeoutException:
        raise DownloadFailedError("Timeout downloading PDF from source URL")
    except httpx.HTTPStatusError as e:
        raise DownloadFailedError(f"HTTP {e.response.status_code} downloading PDF")
    except httpx.RequestError:
        # Never include the exception string: httpx may embed the full
        # presigned URL and its temporary AWS credential in that message.
        raise DownloadFailedError("Network error downloading PDF")
