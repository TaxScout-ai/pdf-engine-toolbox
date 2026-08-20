"""Security policy tests for outbound source downloads."""

import asyncio
import socket
from unittest.mock import AsyncMock

import httpx
import pytest

from app.config import settings
from app.services import download_service
from app.utils.errors import DownloadFailedError, SourceUrlRejectedError

PRESIGNED_QUERY = (
    "X-Amz-Algorithm=AWS4-HMAC-SHA256"
    "&X-Amz-Credential=test"
    "&X-Amz-Signature=deadbeef"
)
ALLOWED_HOST = "taxscout-documents.s3.us-east-1.amazonaws.com"


@pytest.fixture(autouse=True)
def source_policy(monkeypatch):
    monkeypatch.setattr(settings, "pdf_source_allowed_hosts", ALLOWED_HOST)
    monkeypatch.setattr(settings, "max_upload_size_mb", 1)


def source_url(host: str = ALLOWED_HOST, path: str = "/client/file.pdf") -> str:
    return f"https://{host}{path}?{PRESIGNED_QUERY}"


def test_validate_source_url_accepts_exact_allowed_presigned_host():
    host, target = download_service._validate_source_url(source_url())

    assert host == ALLOWED_HOST
    assert target == f"/client/file.pdf?{PRESIGNED_QUERY}"


@pytest.mark.parametrize(
    "url",
    [
        f"http://{ALLOWED_HOST}/file.pdf?{PRESIGNED_QUERY}",
        f"https://169.254.169.254/latest/meta-data?{PRESIGNED_QUERY}",
        f"https://{ALLOWED_HOST}.evil.example/file.pdf?{PRESIGNED_QUERY}",
        f"https://user@{ALLOWED_HOST}/file.pdf?{PRESIGNED_QUERY}",
        f"https://{ALLOWED_HOST}:8443/file.pdf?{PRESIGNED_QUERY}",
        f"https://{ALLOWED_HOST}/file.pdf?{PRESIGNED_QUERY}#fragment",
        f"https://{ALLOWED_HOST}//169.254.169.254/latest/meta-data?{PRESIGNED_QUERY}",
        f"https://{ALLOWED_HOST}/\\evil.example/file.pdf?{PRESIGNED_QUERY}",
        f"https://{ALLOWED_HOST}/file%ZZ.pdf?{PRESIGNED_QUERY}",
        f"https://{ALLOWED_HOST}/file.pdf",
    ],
)
def test_validate_source_url_rejects_untrusted_variants(url):
    with pytest.raises(SourceUrlRejectedError):
        download_service._validate_source_url(url)


def test_validate_source_url_fails_closed_without_allowlist(monkeypatch):
    monkeypatch.setattr(settings, "pdf_source_allowed_hosts", "")

    with pytest.raises(SourceUrlRejectedError, match="allowlist is not configured"):
        download_service._validate_source_url(source_url())


def test_validate_source_url_accepts_percent_encoded_s3_target():
    url = source_url(
        path="/clients/Smith%20%26%20Co/2026%20return%20%281%29.pdf"
    )

    host, target = download_service._validate_source_url(url)

    assert host == ALLOWED_HOST
    assert target.startswith("/clients/Smith%20%26%20Co/")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address",
    [
        "10.0.0.5",
        "127.0.0.1",
        "169.254.169.254",
        "224.0.0.1",
        "fec0::1",
        "ff02::1",
        "::1",
    ],
)
async def test_public_dns_rejects_non_unicast_answer(monkeypatch, address):
    asyncio_loop = asyncio.get_running_loop()
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    monkeypatch.setattr(
        asyncio_loop,
        "getaddrinfo",
        AsyncMock(
            return_value=[
                (family, socket.SOCK_STREAM, 6, "", (address, 443))
            ]
        ),
    )

    with pytest.raises(SourceUrlRejectedError, match="non-public"):
        await download_service._resolve_public_addresses(ALLOWED_HOST)


@pytest.mark.asyncio
async def test_public_dns_returns_validated_address_without_reresolving(monkeypatch):
    asyncio_loop = asyncio.get_running_loop()
    resolver = AsyncMock(
        return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]
    )
    monkeypatch.setattr(asyncio_loop, "getaddrinfo", resolver)

    assert await download_service._resolve_public_addresses(ALLOWED_HOST) == (
        "93.184.216.34",
    )
    resolver.assert_awaited_once()


@pytest.mark.asyncio
async def test_http_client_ignores_ambient_proxy_and_ca_environment(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9998")
    monkeypatch.setenv("SSL_CERT_FILE", "/untrusted/environment/ca.pem")

    real_async_client = httpx.AsyncClient
    captured_kwargs = None

    def capture_async_client(**kwargs):
        nonlocal captured_kwargs
        captured_kwargs = kwargs
        return real_async_client(**kwargs)

    monkeypatch.setattr(download_service.httpx, "AsyncClient", capture_async_client)

    client = download_service._create_http_client("93.184.216.34")
    try:
        assert captured_kwargs is not None
        assert captured_kwargs["base_url"] == "https://93.184.216.34"
        assert captured_kwargs["follow_redirects"] is False
        assert captured_kwargs["trust_env"] is False
    finally:
        await client.aclose()


def install_mock_transport(monkeypatch, handler):
    monkeypatch.setattr(
        download_service,
        "_create_http_client",
        lambda address: httpx.AsyncClient(
            base_url=f"https://{address}",
            transport=httpx.MockTransport(handler),
            follow_redirects=False,
        ),
    )
    monkeypatch.setattr(
        download_service,
        "_resolve_public_addresses",
        AsyncMock(return_value=("93.184.216.34",)),
    )


@pytest.mark.asyncio
async def test_download_pins_validated_ip_with_original_host_and_tls_sni(monkeypatch):
    seen_request = None

    def handler(request):
        nonlocal seen_request
        seen_request = request
        return httpx.Response(200, content=b"%PDF-pinned")

    install_mock_transport(monkeypatch, handler)

    assert await download_service.download_pdf(source_url()) == b"%PDF-pinned"
    assert seen_request is not None
    assert seen_request.url.host == "93.184.216.34"
    assert seen_request.headers["host"] == ALLOWED_HOST
    assert seen_request.extensions["sni_hostname"] == ALLOWED_HOST


@pytest.mark.asyncio
async def test_download_rejects_redirect(monkeypatch):
    install_mock_transport(
        monkeypatch,
        lambda request: httpx.Response(302, headers={"location": "http://127.0.0.1"}),
    )

    with pytest.raises(DownloadFailedError, match="Redirects are not permitted"):
        await download_service.download_pdf(source_url())


@pytest.mark.asyncio
async def test_download_rejects_declared_oversize_body(monkeypatch):
    install_mock_transport(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            headers={"content-length": str(2 * 1024 * 1024)},
            content=b"",
        ),
    )

    with pytest.raises(DownloadFailedError, match="exceeds the maximum size"):
        await download_service.download_pdf(source_url())


@pytest.mark.asyncio
async def test_download_rejects_chunked_oversize_body(monkeypatch):
    class OversizeStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"a" * (600 * 1024)
            yield b"b" * (600 * 1024)

    install_mock_transport(
        monkeypatch,
        lambda request: httpx.Response(200, stream=OversizeStream()),
    )

    with pytest.raises(DownloadFailedError, match="exceeds the maximum size"):
        await download_service.download_pdf(source_url())


@pytest.mark.asyncio
async def test_download_streams_allowed_body(monkeypatch):
    install_mock_transport(
        monkeypatch,
        lambda request: httpx.Response(200, content=b"%PDF-test"),
    )

    assert await download_service.download_pdf(source_url()) == b"%PDF-test"
