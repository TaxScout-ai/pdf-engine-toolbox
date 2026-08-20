#!/usr/bin/env python3
"""Verify a deployed PDF Engine's AGPL network source offer."""

import argparse
import hashlib
import http.client
import ipaddress
import json
import re
import signal
import socket
import ssl
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
REPOSITORY = "https://github.com/TaxScout-ai/pdf-engine-toolbox"
RAW_REPOSITORY = "https://raw.githubusercontent.com/TaxScout-ai/pdf-engine-toolbox"
SYSTEM_CA_FILES = (
    Path("/etc/ssl/certs/ca-certificates.crt"),
    Path("/etc/pki/tls/certs/ca-bundle.crt"),
    Path("/etc/ssl/cert.pem"),
)
MAX_JSON_RESPONSE_BYTES = 1024 * 1024
JSON_TOTAL_TIMEOUT_SECONDS = 20
JSON_READ_CHUNK_BYTES = 64 * 1024


def _trusted_tls_context() -> ssl.SSLContext:
    """Load OS trust roots without honoring SSL_CERT_FILE/SSL_CERT_DIR."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    for ca_file in SYSTEM_CA_FILES:
        if ca_file.is_file():
            context.load_verify_locations(cafile=str(ca_file))
            return context
    raise OSError("no supported system CA bundle is available")


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection whose TCP peer is the address already validated below."""

    def __init__(self, hostname: str, address: str, timeout: float):
        super().__init__(
            hostname,
            port=443,
            timeout=timeout,
            context=_trusted_tls_context(),
        )
        self._validated_address = address

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._validated_address, self.port),
            self.timeout,
            self.source_address,
        )
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("JSON request exceeded total time limit")
    return remaining


@contextmanager
def _json_request_deadline() -> Iterator[float]:
    """Enforce one wall-clock limit across connect, headers, and JSON body."""
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        raise OSError("platform cannot enforce the JSON request total deadline")
    if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        raise OSError("cannot replace an existing process alarm")

    def timeout_handler(_signum, _frame):
        raise TimeoutError("JSON request exceeded total time limit")

    try:
        previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    except ValueError as exc:
        raise OSError("JSON verifier must run on the main thread") from exc

    deadline = time.monotonic() + JSON_TOTAL_TIMEOUT_SECONDS
    signal.setitimer(signal.ITIMER_REAL, JSON_TOTAL_TIMEOUT_SECONDS)
    try:
        yield deadline
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _resolve_public_https_url(url: str) -> tuple[str, str, tuple[str, ...]]:
    """Resolve one HTTPS URL and require every address to be public unicast."""
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(f"source URL must use HTTPS: {url}")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"source URL has an invalid port: {url}") from exc
    if parsed.username or parsed.password or port not in (None, 443):
        raise ValueError(f"source URL has unsafe authority: {url}")
    if parsed.fragment:
        raise ValueError(f"source URL must not contain a fragment: {url}")

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError(f"source URL must not target localhost: {url}")

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        if not _is_public_unicast(address):
            raise ValueError(f"source URL must target a public address: {url}")

    try:
        answers = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"source URL hostname does not resolve: {url}") from exc
    addresses = tuple(dict.fromkeys(answer[4][0] for answer in answers))
    if not addresses:
        raise ValueError(f"source URL hostname has no addresses: {url}")
    for resolved in addresses:
        if not _is_public_unicast(ipaddress.ip_address(resolved)):
            raise ValueError(f"source URL DNS includes a non-public address: {url}")

    request_target = parsed.path or "/"
    if parsed.query:
        request_target = f"{request_target}?{parsed.query}"
    return hostname, request_target, addresses


def _is_public_unicast(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Distinguish routable unicast from multicast/site-local special ranges."""
    return (
        address.is_global
        and not address.is_multicast
        and not getattr(address, "is_site_local", False)
    )


@contextmanager
def _open_public_https(
    url: str,
    *,
    method: str,
    timeout: int,
    max_redirects: int = 5,
    deadline: float | None = None,
) -> Iterator[http.client.HTTPResponse]:
    """Open a public HTTPS URL without proxy trust or DNS-rebinding exposure."""
    current_url = url
    for redirect_count in range(max_redirects + 1):
        hostname, request_target, addresses = _resolve_public_https_url(current_url)
        connection_timeout = timeout
        if deadline is not None:
            connection_timeout = min(timeout, _remaining_seconds(deadline))
        connection = _PinnedHTTPSConnection(hostname, addresses[0], connection_timeout)
        try:
            connection.request(
                method,
                request_target,
                headers={"User-Agent": "TaxScout-AGPL-Verifier/1.0"},
            )
            connection_socket = getattr(connection, "sock", None)
            if deadline is not None and connection_socket is not None:
                connection_socket.settimeout(_remaining_seconds(deadline))
            response = connection.getresponse()
            if deadline is not None:
                _remaining_seconds(deadline)
        except Exception:
            connection.close()
            raise

        if response.status in {301, 302, 303, 307, 308}:
            location = response.getheader("Location")
            response.close()
            connection.close()
            if not location:
                raise ValueError(f"source URL redirect has no Location: {current_url}")
            if redirect_count == max_redirects:
                raise ValueError(f"source URL exceeded redirect limit: {url}")
            current_url = urljoin(current_url, location)
            continue

        if response.status < 200 or response.status >= 300:
            response.close()
            connection.close()
            raise OSError(f"source URL returned HTTP {response.status}: {current_url}")
        try:
            yield response
        finally:
            response.close()
            connection.close()
        return
    raise ValueError(f"source URL exceeded redirect limit: {url}")


def probe_url(url: str) -> None:
    with _open_public_https(url, method="HEAD", timeout=20):
        return


def _set_response_socket_timeout(
    response: http.client.HTTPResponse,
    timeout_seconds: float,
) -> None:
    """Apply the remaining total deadline to the underlying socket when available."""
    fp = getattr(response, "fp", None)
    raw = getattr(fp, "raw", None)
    sock = getattr(raw, "_sock", None)
    if sock is not None:
        sock.settimeout(max(0.001, timeout_seconds))


def _read_bounded_json(
    response: http.client.HTTPResponse,
    deadline: float,
) -> dict[str, Any]:
    """Decode one JSON object within strict byte and wall-clock limits."""
    chunks: list[bytes] = []
    total = 0
    reader = getattr(response, "read1", None)
    if reader is None:
        reader = response.read

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("JSON request exceeded total time limit")
        _set_response_socket_timeout(response, remaining)
        chunk = reader(min(JSON_READ_CHUNK_BYTES, MAX_JSON_RESPONSE_BYTES - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_JSON_RESPONSE_BYTES:
            raise ValueError("JSON response exceeds size limit")
        chunks.append(chunk)

    payload = json.loads(b"".join(chunks))
    if not isinstance(payload, dict):
        raise ValueError("JSON response must be an object")
    return payload


def fetch_public_json(
    url: str,
    *,
    allow_redirects: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Fetch JSON over the pinned public transport and return normalized headers."""
    with _json_request_deadline() as deadline:
        with _open_public_https(
            url,
            method="GET",
            timeout=JSON_TOTAL_TIMEOUT_SECONDS,
            max_redirects=5 if allow_redirects else 0,
            deadline=deadline,
        ) as response:
            payload = _read_bounded_json(response, deadline)
            headers = {key.lower(): value for key, value in response.getheaders()}
    return payload, headers


def verify_hash(url: str, expected: str) -> None:
    digest = hashlib.sha256()
    with _open_public_https(url, method="GET", timeout=120) as response:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"source archive hash mismatch: {url}")


def trusted_manifest_url(expected_commit: str) -> str:
    return f"{RAW_REPOSITORY}/{expected_commit}/third-party-sources.json"


def fetch_trusted_components(expected_commit: str) -> list[dict[str, Any]]:
    """Load component metadata from the expected public Git revision."""
    payload, _ = fetch_public_json(
        trusted_manifest_url(expected_commit),
        allow_redirects=False,
    )
    components = payload.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("trusted third-party source manifest is empty")
    return components


def validate_offer(
    offer: dict[str, Any],
    offer_headers: dict[str, str],
    health: dict[str, Any],
    expected_commit: str,
    trusted_components: list[dict[str, Any]],
    *,
    health_headers: dict[str, str],
) -> list[str]:
    if not FULL_SHA.fullmatch(expected_commit):
        raise ValueError("expected commit must be a full lowercase Git SHA")
    if offer.get("license") != "AGPL-3.0-only":
        raise ValueError("combined service does not report AGPL-3.0-only")
    if offer.get("project_license") != "AGPL-3.0-or-later":
        raise ValueError("project source does not report AGPL-3.0-or-later")
    if offer.get("build_commit") != expected_commit:
        raise ValueError("deployed build commit does not match the expected public revision")
    source_url = f"{REPOSITORY}/tree/{expected_commit}"
    archive_url = f"{REPOSITORY}/archive/{expected_commit}.tar.gz"
    license_url = f"{REPOSITORY}/blob/{expected_commit}/LICENSE"
    manifest_url = f"{REPOSITORY}/blob/{expected_commit}/third-party-sources.json"
    exact_offer_fields = {
        "repository_url": REPOSITORY,
        "source_code_url": source_url,
        "source_archive_url": archive_url,
        "license_url": license_url,
        "third_party_source_manifest_url": manifest_url,
    }
    for field, expected_value in exact_offer_fields.items():
        if offer.get(field) != expected_value:
            raise ValueError(f"{field} does not match the trusted revision-pinned URL")

    exact_health_fields = {
        "build_commit": expected_commit,
        "license": "AGPL-3.0-only",
        "project_license": "AGPL-3.0-or-later",
        "source_code_url": source_url,
    }
    for field, expected_value in exact_health_fields.items():
        if health.get(field) != expected_value:
            raise ValueError(f"health {field} does not match the trusted build identity")

    for endpoint, headers in (("source", offer_headers), ("health", health_headers)):
        if headers.get("x-source-code") != source_url:
            raise ValueError(f"{endpoint} X-Source-Code header is missing or inconsistent")
        link = headers.get("link", "")
        if f'<{source_url}>; rel="source"' not in link:
            raise ValueError(f"{endpoint} Link rel=source is missing")
        if f'<{license_url}>; rel="license"' not in link:
            raise ValueError(f"{endpoint} Link rel=license is missing")

    components = offer.get("third_party_sources")
    if not isinstance(components, list):
        raise ValueError("third-party source list is missing")
    if components != trusted_components:
        raise ValueError("third-party sources do not match the trusted revision manifest")
    names = {component.get("name") for component in components}
    if not {"PyMuPDF", "MuPDF"}.issubset(names):
        raise ValueError("PyMuPDF/MuPDF Corresponding Source is missing")
    by_name = {component.get("name"): component for component in components}
    if by_name["PyMuPDF"].get("license") != "AGPL-3.0-only":
        raise ValueError("PyMuPDF license metadata is inaccurate")
    if by_name["MuPDF"].get("license") != "AGPL-3.0-or-later":
        raise ValueError("MuPDF license metadata is inaccurate")
    for component in components:
        if not SHA256.fullmatch(str(component.get("sha256", ""))):
            raise ValueError(f"invalid source hash for {component.get('name')}")
        _resolve_public_https_url(str(component.get("source_url", "")))

    return [
        source_url,
        archive_url,
        license_url,
        manifest_url,
        *(str(component["source_url"]) for component in components),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url", help="Deployed PDF Engine base URL")
    parser.add_argument("expected_commit", help="Expected public 40-character Git SHA")
    parser.add_argument(
        "--verify-third-party-hashes",
        action="store_true",
        help="Download the large PyMuPDF/MuPDF archives and verify SHA-256",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    offer, offer_headers = fetch_public_json(
        f"{base_url}/source",
        allow_redirects=False,
    )
    health, health_headers = fetch_public_json(
        f"{base_url}/health",
        allow_redirects=False,
    )
    trusted_components = fetch_trusted_components(args.expected_commit)
    urls = validate_offer(
        offer,
        offer_headers,
        health,
        args.expected_commit,
        trusted_components,
        health_headers=health_headers,
    )
    for url in urls:
        probe_url(url)

    if args.verify_third_party_hashes:
        for component in offer["third_party_sources"]:
            verify_hash(component["source_url"], component["sha256"])

    print(f"AGPL deployed-source contract: PASS ({args.expected_commit})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"AGPL deployed-source contract: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
