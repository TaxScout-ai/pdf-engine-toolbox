#!/usr/bin/env python3
"""Bind runtime/build inputs to the exact public source revision they advertise."""

import argparse
import hashlib
import http.client
import ipaddress
import re
import socket
import ssl
import sys
import tarfile
import tempfile
from pathlib import Path

FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY_URL = "https://github.com/TaxScout-ai/pdf-engine-toolbox"
CODELOAD_HOST = "codeload.github.com"
CODELOAD_PATH = "/TaxScout-ai/pdf-engine-toolbox/tar.gz"
MAX_SOURCE_ARCHIVE_BYTES = 50 * 1024 * 1024
MAX_EXTRACTED_SOURCE_BYTES = 200 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 10_000
SYSTEM_CA_FILES = (
    Path("/etc/ssl/certs/ca-certificates.crt"),
    Path("/etc/pki/tls/certs/ca-bundle.crt"),
    Path("/etc/ssl/cert.pem"),
)
EXACT_FILES = {
    "Dockerfile",
    "LICENSE",
    "NOTICE.md",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "third-party-sources.json",
    "scripts/download_models.py",
    "scripts/verify_source_revision.py",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def scoped_files(root: Path) -> dict[str, str]:
    paths = set(EXACT_FILES)
    app_root = root / "app"
    if not app_root.is_dir():
        raise ValueError(f"application source is missing from {root}")
    paths.update(
        path.relative_to(root).as_posix()
        for path in app_root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )

    missing = sorted(path for path in paths if not (root / path).is_file())
    if missing:
        raise ValueError(f"required source inputs are missing: {', '.join(missing)}")
    return {path: digest(root / path) for path in sorted(paths)}


def compare_sources(local_root: Path, public_root: Path) -> None:
    local = scoped_files(local_root)
    public = scoped_files(public_root)
    if local.keys() != public.keys():
        missing = sorted(public.keys() - local.keys())
        extra = sorted(local.keys() - public.keys())
        raise ValueError(
            f"source input set differs from public revision (missing={missing}, extra={extra})"
        )
    changed = [path for path in local if local[path] != public[path]]
    if changed:
        raise ValueError(
            "source inputs differ from advertised public revision: " + ", ".join(changed)
        )


def _is_public_unicast(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        address.is_global
        and not address.is_multicast
        and not getattr(address, "is_site_local", False)
    )


def _resolve_codeload_addresses() -> tuple[str, ...]:
    try:
        answers = socket.getaddrinfo(CODELOAD_HOST, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError("GitHub codeload hostname does not resolve") from exc
    addresses = tuple(dict.fromkeys(answer[4][0] for answer in answers))
    if not addresses:
        raise ValueError("GitHub codeload hostname has no addresses")
    if any(not _is_public_unicast(ipaddress.ip_address(address)) for address in addresses):
        raise ValueError("GitHub codeload DNS includes a non-public address")
    return addresses


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
    def __init__(self, address: str, timeout: int):
        super().__init__(
            CODELOAD_HOST,
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


def _download_archive(commit: str, archive_path: Path) -> None:
    if not FULL_SHA.fullmatch(commit):
        raise ValueError("SOURCE_COMMIT must be a full lowercase Git SHA")
    addresses = _resolve_codeload_addresses()
    connection = _PinnedHTTPSConnection(addresses[0], timeout=60)
    try:
        connection.request(
            "GET",
            f"{CODELOAD_PATH}/{commit}",
            headers={"User-Agent": "TaxScout-Source-Revision-Verifier/1.0"},
        )
        response = connection.getresponse()
        if response.status != 200:
            response.close()
            raise OSError(f"GitHub codeload returned HTTP {response.status}")
        downloaded = 0
        try:
            with archive_path.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    downloaded += len(chunk)
                    if downloaded > MAX_SOURCE_ARCHIVE_BYTES:
                        raise ValueError("public source archive exceeds size limit")
                    output.write(chunk)
        finally:
            response.close()
    finally:
        connection.close()


def _extract_archive(archive_path: Path, extract_root: Path) -> None:
    extract_root.mkdir()
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members: list[tarfile.TarInfo] = []
        expanded_bytes = 0
        for member in archive:
            if len(members) >= MAX_ARCHIVE_MEMBERS:
                raise ValueError("public source archive has too many members")
            if member.isfile():
                expanded_bytes += member.size
                if expanded_bytes > MAX_EXTRACTED_SOURCE_BYTES:
                    raise ValueError("public source archive exceeds expanded size limit")
            members.append(member)
        archive.extractall(extract_root, members=members, filter="data")


def download_public_source(commit: str, destination: Path) -> Path:
    archive_path = destination / "source.tar.gz"
    _download_archive(commit, archive_path)

    extract_root = destination / "public"
    _extract_archive(archive_path, extract_root)
    roots = [path for path in extract_root.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise ValueError("public source archive has an unexpected layout")
    return roots[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("commit", help="Public full 40-character lowercase Git SHA")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--marker", type=Path)
    args = parser.parse_args()

    if not FULL_SHA.fullmatch(args.commit):
        raise ValueError("SOURCE_COMMIT must be a full lowercase Git SHA")
    local_root = args.root.resolve()
    with tempfile.TemporaryDirectory(prefix="pdf-engine-source-") as temporary:
        public_root = download_public_source(args.commit, Path(temporary))
        compare_sources(local_root, public_root)
    if args.marker:
        args.marker.write_text(f"{args.commit}\n", encoding="ascii")
    print(f"Exact public source revision: PASS ({args.commit})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError, tarfile.TarError) as exc:
        print(f"Exact public source revision: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
