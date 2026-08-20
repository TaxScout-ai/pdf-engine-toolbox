#!/usr/bin/env python3
"""Verify a deployed PDF Engine's AGPL network source offer."""

import argparse
import hashlib
import json
import re
import sys
from typing import Any
from urllib.request import Request, urlopen

FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def fetch_json(url: str) -> tuple[dict[str, Any], dict[str, str]]:
    request = Request(url, headers={"User-Agent": "TaxScout-AGPL-Verifier/1.0"})
    with urlopen(request, timeout=20) as response:  # noqa: S310 - operator-supplied URL
        payload = json.load(response)
        headers = {key.lower(): value for key, value in response.headers.items()}
    return payload, headers


def probe_url(url: str) -> None:
    request = Request(
        url,
        method="HEAD",
        headers={"User-Agent": "TaxScout-AGPL-Verifier/1.0"},
    )
    with urlopen(request, timeout=20):  # noqa: S310 - source URLs are offer data
        return


def verify_hash(url: str, expected: str) -> None:
    digest = hashlib.sha256()
    request = Request(url, headers={"User-Agent": "TaxScout-AGPL-Verifier/1.0"})
    with urlopen(request, timeout=120) as response:  # noqa: S310
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"source archive hash mismatch: {url}")


def validate_offer(
    offer: dict[str, Any],
    offer_headers: dict[str, str],
    health: dict[str, Any],
    expected_commit: str,
) -> list[str]:
    if not FULL_SHA.fullmatch(expected_commit):
        raise ValueError("expected commit must be a full lowercase Git SHA")
    if offer.get("license") != "AGPL-3.0-only":
        raise ValueError("combined service does not report AGPL-3.0-only")
    if offer.get("project_license") != "AGPL-3.0-or-later":
        raise ValueError("project source does not report AGPL-3.0-or-later")
    if offer.get("build_commit") != expected_commit:
        raise ValueError("deployed build commit does not match the expected public revision")
    if health.get("build_commit") != expected_commit:
        raise ValueError("health and source endpoints disagree on build identity")

    source_url = str(offer.get("source_code_url", ""))
    license_url = str(offer.get("license_url", ""))
    expected_tree_suffix = f"/tree/{expected_commit}"
    if not source_url.endswith(expected_tree_suffix):
        raise ValueError("source offer does not point to the exact deployed revision")
    if expected_commit not in str(offer.get("source_archive_url", "")):
        raise ValueError("source archive is not revision-pinned")
    if expected_commit not in license_url:
        raise ValueError("license notice is not revision-pinned")
    if expected_commit not in str(offer.get("third_party_source_manifest_url", "")):
        raise ValueError("third-party source manifest is not revision-pinned")

    if offer_headers.get("x-source-code") != source_url:
        raise ValueError("X-Source-Code header is missing or inconsistent")
    link = offer_headers.get("link", "")
    if f'<{source_url}>; rel="source"' not in link:
        raise ValueError("Link rel=source is missing")
    if f'<{license_url}>; rel="license"' not in link:
        raise ValueError("Link rel=license is missing")

    components = offer.get("third_party_sources")
    if not isinstance(components, list):
        raise ValueError("third-party source list is missing")
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
            raise ValueError(f'invalid source hash for {component.get("name")}')

    return [
        source_url,
        str(offer["source_archive_url"]),
        license_url,
        str(offer["third_party_source_manifest_url"]),
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
    offer, offer_headers = fetch_json(f"{base_url}/source")
    health, _ = fetch_json(f"{base_url}/health")
    urls = validate_offer(offer, offer_headers, health, args.expected_commit)
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
