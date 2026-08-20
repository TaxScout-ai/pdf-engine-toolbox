"""Tests for the post-deployment AGPL source-offer verifier."""

import io
import socket

import pytest

import scripts.verify_agpl_offer as verifier
from scripts.verify_agpl_offer import validate_offer

COMMIT = "a" * 40
REPOSITORY = "https://github.com/TaxScout-ai/pdf-engine-toolbox"


@pytest.fixture(autouse=True)
def resolve_test_hosts_to_public_address(monkeypatch):
    def getaddrinfo(_hostname, port, **_kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", port))]

    monkeypatch.setattr(verifier.socket, "getaddrinfo", getaddrinfo)


def valid_offer():
    source_url = f"{REPOSITORY}/tree/{COMMIT}"
    license_url = f"{REPOSITORY}/blob/{COMMIT}/LICENSE"
    offer = {
        "license": "AGPL-3.0-only",
        "project_license": "AGPL-3.0-or-later",
        "repository_url": REPOSITORY,
        "build_commit": COMMIT,
        "source_code_url": source_url,
        "source_archive_url": f"{REPOSITORY}/archive/{COMMIT}.tar.gz",
        "license_url": license_url,
        "third_party_source_manifest_url": (f"{REPOSITORY}/blob/{COMMIT}/third-party-sources.json"),
        "third_party_sources": [
            {
                "name": "PyMuPDF",
                "license": "AGPL-3.0-only",
                "source_url": "https://example.test/pymupdf.tar.gz",
                "sha256": "b" * 64,
            },
            {
                "name": "MuPDF",
                "license": "AGPL-3.0-or-later",
                "source_url": "https://example.test/mupdf.tar.gz",
                "sha256": "c" * 64,
            },
        ],
    }
    headers = {
        "x-source-code": source_url,
        "link": f'<{source_url}>; rel="source", <{license_url}>; rel="license"',
    }
    health = {"build_commit": COMMIT}
    return offer, headers, health


def trusted_components(offer):
    return offer["third_party_sources"]


def test_valid_offer_returns_every_url_that_must_remain_available():
    offer, headers, health = valid_offer()

    urls = validate_offer(offer, headers, health, COMMIT, trusted_components(offer))

    assert urls == [
        offer["source_code_url"],
        offer["source_archive_url"],
        offer["license_url"],
        offer["third_party_source_manifest_url"],
        offer["third_party_sources"][0]["source_url"],
        offer["third_party_sources"][1]["source_url"],
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("build_commit", "d" * 40, "deployed build commit"),
        ("license", "Proprietary", "does not report AGPL"),
        ("source_code_url", REPOSITORY, "trusted revision-pinned URL"),
    ],
)
def test_offer_fails_closed_on_identity_or_license_drift(field, value, message):
    offer, headers, health = valid_offer()
    offer[field] = value

    with pytest.raises(ValueError, match=message):
        validate_offer(offer, headers, health, COMMIT, trusted_components(offer))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_code_url", f"https://attacker.example/tree/{COMMIT}"),
        ("source_archive_url", f"https://attacker.example/{COMMIT}.tar.gz"),
        ("license_url", f"https://attacker.example/{COMMIT}/LICENSE"),
        (
            "third_party_source_manifest_url",
            f"https://attacker.example/{COMMIT}/third-party-sources.json",
        ),
    ],
)
def test_offer_rejects_counterfeit_revision_urls(field, value):
    offer, headers, health = valid_offer()
    offer[field] = value

    with pytest.raises(ValueError, match="trusted revision-pinned URL"):
        validate_offer(offer, headers, health, COMMIT, trusted_components(offer))


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/internal",
        "https://127.0.0.1/internal",
        "https://localhost/internal",
        "https://user:password@example.test/source.tar.gz",
    ],
)
def test_offer_rejects_untrusted_or_private_component_destinations(url):
    offer, headers, health = valid_offer()
    trusted = [dict(component) for component in trusted_components(offer)]
    offer["third_party_sources"][0]["source_url"] = url
    trusted[0]["source_url"] = url

    with pytest.raises(ValueError, match="source URL"):
        validate_offer(offer, headers, health, COMMIT, trusted)


def test_offer_rejects_component_metadata_not_in_trusted_revision():
    offer, headers, health = valid_offer()
    trusted = [dict(component) for component in trusted_components(offer)]
    offer["third_party_sources"][0]["sha256"] = "d" * 64

    with pytest.raises(ValueError, match="trusted revision manifest"):
        validate_offer(offer, headers, health, COMMIT, trusted)


@pytest.mark.parametrize(
    "hostname",
    [
        "2130706433",
        "0x7f000001",
        "localhost.localdomain",
        "127.0.0.1.nip.io",
    ],
)
def test_dns_resolution_rejects_noncanonical_loopback_hosts(monkeypatch, hostname):
    def resolve_loopback(_hostname, port, **_kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", port))]

    monkeypatch.setattr(verifier.socket, "getaddrinfo", resolve_loopback)

    with pytest.raises(ValueError, match="non-public address"):
        verifier._resolve_public_https_url(f"https://{hostname}/source.tar.gz")


@pytest.mark.parametrize(
    "address",
    ["224.0.0.1", "239.255.255.250", "ff02::1", "ff0e::1", "fec0::1"],
)
@pytest.mark.parametrize("as_dns_answer", [False, True])
def test_resolution_rejects_non_unicast_special_ranges(monkeypatch, address, as_dns_answer):
    if as_dns_answer:
        family = socket.AF_INET6 if ":" in address else socket.AF_INET

        def resolve_special(_hostname, port, **_kwargs):
            return [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port))]

        monkeypatch.setattr(verifier.socket, "getaddrinfo", resolve_special)
        url = "https://special-range.example/source.tar.gz"
    else:
        url = (
            f"https://[{address}]/source.tar.gz"
            if ":" in address
            else f"https://{address}/source.tar.gz"
        )

    with pytest.raises(ValueError, match="public address|non-public address"):
        verifier._resolve_public_https_url(url)


def test_redirect_is_revalidated_before_second_connection(monkeypatch):
    class FakeResponse:
        status = 302

        def getheader(self, name):
            assert name == "Location"
            return "https://redirect-to-loopback.test/internal"

        def read(self):
            return b""

        def close(self):
            return None

    class FakeConnection:
        connections = 0

        def __init__(self, *_args, **_kwargs):
            self.__class__.connections += 1

        def request(self, *_args, **_kwargs):
            return None

        def getresponse(self):
            return FakeResponse()

        def close(self):
            return None

    def resolve(hostname, port, **_kwargs):
        address = "127.0.0.1" if hostname == "redirect-to-loopback.test" else "8.8.8.8"
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port))]

    monkeypatch.setattr(verifier.socket, "getaddrinfo", resolve)
    monkeypatch.setattr(verifier, "_PinnedHTTPSConnection", FakeConnection)

    with pytest.raises(ValueError, match="non-public address"):
        verifier.probe_url("https://public.example/source.tar.gz")
    assert FakeConnection.connections == 1


def test_deployment_identity_fetch_ignores_proxy_and_rejects_redirects(monkeypatch):
    class FakeResponse(io.BytesIO):
        status = 302

        def __init__(self):
            super().__init__(b"")

        def getheader(self, name):
            assert name == "Location"
            return "https://counterfeit.example/source"

        def read(self, *_args, **_kwargs):
            raise AssertionError("rejected response bodies must not be drained")

    class FakeConnection:
        seen_hosts = []

        def __init__(self, hostname, _address, _timeout):
            self.__class__.seen_hosts.append(hostname)

        def request(self, *_args, **_kwargs):
            return None

        def getresponse(self):
            return FakeResponse()

        def close(self):
            return None

    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7777")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:7777")
    monkeypatch.setattr(verifier, "_PinnedHTTPSConnection", FakeConnection)

    with pytest.raises(ValueError, match="redirect limit"):
        verifier.fetch_public_json(
            "https://deployment.example/source",
            allow_redirects=False,
        )
    assert FakeConnection.seen_hosts == ["deployment.example"]


def test_tls_context_ignores_poisoned_ca_environment(monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", "/nonexistent/hostile-ca.pem")
    monkeypatch.setenv("SSL_CERT_DIR", "/nonexistent/hostile-ca-directory")

    context = verifier._trusted_tls_context()

    assert context.verify_mode == verifier.ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert context.get_ca_certs()


def test_json_fetch_rejects_oversized_success_response(monkeypatch):
    class OversizedResponse:
        def read1(self, size):
            return b"x" * size

    monkeypatch.setattr(verifier, "MAX_JSON_RESPONSE_BYTES", 16)

    with pytest.raises(ValueError, match="size limit"):
        verifier._read_bounded_json(OversizedResponse())


def test_json_fetch_enforces_total_deadline_during_stream(monkeypatch):
    class SlowResponse:
        reads = 0

        def read1(self, _size):
            self.reads += 1
            return b"{"

    response = SlowResponse()
    moments = iter((0.0, 0.25, 2.0))
    monkeypatch.setattr(verifier, "JSON_TOTAL_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(verifier.time, "monotonic", lambda: next(moments))

    with pytest.raises(TimeoutError, match="total time limit"):
        verifier._read_bounded_json(response)
    assert response.reads == 1
