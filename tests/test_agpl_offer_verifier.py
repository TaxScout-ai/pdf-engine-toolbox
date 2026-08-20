"""Tests for the post-deployment AGPL source-offer verifier."""

import pytest

from scripts.verify_agpl_offer import validate_offer

COMMIT = "a" * 40
REPOSITORY = "https://github.com/TaxScout-ai/pdf-engine-toolbox"


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
