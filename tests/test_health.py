"""Test health endpoint."""

import json
from pathlib import Path

from app import build_identity


def test_health_check(client):
    """Health endpoint should return OK without auth."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"
    assert data["build_commit"] == "development"
    assert "pymupdf_version" in data
    assert data["license"] == "AGPL-3.0-or-later"
    assert data["source_code_url"] == build_identity.REPOSITORY_URL


def test_every_response_prominently_offers_source(client):
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.headers["X-Source-Code"] == build_identity.REPOSITORY_URL
    assert f'<{build_identity.REPOSITORY_URL}>; rel="source"' in response.headers["Link"]
    assert f'<{build_identity.LICENSE_URL}>; rel="license"' in response.headers["Link"]


def test_source_offer_is_public_and_downloadable(client):
    response = client.get("/source")

    assert response.status_code == 200
    data = response.json()
    assert data == {
        "license": "AGPL-3.0-or-later",
        "license_url": build_identity.LICENSE_URL,
        "repository_url": build_identity.REPOSITORY_URL,
        "build_commit": "development",
        "source_code_url": build_identity.REPOSITORY_URL,
        "source_archive_url": (
            f"{build_identity.REPOSITORY_URL}/archive/refs/heads/main.tar.gz"
        ),
        "third_party_source_manifest_url": (
            f"{build_identity.REPOSITORY_URL}/blob/main/third-party-sources.json"
        ),
        "third_party_sources": build_identity.read_third_party_sources(),
    }


def test_exact_build_identity_links_to_exact_source(client, monkeypatch, tmp_path):
    commit = "a" * 40
    identity_file = tmp_path / "build-commit"
    identity_file.write_text(f"{commit}\n", encoding="utf-8")
    monkeypatch.setattr(build_identity, "BUILD_COMMIT_PATH", identity_file)

    health_response = client.get("/health")
    source_response = client.get("/source")

    expected_tree = f"{build_identity.REPOSITORY_URL}/tree/{commit}"
    expected_archive = f"{build_identity.REPOSITORY_URL}/archive/{commit}.tar.gz"
    assert health_response.json()["build_commit"] == commit
    assert health_response.json()["source_code_url"] == expected_tree
    assert health_response.headers["X-Source-Code"] == expected_tree
    assert source_response.json()["source_code_url"] == expected_tree
    assert source_response.json()["source_archive_url"] == expected_archive
    assert source_response.json()["license_url"] == (
        f"{build_identity.REPOSITORY_URL}/blob/{commit}/LICENSE"
    )
    assert source_response.json()["third_party_source_manifest_url"] == (
        f"{build_identity.REPOSITORY_URL}/blob/{commit}/third-party-sources.json"
    )


def test_pymupdf_source_manifest_matches_runtime_requirement():
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    manifest = json.loads(Path("third-party-sources.json").read_text(encoding="utf-8"))
    components = {item["name"]: item for item in manifest["components"]}

    pymupdf = components["PyMuPDF"]
    assert f'PyMuPDF=={pymupdf["version"]}' in requirements
    assert pymupdf["version"] in pymupdf["source_url"]
    assert len(pymupdf["sha256"]) == 64
    assert components["MuPDF"]["version"] == pymupdf["version"]
    assert len(components["MuPDF"]["sha256"]) == 64
