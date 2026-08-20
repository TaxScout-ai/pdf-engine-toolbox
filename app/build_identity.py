"""Immutable build identity and AGPL Corresponding Source links."""

import json
import re
from pathlib import Path
from typing import Any

REPOSITORY_URL = "https://github.com/TaxScout-ai/pdf-engine-toolbox"
LICENSE_IDENTIFIER = "AGPL-3.0-or-later"
LICENSE_URL = f"{REPOSITORY_URL}/blob/main/LICENSE"
BUILD_COMMIT_PATH = Path("/app/build-commit")
THIRD_PARTY_SOURCES_PATH = Path("/app/third-party-sources.json")
_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def read_build_commit() -> str:
    """Return the immutable source revision embedded in the container image.

    Development and unit-test processes may not have an image identity file.
    Production images cannot reach this state because the Docker build rejects
    an absent or malformed SOURCE_COMMIT.
    """
    try:
        commit = BUILD_COMMIT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return "development"

    if not _FULL_GIT_SHA.fullmatch(commit):
        return "invalid"
    return commit


def corresponding_source_url(commit: str | None = None) -> str:
    """Return the public source tree for this exact build when available."""
    revision = commit or read_build_commit()
    if _FULL_GIT_SHA.fullmatch(revision):
        return f"{REPOSITORY_URL}/tree/{revision}"
    return REPOSITORY_URL


def corresponding_source_archive_url(commit: str | None = None) -> str:
    """Return a directly downloadable archive for this exact build."""
    revision = commit or read_build_commit()
    if _FULL_GIT_SHA.fullmatch(revision):
        return f"{REPOSITORY_URL}/archive/{revision}.tar.gz"
    return f"{REPOSITORY_URL}/archive/refs/heads/main.tar.gz"


def corresponding_license_url(commit: str | None = None) -> str:
    """Return the license notice from the same revision as the running code."""
    revision = commit or read_build_commit()
    if _FULL_GIT_SHA.fullmatch(revision):
        return f"{REPOSITORY_URL}/blob/{revision}/LICENSE"
    return LICENSE_URL


def third_party_source_manifest_url(commit: str | None = None) -> str:
    """Return the third-party source manifest from the running revision."""
    revision = commit or read_build_commit()
    if _FULL_GIT_SHA.fullmatch(revision):
        return f"{REPOSITORY_URL}/blob/{revision}/third-party-sources.json"
    return f"{REPOSITORY_URL}/blob/main/third-party-sources.json"


def read_third_party_sources() -> list[dict[str, Any]]:
    """Read the bundled, hash-pinned third-party Corresponding Source list."""
    path = THIRD_PARTY_SOURCES_PATH
    if not path.is_file():
        path = Path(__file__).resolve().parent.parent / "third-party-sources.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    components = payload.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("third-party source manifest has no components")
    return components
