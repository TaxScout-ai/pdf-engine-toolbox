"""Immutable build identity and AGPL Corresponding Source links."""

import json
import re
from pathlib import Path
from typing import Any

from app.config import settings

REPOSITORY_URL = "https://github.com/TaxScout-ai/pdf-engine-toolbox"
PROJECT_LICENSE_IDENTIFIER = "AGPL-3.0-or-later"
RUNTIME_LICENSE_IDENTIFIER = "AGPL-3.0-only"
# Compatibility alias for response/source-link code. The running combined work
# includes PyMuPDF, whose 1.27.1 release metadata does not state "or later".
LICENSE_IDENTIFIER = RUNTIME_LICENSE_IDENTIFIER
LICENSE_URL = f"{REPOSITORY_URL}/blob/main/LICENSE"
BUILD_COMMIT_PATH = Path("/app/build-commit")
THIRD_PARTY_SOURCES_PATH = Path("/app/third-party-sources.json")
_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")

# Sentinels returned instead of a revision. Named so callers can recognise an
# unverified build without re-deriving the rule.
DEVELOPMENT_REVISION = "development"
INVALID_REVISION = "invalid"
UNVERIFIED_REVISIONS = (DEVELOPMENT_REVISION, INVALID_REVISION)


class BuildIdentityError(RuntimeError):
    """The running build cannot prove which source revision it is."""


def verify_build_identity() -> str:
    """Return the build revision, refusing to run when it must be provable.

    The URL helpers below fall back to the mutable `main` branch when the
    revision is unknown. That fallback is honest for a local checkout and wrong
    for a shipped artifact: it would offer a source tree that is neither
    immutable nor necessarily the running code, exactly when the running code
    cannot be identified. Images set require_build_identity so this fails fast.
    """
    revision = read_build_commit()
    if _FULL_GIT_SHA.fullmatch(revision):
        return revision
    if settings.require_build_identity:
        raise BuildIdentityError(
            f"build identity is required but {BUILD_COMMIT_PATH} is "
            f"{'malformed' if revision == INVALID_REVISION else 'absent'}; refusing to "
            "serve, because the AGPL source offer would point at the mutable "
            "main branch rather than at the running code"
        )
    return revision


def read_build_commit() -> str:
    """Return the immutable source revision embedded in the container image.

    Development and unit-test processes may not have an image identity file.
    Production images cannot reach this state because the Docker build rejects
    an absent or malformed SOURCE_COMMIT.
    """
    try:
        commit = BUILD_COMMIT_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return DEVELOPMENT_REVISION

    if not _FULL_GIT_SHA.fullmatch(commit):
        return INVALID_REVISION
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
