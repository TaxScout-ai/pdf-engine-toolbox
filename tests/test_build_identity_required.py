"""A shipped build must be able to name the revision it is running.

Without this, an absent or corrupt /app/build-commit makes the service offer
the mutable `main` branch as Corresponding Source - a source tree that is
neither immutable nor necessarily the running code.
"""

import pytest

from app import build_identity
from app.build_identity import BuildIdentityError, verify_build_identity


@pytest.fixture
def require_identity(monkeypatch):
    monkeypatch.setattr(build_identity.settings, "require_build_identity", True)


def test_required_and_absent_refuses_to_start(require_identity, monkeypatch, tmp_path):
    monkeypatch.setattr(build_identity, "BUILD_COMMIT_PATH", tmp_path / "missing")

    with pytest.raises(BuildIdentityError, match="absent"):
        verify_build_identity()


def test_required_and_malformed_refuses_to_start(require_identity, monkeypatch, tmp_path):
    identity = tmp_path / "build-commit"
    identity.write_text("not-a-sha\n", encoding="utf-8")
    monkeypatch.setattr(build_identity, "BUILD_COMMIT_PATH", identity)

    with pytest.raises(BuildIdentityError, match="malformed"):
        verify_build_identity()


def test_required_and_valid_returns_the_revision(require_identity, monkeypatch, tmp_path):
    commit = "b" * 40
    identity = tmp_path / "build-commit"
    identity.write_text(f"{commit}\n", encoding="utf-8")
    monkeypatch.setattr(build_identity, "BUILD_COMMIT_PATH", identity)

    assert verify_build_identity() == commit


def test_not_required_and_absent_keeps_development_behaviour(monkeypatch, tmp_path):
    """Local checkouts must keep working; the fallback is honest there."""
    monkeypatch.setattr(build_identity.settings, "require_build_identity", False)
    monkeypatch.setattr(build_identity, "BUILD_COMMIT_PATH", tmp_path / "missing")

    assert verify_build_identity() == build_identity.DEVELOPMENT_REVISION


def test_a_required_build_never_advertises_the_main_branch(require_identity, monkeypatch, tmp_path):
    """The regression this exists to prevent, stated as an assertion."""
    identity = tmp_path / "build-commit"
    identity.write_text("garbage", encoding="utf-8")
    monkeypatch.setattr(build_identity, "BUILD_COMMIT_PATH", identity)

    with pytest.raises(BuildIdentityError):
        verify_build_identity()

    # And the URL that would have been served is exactly the unacceptable one.
    assert build_identity.corresponding_source_archive_url() == (
        f"{build_identity.REPOSITORY_URL}/archive/refs/heads/main.tar.gz"
    )
