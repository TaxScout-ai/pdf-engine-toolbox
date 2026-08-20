"""Tests for build-context to public-source binding."""

from pathlib import Path

import pytest

from scripts.verify_source_revision import compare_sources


def _write_source_tree(root: Path, *, app_value: str = "same") -> None:
    for relative in (
        "Dockerfile",
        "LICENSE",
        "NOTICE.md",
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        "third-party-sources.json",
        "scripts/download_models.py",
        "scripts/verify_source_revision.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    (root / "app").mkdir()
    (root / "app/main.py").write_text(app_value, encoding="utf-8")


def test_compare_sources_accepts_identical_runtime_inputs(tmp_path):
    local = tmp_path / "local"
    public = tmp_path / "public"
    _write_source_tree(local)
    _write_source_tree(public)

    compare_sources(local, public)


def test_compare_sources_rejects_code_claiming_another_revision(tmp_path):
    local = tmp_path / "local"
    public = tmp_path / "public"
    _write_source_tree(local, app_value="modified runtime")
    _write_source_tree(public, app_value="public runtime")

    with pytest.raises(ValueError, match="app/main.py"):
        compare_sources(local, public)
