#!/usr/bin/env python3
"""Bind runtime/build inputs to the exact public source revision they advertise."""

import argparse
import hashlib
import re
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY_URL = "https://github.com/TaxScout-ai/pdf-engine-toolbox"
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
            "source input set differs from public revision "
            f"(missing={missing}, extra={extra})"
        )
    changed = [path for path in local if local[path] != public[path]]
    if changed:
        raise ValueError(
            "source inputs differ from advertised public revision: "
            + ", ".join(changed)
        )


def download_public_source(commit: str, destination: Path) -> Path:
    archive_url = f"{REPOSITORY_URL}/archive/{commit}.tar.gz"
    request = Request(
        archive_url,
        headers={"User-Agent": "TaxScout-Source-Revision-Verifier/1.0"},
    )
    archive_path = destination / "source.tar.gz"
    with urlopen(request, timeout=60) as response, archive_path.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)

    extract_root = destination / "public"
    extract_root.mkdir()
    with tarfile.open(archive_path, mode="r:gz") as archive:
        archive.extractall(extract_root, filter="data")
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
