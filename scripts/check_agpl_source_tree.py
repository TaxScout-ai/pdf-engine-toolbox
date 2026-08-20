#!/usr/bin/env python3
"""Fail closed when the public source tree cannot support the AGPL offer."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    notice = (ROOT / "NOTICE.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    app_main = (ROOT / "app/main.py").read_text(encoding="utf-8")
    source_route = (ROOT / "app/routes/source.py").read_text(encoding="utf-8")
    manifest = json.loads(
        (ROOT / "third-party-sources.json").read_text(encoding="utf-8")
    )

    require("GNU AFFERO GENERAL PUBLIC LICENSE" in license_text, "LICENSE is not AGPL")
    require("PyMuPDF" in notice and "MuPDF" in notice, "third-party notice is incomplete")
    require('license = "AGPL-3.0-or-later"' in pyproject, "pyproject license drift")
    require("GET /source" in readme, "README omits the network source offer")
    require('response.headers["X-Source-Code"]' in app_main, "source header is missing")
    require('rel="source"' in app_main, "Link rel=source is missing")
    require('@router.get("/source"' in source_route, "public source route is missing")
    require("ARG SOURCE_COMMIT" in dockerfile, "image is not bound to a source revision")
    require("github.com/TaxScout-ai/pdf-engine-toolbox/tree/$SOURCE_COMMIT" in dockerfile,
            "Docker build does not verify that SOURCE_COMMIT is public")
    require("org.opencontainers.image.source" in dockerfile, "OCI source label is missing")
    require("org.opencontainers.image.revision" in dockerfile, "OCI revision label is missing")

    components = manifest.get("components")
    require(isinstance(components, list) and components, "source manifest has no components")
    by_name = {component.get("name"): component for component in components}
    require({"PyMuPDF", "MuPDF"}.issubset(by_name), "PyMuPDF/MuPDF source is missing")

    pymupdf = by_name["PyMuPDF"]
    require(
        f'PyMuPDF=={pymupdf["version"]}' in requirements,
        "PyMuPDF requirement and source manifest differ",
    )
    require(
        by_name["MuPDF"]["version"] == pymupdf["version"],
        "MuPDF source version does not match the PyMuPDF build dependency",
    )
    require(
        pymupdf.get("license") == "AGPL-3.0-only",
        "PyMuPDF license must conservatively match its 1.27.1 release metadata",
    )
    require(
        by_name["MuPDF"].get("license") == "AGPL-3.0-or-later",
        "MuPDF license must preserve its explicit or-later grant",
    )
    for component in components:
        require(str(component.get("source_url", "")).startswith("https://"),
                f'{component.get("name")} source URL is not HTTPS')
        require(bool(SHA256.fullmatch(str(component.get("sha256", "")))),
                f'{component.get("name")} source SHA-256 is invalid')

    print("AGPL source-tree contract: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"AGPL source-tree contract: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
