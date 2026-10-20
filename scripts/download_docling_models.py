"""Build-time-only, revision- and hash-pinned TableFormer/layout artifacts."""
import hashlib
import json
from pathlib import Path
import urllib.request

REVISIONS = {
    "docling-project--docling-layout-heron": "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8",
    "docling-project--docling-layout-heron-onnx": "40bde044036bb181c130ddf6c51792187268748f",
    "docling-project--docling-models": "fc0f2d45e2218ea24bce5045f58a389aed16dc23",
}


def download(root: Path, manifest: Path):
    for name, expected in json.loads(manifest.read_text()).items():
        repository, filename = name.split("/", 1)
        revision = REVISIONS[repository]
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{repository.replace('--', '/')}/resolve/{revision}/{filename}"
        with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as output:
            digest = hashlib.sha256()
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != expected:
            destination.unlink()
            raise RuntimeError(f"Docling artifact hash mismatch: {name}")


if __name__ == "__main__":
    download(Path("/opt/docling/models"), Path("/tmp/docling_model_hashes.json"))
