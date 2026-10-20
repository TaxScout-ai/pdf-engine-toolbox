"""Pinned, build-time OCR weights; no document data is sent to these URLs."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

MODEL_HASHES = {
    "PP-OCRv6_det_small.onnx": "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
    "PP-OCRv6_rec_small.onnx": "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
    "ch_ppocr_mobile_v2.0_cls_mobile.onnx": "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",  # noqa: E501
}

# Table structure for /text/tables (TAX-6995). Kept apart from MODEL_HASHES,
# which is the OCR provenance every /text/ocr result carries.
TABLE_MODEL = "slanet-plus.onnx"
TABLE_MODEL_HASHES = {
    TABLE_MODEL: "d57a942af6a2f57d6a4a0372573c696a2379bf5857c45e2ac69993f3b334514b",
}

MODEL_BASE = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/"
MODEL_URLS = {
    "PP-OCRv6_det_small.onnx": MODEL_BASE + "PP-OCRv6/det/PP-OCRv6_det_small.onnx",
    "PP-OCRv6_rec_small.onnx": MODEL_BASE + "PP-OCRv6/rec/PP-OCRv6_rec_small.onnx",
    "ch_ppocr_mobile_v2.0_cls_mobile.onnx": (
        MODEL_BASE + "PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx"
    ),
}
TABLE_MODEL_URLS = {
    TABLE_MODEL: "https://www.modelscope.cn/models/RapidAI/RapidTable/resolve/v2.0.0/slanet-plus.onnx",
}
MAX_MODEL_BYTES = 100 * 1024 * 1024


def valid_model(path: Path, expected: str) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() == expected


def _required() -> dict[str, tuple[str, str]]:
    """Every model the image must hold: name -> (sha256, build-time URL)."""
    return {
        **{name: (digest, MODEL_URLS[name]) for name, digest in MODEL_HASHES.items()},
        **{name: (digest, TABLE_MODEL_URLS[name]) for name, digest in TABLE_MODEL_HASHES.items()},
    }


def validate_models(root: Path) -> None:
    for name, (expected, _) in _required().items():
        if not valid_model(root / name, expected):
            raise RuntimeError(f"RapidOCR model missing or hash mismatch: {name}")


def download_models(root: Path) -> None:
    """Build only: verify before an atomic rename; failure must fail the build."""
    root.mkdir(parents=True, exist_ok=True)
    for name, (expected, url) in _required().items():
        target = root / name
        if valid_model(target, expected):
            continue
        temporary = target.with_suffix(".download")
        try:
            size = 0
            with urlopen(url, timeout=120) as response, temporary.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_MODEL_BYTES:
                        raise RuntimeError("RapidOCR model download exceeds size limit")
                    out.write(chunk)
            if not valid_model(temporary, expected):
                raise RuntimeError(f"RapidOCR downloaded model hash mismatch: {name}")
            temporary.chmod(0o644)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    validate_models(root)
