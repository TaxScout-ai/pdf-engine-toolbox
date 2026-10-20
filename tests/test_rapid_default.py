"""RapidOCR packaging, fail-closed startup, and no Paddle rollback (TAX-6995)."""
import hashlib
import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from app.config import Settings, settings
from app.services import rapid_ocr, rapid_ocr_models as models


def test_a_leftover_paddle_pin_neither_selects_paddle_nor_stops_startup(
    monkeypatch, tmp_path
):
    """OCR_BACKEND=paddle from an older deployment is ignored, not fatal."""
    monkeypatch.delenv("WORKERS", raising=False)
    monkeypatch.setenv("OCR_BACKEND", "paddle")
    env_file = tmp_path / ".env"
    env_file.write_text("OCR_BACKEND=paddle\nOCR_DET_LIMIT_SIDE_LEN=1280\n")
    loaded = Settings(_env_file=str(env_file))
    assert not hasattr(loaded, "ocr_backend")
    # One process owns the OCR job queue (TAX-6876).
    with pytest.raises(ValueError, match="WORKERS=1"):
        Settings(workers=2, _env_file=None)


def test_no_paddle_distribution_is_required():
    requirements = Path(__file__).resolve().parents[1].joinpath("requirements.txt").read_text()
    names = [line.split("=")[0].split("[")[0].strip().lower()
             for line in requirements.splitlines() if line and not line.startswith("#")]
    assert not [name for name in names if "paddle" in name]
    assert "rapid-table" in names


def test_download_verifies_and_reuses_weights(tmp_path, monkeypatch):
    expected = hashlib.sha256(b"model").hexdigest()
    monkeypatch.setattr(models, "MODEL_HASHES", {"model.onnx": expected})
    monkeypatch.setattr(models, "MODEL_URLS", {"model.onnx": "https://example.com/model"})
    fetch = MagicMock(side_effect=lambda *a, **k: io.BytesIO(b"model"))
    monkeypatch.setattr(models, "urlopen", fetch)
    monkeypatch.setattr(models, "TABLE_MODEL_HASHES", {})
    models.download_models(tmp_path)
    models.download_models(tmp_path)
    fetch.assert_called_once()
    models.validate_models(tmp_path)


def test_corrupt_download_never_becomes_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "MODEL_HASHES", {"model.onnx": "bad"})
    monkeypatch.setattr(models, "TABLE_MODEL_HASHES", {})
    monkeypatch.setattr(models, "MODEL_URLS", {"model.onnx": "https://example.com/model"})
    monkeypatch.setattr(models, "urlopen", lambda *a, **k: io.BytesIO(b"wrong"))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        models.download_models(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_startup_rejects_missing_weights_before_serving(monkeypatch, tmp_path):
    from app.main import app, lifespan
    monkeypatch.setattr(settings, "rapidocr_model_dir", str(tmp_path))
    monkeypatch.setattr(rapid_ocr.importlib.metadata, "version", lambda n: rapid_ocr.PACKAGES[n])
    with pytest.raises(RuntimeError, match="missing or hash mismatch"):
        async with lifespan(app):
            pytest.fail("missing models must not report a healthy engine")


def test_table_model_is_pinned_and_downloaded_with_the_ocr_weights(tmp_path, monkeypatch):
    payloads = {"ocr.onnx": b"ocr", models.TABLE_MODEL: b"table"}
    monkeypatch.setattr(models, "MODEL_HASHES",
                        {"ocr.onnx": hashlib.sha256(b"ocr").hexdigest()})
    monkeypatch.setattr(models, "MODEL_URLS", {"ocr.onnx": "https://example.com/ocr.onnx"})
    monkeypatch.setattr(models, "TABLE_MODEL_HASHES",
                        {models.TABLE_MODEL: hashlib.sha256(b"table").hexdigest()})
    monkeypatch.setattr(models, "TABLE_MODEL_URLS",
                        {models.TABLE_MODEL: "https://example.com/table.onnx"})
    fetched = []

    def fetch(url, timeout):
        fetched.append(url)
        return io.BytesIO(payloads["ocr.onnx" if url.endswith("ocr.onnx") else models.TABLE_MODEL])

    monkeypatch.setattr(models, "urlopen", fetch)
    models.download_models(tmp_path)
    assert sorted(fetched) == ["https://example.com/ocr.onnx", "https://example.com/table.onnx"]
    (tmp_path / models.TABLE_MODEL).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match=f"hash mismatch: {models.TABLE_MODEL}"):
        models.validate_models(tmp_path)


def test_the_real_table_model_hash_is_pinned():
    assert models.TABLE_MODEL == "slanet-plus.onnx"
    assert models.TABLE_MODEL_HASHES[models.TABLE_MODEL] == (
        "d57a942af6a2f57d6a4a0372573c696a2379bf5857c45e2ac69993f3b334514b"
    )
    assert "/RapidTable/resolve/v2.0.0/" in models.TABLE_MODEL_URLS[models.TABLE_MODEL]
