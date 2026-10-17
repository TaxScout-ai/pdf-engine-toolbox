"""Primary backend packaging, fail-closed startup and explicit rollback."""
import hashlib
import io
from unittest.mock import MagicMock

import pytest
from app.config import Settings, settings
from app.services import rapid_ocr, rapid_ocr_models as models


def test_default_and_explicit_rollback(monkeypatch):
    monkeypatch.delenv("OCR_BACKEND", raising=False)
    monkeypatch.delenv("WORKERS", raising=False)
    assert Settings(_env_file=None).ocr_backend == "rapid"
    assert Settings(ocr_backend="paddle", _env_file=None).ocr_backend == "paddle"
    # One process owns the OCR job queue on either backend (TAX-6876).
    for backend in ("rapid", "paddle"):
        with pytest.raises(ValueError, match="WORKERS=1"):
            Settings(ocr_backend=backend, workers=2, _env_file=None)


def test_download_verifies_and_reuses_weights(tmp_path, monkeypatch):
    expected = hashlib.sha256(b"model").hexdigest()
    monkeypatch.setattr(models, "MODEL_HASHES", {"model.onnx": expected})
    monkeypatch.setattr(models, "MODEL_URLS", {"model.onnx": "https://example.com/model"})
    fetch = MagicMock(side_effect=lambda *a, **k: io.BytesIO(b"model"))
    monkeypatch.setattr(models, "urlopen", fetch)
    models.download_models(tmp_path)
    models.download_models(tmp_path)
    fetch.assert_called_once()
    models.validate_models(tmp_path)


def test_corrupt_download_never_becomes_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "MODEL_HASHES", {"model.onnx": "bad"})
    monkeypatch.setattr(models, "MODEL_URLS", {"model.onnx": "https://example.com/model"})
    monkeypatch.setattr(models, "urlopen", lambda *a, **k: io.BytesIO(b"wrong"))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        models.download_models(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_startup_rejects_missing_weights_before_serving(monkeypatch, tmp_path):
    from app.main import app, lifespan
    monkeypatch.setattr(settings, "ocr_backend", "rapid")
    monkeypatch.setattr(settings, "rapidocr_model_dir", str(tmp_path))
    monkeypatch.setattr(rapid_ocr.importlib.metadata, "version", lambda n: rapid_ocr.PACKAGES[n])
    with pytest.raises(RuntimeError, match="missing or hash mismatch"):
        async with lifespan(app):
            pytest.fail("missing models must not report a healthy engine")
