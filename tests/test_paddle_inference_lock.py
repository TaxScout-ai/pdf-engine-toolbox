"""TAX-6255 / TAX-6876: at most one Paddle inference runs in the process at a time.

Carried over from draft PR #15, which this change supersedes. With RapidOCR as
the OCR backend Paddle serves only tables and orientation; under the
OCR_BACKEND=paddle rollback, OCR jobs from the queue's parallel workers reach
Paddle too, and the lock is what keeps them off the predictor together.

Paddle predictors are not thread-safe and every model is a cached singleton.
On 2026-10-10 a second OCR task started while another was mid-scan; both
threads called ``.predict()`` on the same predictor, Paddle segfaulted and
the container restarted, losing both in-memory tasks.

The fakes below stand in for the Paddle models and record how many threads
are inside "inference" at once. No real model is loaded.
"""

import sys
import threading
import time
import types

import fitz

from app.config import settings
from app.services import ocr_queue, pdf_service

INFERENCE_SECONDS = 0.05


class _Probe:
    """Counts threads that are inside a fake inference at the same moment."""

    def __init__(self):
        self._guard = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls: list[str] = []

    def infer(self, label: str = ""):
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append(label or threading.current_thread().name)
        time.sleep(INFERENCE_SECONDS)
        with self._guard:
            self.active -= 1


class _OcrResult:
    json = {
        "res": {
            "rec_texts": ["Form W-2"],
            "rec_scores": [0.97],
            "dt_polys": [[[50.0, 50.0], [200.0, 50.0], [200.0, 75.0], [50.0, 75.0]]],
        }
    }


class _FakeOcr:
    """PaddleOCR-shaped predictor that returns an eager list."""

    def __init__(self, probe: _Probe):
        self.probe = probe

    def predict(self, input):  # noqa: A002 - mirrors the PaddleOCR keyword
        self.probe.infer()
        return [_OcrResult()]


class _LazyFakeOcr(_FakeOcr):
    """Predictor whose inference only runs when the result is consumed.

    This is what a PaddleX model's ``predict`` does: it returns a generator.
    """

    def predict(self, input):  # noqa: A002
        def results():
            self.probe.infer()
            yield _OcrResult()

        return results()


class _FakeStructure:
    def __init__(self, probe: _Probe):
        self.probe = probe

    def predict(self, input):  # noqa: A002
        self.probe.infer("tables")
        return []


class _OrientationPrediction:
    json = {"res": {"label_names": ["0"], "scores": [0.99]}}


class _FakeOrientationModel:
    def __init__(self, probe: _Probe):
        self.probe = probe

    def predict(self, images, batch_size):
        def results():
            self.probe.infer("orientation")
            for _ in images:
                yield _OrientationPrediction()

        return results()


def _scanned_pdf(page_count: int = 1) -> bytes:
    """Pages with no text layer, so every page goes to OCR."""
    doc = fitz.open()
    for _ in range(page_count):
        page = doc.new_page(width=612, height=792)
        page.draw_rect(
            fitz.Rect(50, 50, 200, 80), color=(0, 0, 0), fill=(0.9, 0.9, 0.9)
        )
    return doc.tobytes()


def _run_together(*targets):
    """Start every target at the same instant and re-raise the first error."""
    start = threading.Barrier(len(targets))
    errors: list[BaseException] = []

    def runner(target):
        try:
            start.wait(timeout=10)
            target()
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [
        threading.Thread(target=runner, args=(t,), name=f"task-{i}")
        for i, t in enumerate(targets)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
        assert not thread.is_alive(), "worker did not finish"
    if errors:
        raise errors[0]


def test_two_ocr_tasks_never_infer_at_the_same_time(monkeypatch):
    """The production crash: two queued Paddle OCR jobs, one shared predictor."""
    probe = _Probe()
    predictor = _FakeOcr(probe)
    monkeypatch.setattr(settings, "ocr_backend", "paddle")
    monkeypatch.setattr(pdf_service, "_get_paddle_ocr", lambda lang="en": predictor)
    monkeypatch.setattr(
        "app.services.cache_service.put_cached", lambda *a, **k: None
    )
    queue = ocr_queue.get_queue()
    assert queue.workers > 1  # both jobs really run on separate workers
    ids = [
        queue.submit_ocr(_scanned_pdf(3), None, "en", 150, "paddle") for _ in range(2)
    ]

    for task_id in ids:
        stored = queue.wait(task_id)
        assert stored["status"] == "completed", stored.get("error")
        assert stored["result"]["total_words"] == 3
    assert len(probe.calls) == 6
    assert probe.max_active == 1
    assert queue.max_running == 2


def test_lazy_results_are_consumed_while_the_lock_is_held(monkeypatch):
    """A generator runs its inference on iteration, not on ``predict()``."""
    probe = _Probe()
    predictor = _LazyFakeOcr(probe)
    monkeypatch.setattr(pdf_service, "_get_paddle_ocr", lambda lang="en": predictor)
    pdf_bytes = _scanned_pdf(2)
    results: list[dict] = []

    def ocr():
        results.append(pdf_service.ocr_pages(pdf_bytes, dpi=72))

    _run_together(ocr, ocr)

    assert len(probe.calls) == 4
    assert probe.max_active == 1
    assert [r["total_words"] for r in results] == [2, 2]


def test_ocr_tables_and_orientation_share_one_lock(monkeypatch):
    """A table or orientation request must not infer during an OCR page."""
    probe = _Probe()
    ocr = _FakeOcr(probe)
    structure = _FakeStructure(probe)
    orientation = _FakeOrientationModel(probe)
    monkeypatch.setattr(pdf_service, "_get_paddle_ocr", lambda lang="en": ocr)
    monkeypatch.setattr(pdf_service, "_get_pp_structure", lambda lang="en": structure)
    monkeypatch.setattr(pdf_service, "_get_doc_orientation_model", lambda: orientation)
    pdf_bytes = _scanned_pdf(3)

    _run_together(
        lambda: pdf_service.ocr_pages(pdf_bytes, dpi=72),
        lambda: pdf_service.extract_tables(pdf_bytes, strategy="ppstructure", dpi=72),
        lambda: pdf_service.detect_page_orientations(pdf_bytes),
    )

    assert probe.calls.count("tables") == 3
    assert probe.calls.count("orientation") == 1
    assert len(probe.calls) == 7
    assert probe.max_active == 1


def test_tasks_interleave_pages_instead_of_queueing_whole_documents(monkeypatch):
    """The lock is per page: a second task starts before the first finishes."""
    probe = _Probe()
    predictor = _FakeOcr(probe)
    monkeypatch.setattr(pdf_service, "_get_paddle_ocr", lambda lang="en": predictor)
    pdf_bytes = _scanned_pdf(6)

    _run_together(
        lambda: pdf_service.ocr_pages(pdf_bytes, dpi=72),
        lambda: pdf_service.ocr_pages(pdf_bytes, dpi=72),
    )

    assert probe.max_active == 1
    assert len(probe.calls) == 12
    # Whole-document locking would give six calls of one task, then six of
    # the other.
    assert len(set(probe.calls[:6])) == 2, probe.calls


def test_a_model_is_built_once_and_never_during_an_inference(monkeypatch):
    """Two cold requests must not both load the model, nor load it mid-predict."""
    probe = _Probe()
    built: list[str] = []

    class FakePaddleOCR(_FakeOcr):
        def __init__(self, **kwargs):
            super().__init__(probe)
            probe.infer("construct")
            built.append(kwargs["lang"])

    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)
    monkeypatch.setattr(pdf_service, "_paddle_ocr_instances", {})
    pdf_bytes = _scanned_pdf(1)

    _run_together(
        lambda: pdf_service.ocr_pages(pdf_bytes, dpi=72),
        lambda: pdf_service.ocr_pages(pdf_bytes, dpi=72),
        # A different language builds a second model while "en" is in use.
        lambda: pdf_service.ocr_pages(pdf_bytes, language="fr", dpi=72),
    )

    assert sorted(built) == ["en", "fr"]
    assert probe.max_active == 1


def test_concurrent_table_and_orientation_requests_never_overlap(
    client, auth_headers, monkeypatch
):
    """Two table requests and an orientation request at once, over HTTP."""
    import json
    from unittest.mock import AsyncMock

    from app.services import download_service

    probe = _Probe()
    structure = _FakeStructure(probe)
    orientation = _FakeOrientationModel(probe)
    monkeypatch.setattr(pdf_service, "_get_pp_structure", lambda lang="en": structure)
    monkeypatch.setattr(pdf_service, "_get_doc_orientation_model", lambda: orientation)
    monkeypatch.setattr(
        download_service, "download_pdf", AsyncMock(return_value=_scanned_pdf(2))
    )

    def post(path, payload):
        body = json.dumps(payload)
        response = client.post(path, content=body, headers=auth_headers("POST", path, body))
        assert response.status_code == 200, response.text

    tables = {"source_url": "https://example.com/t.pdf", "strategy": "ppstructure"}
    _run_together(
        lambda: post("/text/tables", tables),
        lambda: post("/text/tables", tables),
        lambda: post("/transform/orientation", {"source_url": "https://example.com/t.pdf"}),
    )

    assert probe.calls.count("tables") == 4
    assert probe.calls.count("orientation") == 1
    assert probe.max_active == 1
