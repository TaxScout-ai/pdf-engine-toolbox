"""TAX-6876: persistent, fair OCR job queue with bounded parallel workers.

A fake RapidOCR engine stands in for the model. It reads the page index from
the rendered image width (page ``i`` is ``300 + 10 * i`` points wide, rendered
at 216 DPI = 3 px/pt) and counts how many inferences run at the same moment.
No real model is loaded.
"""

import base64
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import fitz
import numpy as np
import pytest

from app.config import settings
from app.services import download_service, ocr_queue, rapid_ocr

POLYGON = [[30, 60], [330, 60], [330, 120], [30, 120]]
# POLYGON in visual PDF points at 3 px/pt, on every page.
EXPECTED_BBOX = {"x": 10, "y": 20, "w": 100, "h": 20}


def page_width(index: int) -> int:
    return 300 + 10 * index


def scanned(pages: int, tag: str = "") -> bytes:
    """Image-free pages without a text layer, so every page goes to the engine."""
    with fitz.open() as doc:
        for index in range(pages):
            doc.new_page(width=page_width(index), height=400)
        doc.set_metadata({"title": tag})  # distinct bytes, distinct cache key
        return doc.tobytes()


class FakeEngine:
    """Thread-safe stand-in for a RapidOCR engine with concurrency counters."""

    def __init__(self, delay=0.02, gate: threading.Event | None = None, free_calls=0):
        self.delay = delay
        self.gate = gate
        self.free_calls = free_calls
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls: list[int] = []
        self.finished: list[int] = []

    def __call__(self, image, **_):
        page = round((image.shape[1] / 3 - 300) / 10)
        with self.lock:
            self.calls.append(page)
            blocked = self.gate is not None and len(self.calls) > self.free_calls
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if blocked:
                assert self.gate.wait(30), "test gate never opened"
            time.sleep(self.delay(page) if callable(self.delay) else self.delay)
        finally:
            with self.lock:
                self.active -= 1
                self.finished.append(page)
        polygon = np.array(POLYGON, dtype=float)
        text = f"p{page}"
        return SimpleNamespace(
            txts=(text,), boxes=np.array([polygon]), word_results=[[(text, 0.98, polygon)]]
        )


@pytest.fixture(autouse=True)
def rapid_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "ocr_backend", "rapid")
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path / "cache"))
    monkeypatch.setattr(settings, "ocr_workers", 4)


@pytest.fixture
def documents(monkeypatch):
    """source_url -> bytes served by the mocked downloader."""
    docs: dict[str, bytes] = {}
    monkeypatch.setattr(
        download_service, "download_pdf", AsyncMock(side_effect=lambda url: docs[url])
    )
    return docs


def submit(client, auth_headers, url: str):
    body = json.dumps({"source_url": url})
    return client.post("/text/ocr", content=body, headers=auth_headers("POST", "/text/ocr", body))


def get_task(client, auth_headers, task_id: str):
    path = f"/tasks/{task_id}"
    return client.get(path, headers=auth_headers("GET", path, ""))


def poll(client, auth_headers, task_id: str, timeout: float = 60) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = get_task(client, auth_headers, task_id).json()
        if not body["success"] or body["data"]["status"] in ("completed", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"task {task_id} did not finish")


def test_twenty_concurrent_submissions_all_complete_with_bounded_parallelism(
    client, auth_headers, documents, monkeypatch
):
    engine = FakeEngine(delay=0.03)
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: engine)
    for index in range(20):
        documents[f"https://example.com/{index}.pdf"] = scanned(2, tag=str(index))
    accepted: list[dict] = []

    def post(index):
        response = submit(client, auth_headers, f"https://example.com/{index}.pdf")
        assert response.status_code == 200, response.text
        accepted.append(response.json())

    threads = [threading.Thread(target=post, args=(i,)) for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)

    assert len(accepted) == 20
    results = [poll(client, auth_headers, a["task_id"]) for a in accepted]
    for result in results:
        assert result["success"] is True
        assert result["data"]["status"] == "completed", result["data"].get("error")
        assert [p["full_text"] for p in result["data"]["result"]["pages"]] == ["p0", "p1"]
    assert len(engine.calls) == 40
    # Never more inferences than workers, and the workers really run together.
    assert engine.max_active <= settings.ocr_workers
    assert engine.max_active >= 2


def test_queued_and_in_flight_jobs_resume_after_a_restart(
    client, auth_headers, documents, monkeypatch
):
    monkeypatch.setattr(settings, "ocr_workers", 2)
    gate = threading.Event()
    before = FakeEngine(delay=0, gate=gate, free_calls=2)
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: before)
    for index in range(5):
        documents[f"https://example.com/{index}.pdf"] = scanned(3, tag=str(index))
    ids = [
        submit(client, auth_headers, f"https://example.com/{i}.pdf").json()["task_id"]
        for i in range(5)
    ]
    deadline = time.monotonic() + 10
    while before.active < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert before.active == 2  # two pages in flight, two already read

    statuses = [get_task(client, auth_headers, i).json()["data"] for i in ids]
    assert {s["status"] for s in statuses} <= {"pending", "processing"}
    assert "processing" in {s["status"] for s in statuses}
    last = statuses[-1]
    assert last["status"] == "pending"
    assert last["queue_position"] == 4
    assert isinstance(last["eta_seconds"], int) and last["eta_seconds"] > 0

    # The process dies: nothing the stuck inferences produce is recorded.
    ocr_queue.shutdown()
    after = FakeEngine(delay=0)
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: after)
    gate.set()
    ocr_queue.get_queue()  # the next process starts and recovers the ledger

    for index, task_id in enumerate(ids):
        result = poll(client, auth_headers, task_id)
        assert result["data"]["status"] == "completed", result["data"].get("error")
        pages = result["data"]["result"]["pages"]
        assert [p["page_index"] for p in pages] == [0, 1, 2]
        assert [p["full_text"] for p in pages] == ["p0", "p1", "p2"]
    # The two pages read before the crash were checkpointed, not read again.
    assert len(after.calls) == 15 - 2


def test_one_document_is_spread_over_workers_in_page_order(
    client, auth_headers, documents, monkeypatch
):
    # Page 0 is slowest, so pages finish in reverse order.
    engine = FakeEngine(delay=lambda page: 0.02 * (8 - page))
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: engine)
    documents["https://example.com/scan.pdf"] = scanned(8)

    accepted = submit(client, auth_headers, "https://example.com/scan.pdf").json()
    result = poll(client, auth_headers, accepted["task_id"])["data"]["result"]

    assert engine.max_active >= 2  # one scan, several workers
    assert engine.finished != sorted(engine.finished)
    pages = result["pages"]
    assert [p["page_index"] for p in pages] == list(range(8))
    for index, page in enumerate(pages):
        assert page["full_text"] == f"p{index}"
        assert page["visual_width"] == page_width(index)
        assert page["words"][0]["text"] == f"p{index}"
        assert page["words"][0]["bbox"] == pytest.approx(EXPECTED_BBOX)
    with fitz.open(stream=base64.b64decode(result["pdf_base64"]), filetype="pdf") as doc:
        for index, page in enumerate(doc):
            assert page.get_text().strip() == f"p{index}"  # search layer on its own page


def test_short_document_is_not_starved_by_a_long_scan(
    client, auth_headers, documents, monkeypatch
):
    monkeypatch.setattr(settings, "ocr_workers", 2)
    engine = FakeEngine(delay=0.03)
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: engine)
    documents["https://example.com/long.pdf"] = scanned(30, tag="long")
    documents["https://example.com/short.pdf"] = scanned(1, tag="short")

    long_id = submit(client, auth_headers, "https://example.com/long.pdf").json()["task_id"]
    short_id = submit(client, auth_headers, "https://example.com/short.pdf").json()["task_id"]

    short = poll(client, auth_headers, short_id)["data"]
    assert short["status"] == "completed"
    assert get_task(client, auth_headers, long_id).json()["data"]["status"] == "processing"
    long = poll(client, auth_headers, long_id)["data"]
    assert long["status"] == "completed"
    assert short["completed_at"] < long["completed_at"]


def test_unknown_task_keeps_the_task_not_found_shape(client, auth_headers, documents, monkeypatch):
    expected = {
        "success": False,
        "error": {"code": "TASK_NOT_FOUND", "message": "Task no-such-task not found"},
    }
    response = get_task(client, auth_headers, "no-such-task")
    assert response.status_code == 200
    assert response.json() == expected

    # A finished result outlives a restart, then expires into the same shape.
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: FakeEngine(delay=0))
    documents["https://example.com/a.pdf"] = scanned(1)
    task_id = submit(client, auth_headers, "https://example.com/a.pdf").json()["task_id"]
    assert poll(client, auth_headers, task_id)["data"]["status"] == "completed"
    ocr_queue.shutdown()
    ocr_queue.get_queue()
    survived = get_task(client, auth_headers, task_id).json()
    assert survived["data"]["status"] == "completed"
    assert survived["data"]["result"]["pages"][0]["full_text"] == "p0"

    ocr_queue.shutdown()
    monkeypatch.setattr(settings, "ocr_result_ttl_seconds", 0)
    ocr_queue.get_queue()
    response = get_task(client, auth_headers, task_id)
    assert response.status_code == 200
    assert response.json() == {
        "success": False,
        "error": {"code": "TASK_NOT_FOUND", "message": f"Task {task_id} not found"},
    }
    assert not list((Path(settings.ocr_queue_dir) / "results").iterdir())


def test_full_queue_answers_429_with_retry_after(client, auth_headers, documents, monkeypatch):
    monkeypatch.setattr(settings, "ocr_queue_max_jobs", 2)
    gate = threading.Event()
    engine = FakeEngine(delay=0, gate=gate)
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: engine)
    for index in range(3):
        documents[f"https://example.com/{index}.pdf"] = scanned(1, tag=str(index))

    first = submit(client, auth_headers, "https://example.com/0.pdf")
    second = submit(client, auth_headers, "https://example.com/1.pdf")
    refused = submit(client, auth_headers, "https://example.com/2.pdf")

    assert first.status_code == second.status_code == 200
    assert refused.status_code == 429
    # RFC 9110 delay-seconds: a bare non-negative integer (the app parses it).
    retry_after = refused.headers["Retry-After"]
    assert retry_after.isdigit(), retry_after
    assert 5 <= int(retry_after) <= 300
    assert retry_after in refused.json()["error"]["message"]
    assert refused.json()["success"] is False
    assert refused.json()["error"]["code"] == "OCR_QUEUE_FULL"
    gate.set()
    for accepted in (first, second):
        task = poll(client, auth_headers, accepted.json()["task_id"])["data"]
        assert task["status"] == "completed"
    # Space again once the queue drains.
    assert submit(client, auth_headers, "https://example.com/2.pdf").status_code == 200


def test_each_worker_owns_its_engine(monkeypatch):
    """A RapidOCR object mutates per-call state; it must never be shared."""
    created: list["OwnedEngine"] = []
    guard = threading.Lock()

    class OwnedEngine(FakeEngine):
        def __init__(self):
            super().__init__(delay=0.02)
            self.threads: set[int] = set()

        def __call__(self, image, **kwargs):
            self.threads.add(threading.get_ident())
            return super().__call__(image, **kwargs)

    def new_engine():
        engine = OwnedEngine()
        with guard:
            created.append(engine)
        return engine

    monkeypatch.setattr(rapid_ocr, "_new_engine", new_engine)
    results: list[dict] = []
    threads = [
        threading.Thread(target=lambda: results.append(rapid_ocr.ocr_pages(scanned(4))))
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)

    assert len(results) == 3
    assert 1 < len(created) <= settings.ocr_workers
    for engine in created:
        assert len(engine.threads) == 1
        assert engine.max_active == 1
    assert sum(len(engine.calls) for engine in created) == 12


def test_identifier_redaction_reads_with_rapid_and_never_touches_paddle(monkeypatch):
    from app.services import identifier_redaction as ir
    from app.services import pdf_service

    def no_paddle(*_, **__):
        raise AssertionError("identifier redaction loaded Paddle")

    monkeypatch.setattr(pdf_service, "_get_paddle_ocr", no_paddle)
    calls = []

    def engine(image, **_):
        # The scan shows an SSN until it is redacted.
        calls.append(image.shape)
        text = "123-45-6789" if len(calls) == 1 else "XXX"
        polygon = np.array([[150, 150], [450, 150], [450, 180], [150, 180]], dtype=float)
        return SimpleNamespace(
            txts=(text,), boxes=np.array([polygon]), word_results=[[(text, 0.98, polygon)]]
        )

    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: engine)
    result = ir.redact_identifiers(scanned(1))
    assert result["redactions"] == [{"page": 0, "kind": "ssn", "last4": "6789", "source": "ocr"}]
    assert result["ocr_pages"] == [0]
    assert len(calls) == 2  # read, then re-read to prove nothing is left
    assert pdf_service._paddle_ocr_instances == {}


def test_identifier_redaction_does_not_import_paddle(tmp_path):
    """In a fresh interpreter the default redaction path never imports Paddle."""
    script = f"""
import sys
from types import SimpleNamespace
import fitz, numpy as np
from app.config import settings
settings.ocr_queue_dir = {str(tmp_path / 'queue')!r}
settings.ocr_workers = 2
from app.services import identifier_redaction as ir, rapid_ocr, ocr_queue
poly = np.array([[30, 60], [330, 60], [330, 120], [30, 120]], dtype=float)
rapid_ocr._get_engine = lambda: (lambda image, **_: SimpleNamespace(
    txts=("Wages",), boxes=np.array([poly]), word_results=[[("Wages", 0.98, poly)]]))
doc = fitz.open(); doc.new_page()
result = ir.redact_identifiers(doc.tobytes())
assert result["ocr_pages"] == [0], result
loaded = sorted(m for m in sys.modules if m.split('.')[0] in ('paddle', 'paddleocr', 'paddlex'))
print('PADDLE_MODULES', loaded)
ocr_queue.shutdown()
"""
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run(
        [sys.executable, "-c", script], cwd=root, capture_output=True, text=True, timeout=120
    )
    assert run.returncode == 0, run.stderr[-2000:]
    assert "PADDLE_MODULES []" in run.stdout, run.stdout[-2000:]


def test_paddle_rollback_runs_through_the_same_queue(client, auth_headers, documents, monkeypatch):
    from app.services import pdf_service

    monkeypatch.setattr(settings, "ocr_backend", "paddle")

    class Paddle:
        def predict(self, input):  # noqa: A002 - mirrors PaddleOCR
            return [SimpleNamespace(json={"res": {
                "rec_texts": ["Form W-2"], "rec_scores": [0.97],
                "dt_polys": [[[50.0, 50.0], [200.0, 50.0], [200.0, 75.0], [50.0, 75.0]]],
            }})]

    monkeypatch.setattr(pdf_service, "_get_paddle_ocr", lambda lang="en": Paddle())
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: pytest.fail("Rapid used"))
    documents["https://example.com/p.pdf"] = scanned(2)
    accepted = submit(client, auth_headers, "https://example.com/p.pdf").json()
    task = poll(client, auth_headers, accepted["task_id"])["data"]
    assert task["status"] == "completed", task.get("error")
    assert [p["full_text"] for p in task["result"]["pages"]] == ["Form W-2", "Form W-2"]
    stored = ocr_queue.get_queue().store.get(accepted["task_id"])
    assert stored["backend"] == "paddle"
