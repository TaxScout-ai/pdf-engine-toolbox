"""TAX-6876 checker round 1: the OCR queue under failure, at rest and at its edges."""
# Fixtures are imported from test_ocr_queue and requested by name.
# ruff: noqa: F811

import asyncio
import base64
import errno
import json
import os
import stat
import threading
import time
from pathlib import Path

import fitz
import pytest

from app.config import Settings, settings
from app.services import identifier_redaction as ir
from app.services import ocr_queue, rapid_ocr
from tests.test_ocr_queue import (  # noqa: F401 - fixtures
    FakeEngine,
    documents,
    get_task,
    poll,
    rapid_queue,
    scanned,
    submit,
)


def queue_files(kind: str) -> list[Path]:
    return sorted((Path(settings.ocr_queue_dir) / kind).iterdir())


def wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition not reached")


# 1 ------------------------------------------------------------ zero-unit jobs


def test_a_request_with_no_pages_finishes_and_frees_its_slot(
    client, auth_headers, documents, monkeypatch
):
    monkeypatch.setattr(settings, "ocr_max_active_jobs", 1)
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: FakeEngine(delay=0))
    documents["https://example.com/a.pdf"] = scanned(2, "a")
    body = json.dumps({"source_url": "https://example.com/a.pdf", "pages": []})
    empty = client.post("/text/ocr", content=body, headers=auth_headers("POST", "/text/ocr", body))
    task = poll(client, auth_headers, empty.json()["task_id"], timeout=10)["data"]
    assert task["status"] == "completed"
    assert task["result"]["pages"] == []
    documents["https://example.com/b.pdf"] = scanned(1, "b")
    after = submit(client, auth_headers, "https://example.com/b.pdf").json()["task_id"]
    assert poll(client, auth_headers, after, timeout=10)["data"]["status"] == "completed"


def test_restart_after_every_page_was_recorded_finishes_the_job(
    client, auth_headers, documents, monkeypatch
):
    """Crash between the last checkpoint and complete(): a real restart finishes it."""
    monkeypatch.setattr(settings, "ocr_max_active_jobs", 1)
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: FakeEngine(delay=0))
    queue = ocr_queue.get_queue()
    queue._finish = lambda job: None  # the process dies before completing
    documents["https://example.com/a.pdf"] = scanned(2, "a")
    task_id = submit(client, auth_headers, "https://example.com/a.pdf").json()["task_id"]
    wait_for(lambda: len(queue.store.units(task_id)) == 2)
    ocr_queue.shutdown()

    after = FakeEngine(delay=0)
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: after)
    ocr_queue.get_queue()  # new queue object on the same ledger and directory
    task = poll(client, auth_headers, task_id, timeout=10)["data"]
    assert task["status"] == "completed"
    assert [p["full_text"] for p in task["result"]["pages"]] == ["p0", "p1"]
    assert after.calls == []  # nothing was read again
    documents["https://example.com/b.pdf"] = scanned(1, "b")
    other = submit(client, auth_headers, "https://example.com/b.pdf").json()["task_id"]
    assert poll(client, auth_headers, other, timeout=10)["data"]["status"] == "completed"


# 2 ------------------------------------------------- failures after a unit ran


def test_disk_full_while_checkpointing_fails_the_job_and_keeps_workers(
    client, auth_headers, documents, monkeypatch
):
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: FakeEngine(delay=0))
    queue = ocr_queue.get_queue()
    real_save = queue.store.save_unit

    def full(*args):
        raise OSError(errno.ENOSPC, "No space left on device")

    queue.store.save_unit = full
    documents["https://example.com/a.pdf"] = scanned(2, "a")
    task_id = submit(client, auth_headers, "https://example.com/a.pdf").json()["task_id"]
    task = poll(client, auth_headers, task_id, timeout=10)["data"]
    assert task["status"] == "failed"
    assert "ENOSPC" in task["error"]
    assert all(thread.is_alive() for thread in queue._threads)

    queue.store.save_unit = real_save
    documents["https://example.com/b.pdf"] = scanned(1, "b")
    other = submit(client, auth_headers, "https://example.com/b.pdf").json()["task_id"]
    assert poll(client, auth_headers, other, timeout=10)["data"]["status"] == "completed"


def test_disk_full_while_writing_the_result_fails_the_job(
    client, auth_headers, documents, monkeypatch
):
    gate = threading.Event()
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: FakeEngine(delay=0, gate=gate))
    queue = ocr_queue.get_queue()
    documents["https://example.com/a.pdf"] = scanned(1, "a")
    task_id = submit(client, auth_headers, "https://example.com/a.pdf").json()["task_id"]

    def full(*args):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(ocr_queue, "_write_private", full)  # the disk fills up now
    gate.set()
    task = poll(client, auth_headers, task_id, timeout=10)["data"]
    assert task["status"] == "failed"
    assert "ENOSPC" in task["error"]
    assert all(thread.is_alive() for thread in queue._threads)


def test_a_failed_ledger_transaction_rolls_back_and_the_outcome_is_still_reported(
    client, auth_headers, documents, monkeypatch
):
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: FakeEngine(delay=0))
    queue = ocr_queue.get_queue()
    store = queue.store
    store._execute(
        "CREATE TRIGGER refuse BEFORE DELETE ON ocr_job_units"
        " BEGIN SELECT RAISE(ABORT, 'disk I/O error'); END"
    )
    documents["https://example.com/a.pdf"] = scanned(1, "a")
    task_id = submit(client, auth_headers, "https://example.com/a.pdf").json()["task_id"]
    task = poll(client, auth_headers, task_id, timeout=10)["data"]
    assert task["status"] == "failed"
    assert "disk I/O error" in task["error"]
    assert store._db.in_transaction is False  # no dangling BEGIN IMMEDIATE

    store._execute("DROP TRIGGER refuse")
    documents["https://example.com/b.pdf"] = scanned(1, "b")
    other = submit(client, auth_headers, "https://example.com/b.pdf").json()["task_id"]
    assert poll(client, auth_headers, other, timeout=10)["data"]["status"] == "completed"


def test_a_dead_worker_is_replaced(monkeypatch):
    queue = ocr_queue.get_queue()
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    queue._threads[0] = dead
    queue.sweep()
    assert queue.worker_restarts == 1
    assert all(thread.is_alive() for thread in queue._threads)


# 3 ---------------------------------------------------- retention and disk use


def test_a_late_checkpoint_cannot_orphan_a_row():
    queue = ocr_queue.get_queue()
    task_id = queue.submit_ocr(scanned(1), [], "en", 300, "rapid")
    assert queue.wait(task_id)["status"] == "completed"
    queue.store.save_unit(task_id, 0, {"full_text": "late"})  # after _finish
    assert queue.store._execute("SELECT COUNT(*) AS n FROM ocr_job_units")[0]["n"] == 0


def test_sweeper_removes_unreferenced_files_stale_temporaries_and_orphan_rows(
    client, auth_headers, documents, monkeypatch
):
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: FakeEngine(delay=0))
    queue = ocr_queue.get_queue()
    documents["https://example.com/a.pdf"] = scanned(1, "a")
    kept = submit(client, auth_headers, "https://example.com/a.pdf").json()["task_id"]
    poll(client, auth_headers, kept, timeout=10)
    root = Path(settings.ocr_queue_dir)
    old = time.time() - 3600
    stray = root / "inputs" / "stray.bin"
    stale = root / "results" / "half-written.bin.tmp"
    fresh = root / "results" / "being-written.bin.tmp"
    for path in (stray, stale, fresh):
        path.write_bytes(b"client data")
    for path in (stray, stale):
        os.utime(path, (old, old))
    queue.store._execute(
        "INSERT INTO ocr_job_units (job_id, unit, result) VALUES ('gone', 0, x'00')"
    )

    removed = queue.sweep()

    assert not stray.exists() and not stale.exists()
    assert fresh.exists()  # a write in progress is left alone
    assert removed["unit_rows"] == 1
    assert get_task(client, auth_headers, kept).json()["data"]["status"] == "completed"


def test_an_expired_result_reads_as_unknown_before_any_sweep(
    client, auth_headers, documents, monkeypatch
):
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: FakeEngine(delay=0))
    documents["https://example.com/a.pdf"] = scanned(1, "a")
    task_id = submit(client, auth_headers, "https://example.com/a.pdf").json()["task_id"]
    assert poll(client, auth_headers, task_id, timeout=10)["data"]["status"] == "completed"
    ocr_queue.get_queue().ttl = 0
    time.sleep(0.01)
    assert get_task(client, auth_headers, task_id).json() == {
        "success": False,
        "error": {"code": "TASK_NOT_FOUND", "message": f"Task {task_id} not found"},
    }


# 4 ---------------------------------------------------------------- PII at rest


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_inputs_results_and_checkpoints_are_sealed_and_private(
    client, auth_headers, documents, monkeypatch
):
    gate = threading.Event()
    engine = FakeEngine(delay=0, gate=gate, free_calls=1)
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: engine)
    queue = ocr_queue.get_queue()
    documents["https://example.com/a.pdf"] = scanned(2, "SECRET-TITLE")
    task_id = submit(client, auth_headers, "https://example.com/a.pdf").json()["task_id"]
    wait_for(lambda: len(queue.store.units(task_id)) == 1)

    root = Path(settings.ocr_queue_dir)
    [spilled] = queue_files("inputs")
    raw = spilled.read_bytes()
    assert b"%PDF" not in raw and b"SECRET-TITLE" not in raw
    blobs = queue.store._execute("SELECT result FROM ocr_job_units")
    assert blobs and all(b"p0" not in bytes(r["result"]) for r in blobs)
    for folder in (root, root / "inputs", root / "results"):
        assert mode(folder) == 0o700
    assert mode(spilled) == 0o600
    for suffix in ("", "-wal", "-shm"):
        db = Path(f"{root / 'jobs.sqlite3'}{suffix}")
        if db.exists():
            assert mode(db) == 0o600, db

    gate.set()
    task = poll(client, auth_headers, task_id, timeout=10)["data"]
    assert task["result"]["pages"][0]["full_text"] == "p0"
    assert queue_files("inputs") == []  # deleted as soon as the job finished
    [result] = queue_files("results")
    assert mode(result) == 0o600
    raw = result.read_bytes()
    assert b"p0" not in raw and b"pdf_base64" not in raw


def test_a_tampered_input_fails_its_job(client, auth_headers, documents, monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(settings, "ocr_max_active_jobs", 1)
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: FakeEngine(delay=0, gate=gate))
    documents["https://example.com/a.pdf"] = scanned(1, "a")
    documents["https://example.com/b.pdf"] = scanned(1, "b")
    first = submit(client, auth_headers, "https://example.com/a.pdf").json()["task_id"]
    second = submit(client, auth_headers, "https://example.com/b.pdf").json()["task_id"]
    path = Path(settings.ocr_queue_dir) / "inputs" / f"{second}.bin"
    blob = bytearray(path.read_bytes())
    blob[-1] ^= 1
    path.write_bytes(bytes(blob))
    gate.set()
    assert poll(client, auth_headers, first, timeout=10)["data"]["status"] == "completed"
    task = poll(client, auth_headers, second, timeout=10)["data"]
    assert task["status"] == "failed"
    assert "authentication" in task["error"]


def test_queue_key_comes_from_ocr_queue_key_or_the_engine_secret(monkeypatch):
    explicit = base64.urlsafe_b64encode(bytes(range(32))).decode()
    monkeypatch.setattr(settings, "ocr_queue_key", explicit)
    assert ocr_queue.queue_key() == bytes(range(32))
    monkeypatch.setattr(settings, "ocr_queue_key", "a passphrase")
    from_passphrase = ocr_queue.queue_key()
    monkeypatch.setattr(settings, "ocr_queue_key", "")
    derived = ocr_queue.queue_key()
    monkeypatch.setattr(settings, "pdf_engine_secret", "another-secret")
    assert len({from_passphrase, derived, ocr_queue.queue_key()}) == 3
    assert all(len(key) == 32 for key in (from_passphrase, derived))


def test_a_queued_job_under_a_changed_key_fails_closed(
    client, auth_headers, documents, monkeypatch
):
    gate = threading.Event()
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: FakeEngine(delay=0, gate=gate))
    documents["https://example.com/a.pdf"] = scanned(1, "a")
    task_id = submit(client, auth_headers, "https://example.com/a.pdf").json()["task_id"]
    wait_for(lambda: get_task(client, auth_headers, task_id).json()["data"]["status"]
             == "processing")
    ocr_queue.shutdown()
    gate.set()
    monkeypatch.setattr(settings, "ocr_queue_key", "rotated without migration")
    ocr_queue.get_queue()
    task = poll(client, auth_headers, task_id, timeout=10)["data"]
    assert task["status"] == "failed"
    assert "authentication" in task["error"]


# 5 ------------------------------------------------- redaction fails closed


def _empty_engine(image, **_):
    from types import SimpleNamespace

    return SimpleNamespace(txts=None, boxes=None, word_results=[[[], [], []]])


def _image_only_pdf(draw) -> bytes:
    """A scan: the page is one picture, no text layer to read identifiers from."""
    with fitz.open() as source:
        page = source.new_page(width=612, height=792)
        draw(page)
        png = page.get_pixmap(dpi=200).tobytes("png")
    with fitz.open() as doc:
        page = doc.new_page(width=612, height=792)
        page.insert_image(page.rect, stream=png)
        return doc.tobytes()


def test_redaction_accepts_a_blank_scan_back(monkeypatch):
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: _empty_engine)
    blank = _image_only_pdf(lambda page: None)
    result = ir.redact_identifiers(blank)
    assert result["ocr_pages"] == [0]
    assert result["redactions"] == []
    assert ir.ink_fraction(blank, 0, 216) == 0


def test_redaction_refuses_a_faint_ssn_ocr_missed(monkeypatch):
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: _empty_engine)
    faint = _image_only_pdf(
        lambda page: page.insert_text((72, 120), "SSN 123-45-6789", fontsize=10,
                                      color=(0.8, 0.8, 0.8))
    )
    with pytest.raises(ir.RedactionIncompleteError,
                       match=r"page\(s\) \[0\] show ink but OCR read no text"):
        ir.redact_identifiers(faint)
    assert ir.ink_fraction(faint, 0, 216) > ir._BLANK_MAX_INK_FRACTION


def test_redaction_refuses_a_page_ocr_failed_on(monkeypatch):
    def crash(image, **_):
        raise RuntimeError("inference crashed")

    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: crash)
    with pytest.raises(ir.RedactionIncompleteError, match=r"OCR failed on page\(s\) \[0\]"):
        ir.redact_identifiers(_image_only_pdf(lambda page: None))


# 6 ---------------------------------------------------------------- WORKERS


def test_workers_above_one_fail_startup_with_an_actionable_message():
    with pytest.raises(ValueError) as error:
        Settings(workers=3, _env_file=None)
    message = str(error.value)
    assert "WORKERS=3 is not supported" in message
    assert "WORKERS must be 1" in message and "OCR_WORKERS" in message


# 7 ------------------------------------------------------------------- minor


def test_task_status_is_read_off_the_event_loop(client, auth_headers, monkeypatch):
    from app.services import task_service

    seen = []

    def get_task_probe(task_id):
        try:
            asyncio.get_running_loop()
            seen.append("event loop")
        except RuntimeError:
            seen.append("worker thread")
        return None

    monkeypatch.setattr(task_service, "get_task", get_task_probe)
    assert get_task(client, auth_headers, "x").json()["error"]["code"] == "TASK_NOT_FOUND"
    assert seen == ["worker thread"]


def test_queue_limit_is_atomic_with_the_enqueue(monkeypatch):
    monkeypatch.setattr(settings, "ocr_queue_max_jobs", 3)
    gate = threading.Event()
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: FakeEngine(delay=0, gate=gate))
    real = ocr_queue._as_pdf_bytes

    def slow(data):
        time.sleep(0.05)  # widen the window between the check and the enqueue
        return real(data)

    monkeypatch.setattr(ocr_queue, "_as_pdf_bytes", slow)
    queue = ocr_queue.get_queue()
    accepted, refused = [], []

    def go(index):
        try:
            accepted.append(queue.submit_ocr(scanned(1, str(index)), None, "en", 300, "rapid"))
        except ocr_queue.QueueFullError:
            refused.append(index)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    gate.set()
    assert len(accepted) == 3
    assert len(refused) == 7


def test_synchronous_ocr_fails_closed_after_its_budget(monkeypatch):
    gate = threading.Event()
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: FakeEngine(delay=0, gate=gate))
    monkeypatch.setattr(settings, "ocr_sync_timeout_seconds", 0.3)
    started = time.monotonic()
    with pytest.raises(ocr_queue.OcrTimeoutError, match="did not finish within 0.3s"):
        ir.redact_identifiers(scanned(2))
    assert time.monotonic() - started < 5
    gate.set()
    # The pool is still usable once the stuck inference returns.
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: FakeEngine(delay=0))
    result = rapid_ocr.ocr_pages(scanned(1))
    assert result["pages"][0]["full_text"] == "p0"
