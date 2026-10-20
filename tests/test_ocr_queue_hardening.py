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
    task_id = queue.submit_ocr(scanned(1), [], "en", 300)
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
    # Plaintext markers long enough not to occur in ciphertext by chance.
    assert blobs and all(b'"full_text"' not in bytes(r["result"]) for r in blobs)
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
    assert b'"full_text"' not in raw and b"pdf_base64" not in raw


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
    monkeypatch.setattr(settings, "pdf_engine_secret", "a-real-secret")
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


def _noisy_scan(sigma: float, specks: int = 0, seed: int = 0) -> bytes:
    """A blank scan back: off-white paper, Gaussian sensor noise, dust specks."""
    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    page = np.clip(rng.normal(250, sigma, (2200, 1700)), 0, 255).astype(np.uint8)
    for _ in range(specks):
        center = (int(rng.integers(0, 1700)), int(rng.integers(0, 2200)))
        cv2.circle(page, center, int(rng.integers(1, 4)), int(rng.integers(60, 200)), -1)
    with fitz.open() as doc:
        out = doc.new_page(width=612, height=792)
        out.insert_image(out.rect, stream=cv2.imencode(".png", page)[1].tobytes())
        return doc.tobytes()


@pytest.mark.parametrize("fixture", ["blank", "light noise", "50 dust specks"])
def test_redaction_accepts_a_blank_scan_back(monkeypatch, fixture):
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: _empty_engine)
    page = {
        "blank": lambda: _image_only_pdf(lambda page: None),
        "light noise": lambda: _noisy_scan(sigma=4),
        "50 dust specks": lambda: _noisy_scan(sigma=4, specks=50),
    }[fixture]()
    result = ir.redact_identifiers(page)
    assert result["ocr_pages"] == [0]
    assert result["redactions"] == []
    assert ir.text_like_marks(page, 0, 216)["text_like"] is False


@pytest.mark.parametrize("gray", [0, 150, 200, 215, 225, 235, 240, 244, 246])
@pytest.mark.parametrize("size", [4, 6, 8, 10])
def test_redaction_refuses_any_printed_ssn_ocr_missed(monkeypatch, size, gray):
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: _empty_engine)
    level = gray / 255
    page = _image_only_pdf(
        lambda p: p.insert_text((72, 120), "123-45-6789", fontsize=size,
                                color=(level, level, level))
    )
    with pytest.raises(ir.RedactionIncompleteError,
                       match=r"page\(s\) \[0\] contain text OCR did not read"):
        ir.redact_identifiers(page)


def _blurred_ssn_pdf(size: int, gray: int, sigma: float) -> bytes:
    """A soft scan: the printed SSN Gaussian-blurred until glyphs merge."""
    import cv2
    import numpy as np

    level = gray / 255
    with fitz.open() as source:
        page = source.new_page(width=612, height=792)
        page.insert_text((72, 120), "123-45-6789", fontsize=size, color=(level, level, level))
        pix = page.get_pixmap(dpi=200, colorspace=fitz.csGRAY)
        image = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
    image = cv2.GaussianBlur(image, (0, 0), sigma)
    with fitz.open() as doc:
        out = doc.new_page(width=612, height=792)
        out.insert_image(out.rect, stream=cv2.imencode(".png", image)[1].tobytes())
        return doc.tobytes()


@pytest.mark.parametrize("sigma", [1, 2, 3])
@pytest.mark.parametrize("gray", [0, 200])
@pytest.mark.parametrize("size", [10, 12])
def test_redaction_refuses_a_blurred_ssn_ocr_missed(monkeypatch, size, gray, sigma):
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: _empty_engine)
    with pytest.raises(ir.RedactionIncompleteError, match="contain text OCR did not read"):
        ir.redact_identifiers(_blurred_ssn_pdf(size, gray, sigma))


@pytest.mark.parametrize("rotate", [90, 270])
@pytest.mark.parametrize("size", [6, 10])
def test_redaction_refuses_a_rotated_ssn_ocr_missed(monkeypatch, size, rotate):
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: _empty_engine)
    page = _image_only_pdf(
        lambda p: p.insert_text((300, 200), "123-45-6789", fontsize=size, rotate=rotate)
    )
    with pytest.raises(ir.RedactionIncompleteError, match="contain text OCR did not read"):
        ir.redact_identifiers(page)


def test_dusty_and_noisy_scan_false_refusal_profile():
    """Documents where a blank back is refused (fails closed, never leaks).

    Sparse dust passes; heavy dust can put two specks side by side, and heavy
    sensor noise (sigma 8 on 255) forms aligned clusters: both refuse.
    """
    profile = {
        (sigma, specks): ir.text_like_marks(_noisy_scan(sigma, specks), 0, 216)["text_like"]
        for sigma, specks in [(4, 0), (4, 50), (4, 200), (8, 0)]
    }
    assert profile[(4, 0)] is False and profile[(4, 50)] is False
    assert profile[(4, 200)] is False
    assert profile[(8, 0)] is True


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
            accepted.append(queue.submit_ocr(scanned(1, str(index)), None, "en", 300))
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


# re-verify round ----------------------------------------------------------


def test_an_unreadable_result_is_a_failed_task_not_a_500(
    client, auth_headers, documents, monkeypatch
):
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: FakeEngine(delay=0))
    documents["https://example.com/a.pdf"] = scanned(1, "a")
    task_id = submit(client, auth_headers, "https://example.com/a.pdf").json()["task_id"]
    assert poll(client, auth_headers, task_id, timeout=10)["data"]["status"] == "completed"

    ocr_queue.shutdown()
    monkeypatch.setattr(settings, "ocr_queue_key", "rotated")
    ocr_queue.get_queue()
    response = get_task(client, auth_headers, task_id)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["status"] == "failed"
    assert "OCR result unreadable" in body["data"]["error"]

    [result] = queue_files("results")
    result.write_bytes(b"truncated")
    response = get_task(client, auth_headers, task_id)
    assert response.status_code == 200
    assert response.json()["data"]["status"] == "failed"


def test_default_secret_without_a_queue_key_refuses_to_start(monkeypatch):
    from app.main import app, lifespan

    monkeypatch.setattr(settings, "ocr_queue_key", "")
    monkeypatch.setattr(settings, "pdf_engine_secret", "change-me")
    monkeypatch.setattr(rapid_ocr, "validate_runtime", lambda: None)

    async def boot():
        async with lifespan(app):
            pytest.fail("the engine started with a publicly known queue key")

    with pytest.raises(RuntimeError, match="Set OCR_QUEUE_KEY"):
        asyncio.run(boot())


def test_sweep_survives_a_file_vanishing_mid_listing(monkeypatch):
    queue = ocr_queue.get_queue()
    ghost = Path(settings.ocr_queue_dir) / "inputs" / "ghost.bin"
    ghost.write_bytes(b"x")
    real_iterdir = Path.iterdir

    def listing_then_vanish(self):
        entries = list(real_iterdir(self))
        if ghost in entries:
            ghost.unlink()  # finished and deleted between iterdir and stat
        return iter(entries)

    monkeypatch.setattr(Path, "iterdir", listing_then_vanish)
    removed = queue.sweep()
    assert removed["files"] == 0
    assert queue.worker_restarts == 0


def test_ocr_results_never_reach_the_plaintext_cache_and_still_dedupe(
    client, auth_headers, documents, monkeypatch
):
    engine = FakeEngine(delay=0)
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: engine)
    documents["https://example.com/a.pdf"] = scanned(1, "a")
    first = submit(client, auth_headers, "https://example.com/a.pdf").json()["task_id"]
    assert poll(client, auth_headers, first, timeout=10)["data"]["status"] == "completed"

    cache = Path(settings.cache_dir)
    leaked = [
        path for path in (cache.rglob("*") if cache.exists() else [])
        if path.is_file()
        and (b'"full_text"' in path.read_bytes() or b"pdf_base64" in path.read_bytes())
    ]
    assert leaked == []
    again = submit(client, auth_headers, "https://example.com/a.pdf").json()
    assert again["success"] is True
    assert again["data"]["pages"][0]["full_text"] == "p0"  # served from the sealed ledger
    assert len(engine.calls) == 1


# adversarial round --------------------------------------------------------


def _w2_scan_with_words():
    """An image-only W-2 page and its words in visual points (from the source)."""
    lines = [
        ((72, 100), "Form W-2 Wage and Tax Statement"),
        ((72, 160), "Employee SSN 123-45-6789"),
        ((72, 220), "Wages 75250.00"),
    ]
    with fitz.open() as source:
        page = source.new_page(width=612, height=792)
        for origin, text in lines:
            page.insert_text(origin, text, fontsize=11)
        # Corner dots stop scan preparation from cropping the margins, so the
        # fake engine's pixel boxes line up with the full render.
        for x, y in [(2, 2), (608, 2), (2, 788), (608, 788)]:
            page.draw_rect(fitz.Rect(x, y, x + 1, y + 1), color=(0.5, 0.5, 0.5), width=0.3)
        words = [(w[4], fitz.Rect(w[:4])) for w in page.get_text("words")]
        png = page.get_pixmap(dpi=200).tobytes("png")
    with fitz.open() as doc:
        out = doc.new_page(width=612, height=792)
        out.insert_image(out.rect, stream=png)
        return doc.tobytes(), words


def _engine_reading(words, keep=lambda text: True, redacted=False):
    """A RapidOCR stand-in that reads the given words (216 DPI = 3 px/pt)."""
    from types import SimpleNamespace

    import numpy as np

    def engine(image, **_):
        texts, groups, boxes = [], [], []
        for text, rect in words:
            if not keep(text):
                continue
            if redacted and text == "123-45-6789":
                text = "XXX-XX-6789"
            polygon = np.array([[rect.x0, rect.y0], [rect.x1, rect.y0],
                                [rect.x1, rect.y1], [rect.x0, rect.y1]], dtype=float) * 3
            texts.append(text)
            groups.append([(text, 0.95, polygon)])
            boxes.append(polygon)
        return SimpleNamespace(txts=tuple(texts) or None,
                               boxes=np.array(boxes) if boxes else None,
                               word_results=groups or [[[], [], []]])

    return engine


def test_a_partly_read_page_with_an_unread_ssn_refuses(monkeypatch):
    """OCR reads the title but misses the SSN line: the page is not 'empty'."""
    scan, words = _w2_scan_with_words()
    title = {"Form", "W-2", "Wage", "and", "Tax", "Statement"}
    engine = _engine_reading(words, keep=lambda text: text in title)
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: engine)
    with pytest.raises(ir.RedactionIncompleteError,
                       match=r"page\(s\) \[0\] contain text OCR did not read"):
        ir.redact_identifiers(scan)


def test_a_fully_read_w2_page_is_redacted(monkeypatch):
    scan, words = _w2_scan_with_words()
    calls = []

    def engine(image, **kwargs):
        calls.append(1)
        # The re-read of the redacted page sees the box, not the digits.
        return _engine_reading(words, redacted=len(calls) > 1)(image, **kwargs)

    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: engine)
    result = ir.redact_identifiers(scan)
    assert result["redactions"] == [{"page": 0, "kind": "ssn", "last4": "6789", "source": "ocr"}]
    assert len(calls) == 2


def test_an_empty_page_request_is_not_cached_for_the_full_document(
    client, auth_headers, documents, monkeypatch
):
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: FakeEngine(delay=0))
    documents["https://example.com/a.pdf"] = scanned(2, "a")
    body = json.dumps({"source_url": "https://example.com/a.pdf", "pages": []})
    empty = client.post("/text/ocr", content=body, headers=auth_headers("POST", "/text/ocr", body))
    assert poll(client, auth_headers, empty.json()["task_id"], timeout=10)["data"][
        "result"]["pages"] == []

    full = submit(client, auth_headers, "https://example.com/a.pdf").json()
    if "task_id" in full:
        full = poll(client, auth_headers, full["task_id"], timeout=10)["data"]["result"]
    else:
        full = full["data"]
    assert [p["full_text"] for p in full["pages"]] == ["p0", "p1"]


def test_sealed_data_is_bound_to_its_job(client, auth_headers, documents, monkeypatch):
    """Associated data: job A's blobs are not readable as job B's."""
    monkeypatch.setattr(settings, "ocr_max_active_jobs", 1)
    gate = threading.Event()
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: FakeEngine(delay=0, gate=gate))
    queue = ocr_queue.get_queue()
    for name in ("a", "b", "c", "d"):
        documents[f"https://example.com/{name}.pdf"] = scanned(1, name)
    a = submit(client, auth_headers, "https://example.com/a.pdf").json()["task_id"]
    b = submit(client, auth_headers, "https://example.com/b.pdf").json()["task_id"]
    inputs = Path(settings.ocr_queue_dir) / "inputs"

    # Input: B, still queued, gets A's sealed input.
    (inputs / f"{b}.bin").write_bytes((inputs / f"{a}.bin").read_bytes())
    # Unit checkpoints: A's page 0 copied to B's page 0 and to A's page 1.
    store = queue.store
    store.save_unit(a, 0, {"full_text": "A"})
    blob = store._execute("SELECT result FROM ocr_job_units WHERE job_id = ?", (a,))[0][0]
    store._execute("INSERT INTO ocr_job_units (job_id, unit, result) VALUES (?, 0, ?)", (b, blob))
    with pytest.raises(RuntimeError, match="authentication"):
        store.units(b)
    store._execute("INSERT INTO ocr_job_units (job_id, unit, result) VALUES (?, 1, ?)", (a, blob))
    with pytest.raises(RuntimeError, match="authentication"):
        store.units(a)
    store._execute("DELETE FROM ocr_job_units WHERE job_id = ? AND unit = 1", (a,))
    store._execute("DELETE FROM ocr_job_units WHERE job_id = ?", (b,))

    gate.set()
    assert poll(client, auth_headers, a, timeout=10)["data"]["status"] == "completed"
    task_b = poll(client, auth_headers, b, timeout=10)["data"]
    assert task_b["status"] == "failed" and "authentication" in task_b["error"]

    # Result: D shows C's sealed result.
    c = submit(client, auth_headers, "https://example.com/c.pdf").json()["task_id"]
    d = submit(client, auth_headers, "https://example.com/d.pdf").json()["task_id"]
    for task_id in (c, d):
        assert poll(client, auth_headers, task_id, timeout=10)["data"]["status"] == "completed"
    results = Path(settings.ocr_queue_dir) / "results"
    (results / f"{d}.bin").write_bytes((results / f"{c}.bin").read_bytes())
    task_d = get_task(client, auth_headers, d).json()["data"]
    assert task_d["status"] == "failed"
    assert "authentication" in task_d["error"]


# submit-path input validation ---------------------------------------------


def _zero_page_pdf() -> bytes:
    return (
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
        b"trailer<</Root 1 0 R>>\n%%EOF\n"
    )


def _password_pdf() -> bytes:
    with fitz.open() as doc:
        doc.new_page()
        return doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="secret",
                           owner_pw="owner")


@pytest.mark.parametrize(
    "name, data, code",
    [
        ("corrupt", b"%PDF-1.7\nthis is not a pdf at all", "INVALID_PDF"),
        ("password", None, "INVALID_PDF"),
        ("zero pages", None, "INVALID_PDF"),
    ],
)
def test_unreadable_ocr_input_is_a_422_not_a_500(
    client, auth_headers, documents, monkeypatch, name, data, code
):
    data = {"password": _password_pdf, "zero pages": _zero_page_pdf}.get(name, lambda: data)()
    documents["https://example.com/bad.pdf"] = data
    response = submit(client, auth_headers, "https://example.com/bad.pdf")
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == code
    assert body["error"]["message"]
    inputs = Path(settings.ocr_queue_dir) / "inputs"
    assert not inputs.exists() or not list(inputs.iterdir())  # nothing was queued


def test_a_healthy_pdf_is_still_queued_after_validation(
    client, auth_headers, documents, monkeypatch
):
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: FakeEngine(delay=0))
    documents["https://example.com/ok.pdf"] = scanned(2, "ok")
    response = submit(client, auth_headers, "https://example.com/ok.pdf")
    assert response.status_code == 200
    accepted = response.json()
    assert accepted["status"] == "pending" and "2 pages" in accepted["message"]
    task = poll(client, auth_headers, accepted["task_id"], timeout=10)["data"]
    assert task["status"] == "completed"


# (b) + (c): identifier-shaped unread print, low-confidence words, lines ----


def _scan_with(draw):
    """An image-only page plus its source words; corner dots prevent cropping."""
    with fitz.open() as source:
        page = source.new_page(width=612, height=792)
        draw(page)
        for x, y in [(2, 2), (608, 2), (2, 788), (608, 788)]:
            page.draw_rect(fitz.Rect(x, y, x + 1, y + 1), color=(0.5, 0.5, 0.5), width=0.3)
        words = [(w[4], fitz.Rect(w[:4])) for w in page.get_text("words")]
        png = page.get_pixmap(dpi=200).tobytes("png")
    with fitz.open() as doc:
        out = doc.new_page(width=612, height=792)
        out.insert_image(out.rect, stream=png)
        return doc.tobytes(), words


def _engine(lines):
    """Fake RapidOCR: lines = [(line text, line rect, [(word, rect, score)])]."""
    from types import SimpleNamespace

    import numpy as np

    def poly(rect):
        return np.array([[rect.x0, rect.y0], [rect.x1, rect.y0],
                         [rect.x1, rect.y1], [rect.x0, rect.y1]], dtype=float) * 3

    def engine(image, **_):
        if not lines:
            return SimpleNamespace(txts=None, boxes=None, word_results=[[[], [], []]])
        return SimpleNamespace(
            txts=tuple(text for text, _, _ in lines),
            boxes=np.array([poly(rect) for _, rect, _ in lines]),
            word_results=[[(w, s, poly(r)) for w, r, s in ws] for _, _, ws in lines],
        )

    return engine


def test_a_low_confidence_ssn_is_redacted(monkeypatch):
    scan, words = _scan_with(lambda p: p.insert_text((72, 160), "SSN 123-45-6789", fontsize=11))
    (_, label), (_, ssn) = words
    line = label | ssn
    reads = []

    def engine(image, **kwargs):
        reads.append(1)
        text = "123-45-6789" if len(reads) == 1 else "XXX-XX-6789"
        return _engine([(f"SSN {text}", line, [("SSN", label, 0.9), (text, ssn, 0.31)])])(
            image, **kwargs)

    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: engine)
    result = ir.redact_identifiers(scan)
    assert result["redactions"] == [{"page": 0, "kind": "ssn", "last4": "6789", "source": "ocr"}]


def test_low_confidence_words_and_lines_stay_out_of_the_ocr_response(monkeypatch):
    scan, words = _scan_with(lambda p: p.insert_text((72, 160), "SSN 123-45-6789", fontsize=11))
    (_, label), (_, ssn) = words
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: _engine(
        [("SSN 123-45-6789", label | ssn, [("SSN", label, 0.9), ("123-45-6789", ssn, 0.31)])]))
    page = rapid_ocr.ocr_pages(scan)["pages"][0]
    assert [w["text"] for w in page["words"]] == ["SSN"]
    assert page["discarded_words"] == 1
    assert not any(key.startswith("_") for key in page)


def test_an_ssn_read_only_as_a_line_is_redacted(monkeypatch):
    """RapidOCR read the line but split it into fragments: the line text is searched."""
    scan, words = _scan_with(lambda p: p.insert_text((72, 160), "SSN 123-45-6789", fontsize=11))
    (_, label), (_, ssn) = words
    line = label | ssn
    reads = []

    def engine(image, **kwargs):
        reads.append(1)
        text = "SSN 123-45-6789" if len(reads) == 1 else "SSN XXX-XX-6789"
        fragment = fitz.Rect(ssn.x0, ssn.y0, ssn.x0 + 6, ssn.y1)
        return _engine([(text, line, [("S", fragment, 0.9)])])(image, **kwargs)

    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: engine)
    result = ir.redact_identifiers(scan)
    assert result["redactions"] == [{"page": 0, "kind": "ssn", "last4": "6789", "source": "ocr"}]


def test_a_logo_and_a_shaded_box_do_not_refuse(monkeypatch):
    def draw(page):
        page.insert_text((72, 100), "Form 1099-INT", fontsize=12)
        page.draw_circle((480, 100), 30, color=(0.1, 0.2, 0.6), fill=(0.1, 0.2, 0.6))
        page.draw_polyline([(440, 160), (520, 160), (480, 120), (440, 160)],
                           color=(0.6, 0.1, 0.1), fill=(0.6, 0.1, 0.1))
        page.draw_rect(fitz.Rect(72, 300, 400, 340), color=(0.85, 0.85, 0.85),
                       fill=(0.85, 0.85, 0.85))

    scan, words = _scan_with(draw)
    line = words[0][1] | words[1][1]
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: _engine(
        [("Form 1099-INT", line, [(w, r, 0.95) for w, r in words])]))
    result = ir.redact_identifiers(scan)
    assert result["redactions"] == []


def test_unread_nine_digit_print_refuses(monkeypatch):
    def draw(page):
        page.insert_text((72, 100), "Form W-2", fontsize=12)
        page.insert_text((72, 200), "123456789", fontsize=10)

    scan, words = _scan_with(draw)
    title = [(w, r) for w, r in words if w in ("Form", "W-2")]
    line = title[0][1] | title[1][1]
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: _engine(
        [("Form W-2", line, [(w, r, 0.95) for w, r in title])]))
    with pytest.raises(ir.RedactionIncompleteError, match="contain text OCR did not read"):
        ir.redact_identifiers(scan)


def test_an_unread_three_glyph_mark_does_not_refuse(monkeypatch):
    def draw(page):
        page.insert_text((72, 100), "Form W-2", fontsize=12)
        page.insert_text((300, 400), "A1B", fontsize=10)

    scan, words = _scan_with(draw)
    title = [(w, r) for w, r in words if w in ("Form", "W-2")]
    line = title[0][1] | title[1][1]
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: _engine(
        [("Form W-2", line, [(w, r, 0.95) for w, r in title])]))
    assert ir.redact_identifiers(scan)["redactions"] == []


def test_rapid_provenance_is_no_longer_experimental(monkeypatch):
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: FakeEngine(delay=0))
    provenance = rapid_ocr.ocr_pages(scanned(1))["provenance"]
    assert provenance["experimental"] is False
    assert provenance["profile"] == rapid_ocr.PROFILE


# held-out corpus round: confusions, digit budget, inverted print ----------


def _segmented_cases():
    """Identifier print that OCR did not read, in layouts that defeat runs."""

    def groups(parts, size, gap, leaders=False):
        def draw(p):
            x, cut, digits = 72, 0, "123456789"
            for n, length in enumerate(parts):
                chunk = digits[cut:cut + length]
                cut += length
                p.insert_text((x, 200), chunk, fontsize=size)
                width = fitz.get_text_length(chunk, fontsize=size)
                if leaders and n < len(parts) - 1:
                    p.insert_text((x + width + 2, 200),
                                  "." * max(1, int((gap - 4) / (size * 0.28))), fontsize=size)
                x += width + gap
        return draw

    def comb(size, factor):
        def draw(p):
            x = 72
            for d in "123456789":
                p.draw_rect(fitz.Rect(x - size * 0.4, 200 - size * 1.1, x + size,
                                      200 + size * 0.4), width=0.6)
                p.insert_text((x, 200), d, fontsize=size)
                x += size * factor
        return draw

    def w9(p, size=10):
        x = 72
        for part in ("123", "45", "6789"):
            width = fitz.get_text_length(part, fontsize=size)
            p.draw_rect(fitz.Rect(x - 4, 200 - size * 1.2, x + width + 4, 200 + size * 0.5),
                        width=0.8)
            p.insert_text((x, 200), part, fontsize=size)
            x += width + 26

    def inverted(fill):
        def draw(p):
            p.draw_rect(fitz.Rect(60, 180, 260, 210), color=fill, fill=fill)
            p.insert_text((72, 200), "123-45-6789", fontsize=12, color=(1, 1, 1))
        return draw

    cases = {}
    for size in (8, 10):
        for gap in (12, 20, 30):
            cases[f"3|2|4 {size}pt gap {gap}pt"] = groups((3, 2, 4), size, gap)
            cases[f"leaders {size}pt gap {gap}pt"] = groups((3, 2, 4), size, gap, True)
        cases[f"3|2|3|1 {size}pt"] = groups((3, 2, 3, 1), size, 20)
    for size in (6, 8, 10, 12, 14):
        for factor in (3, 4):
            cases[f"comb {size}pt x{factor}"] = comb(size, factor)
    cases["w9 boxes"] = w9
    cases["split across lines"] = lambda p: (p.insert_text((72, 200), "123-45-", fontsize=10),
                                             p.insert_text((72, 260), "6789", fontsize=10))
    cases["ein 180pt gap"] = lambda p: (p.insert_text((72, 200), "12-", fontsize=10),
                                        p.insert_text((252, 200), "3456789", fontsize=10))
    cases["white on black"] = inverted((0, 0, 0))
    cases["white on 30% gray"] = inverted((0.7, 0.7, 0.7))
    return cases


@pytest.mark.parametrize("name", sorted(_segmented_cases()))
def test_unread_segmented_or_inverted_identifiers_refuse(name):
    page = _image_only_pdf(_segmented_cases()[name])
    assert ir.text_like_marks(page, 0, 216)["text_like"], name


def _explicit_ocr(scan_words, replace):
    """An OCR stand-in returning the source words, some texts replaced."""
    calls = []

    def ocr(pdf_bytes, pages):
        calls.append(1)
        words = []
        for text, rect in scan_words:
            if len(calls) == 1 and text in replace:
                text = replace[text]
            elif len(calls) > 1 and text in replace:
                text = "XXXXXXX"
            words.append(ir.Word(text, rect))
        return {0: words}

    return ocr


@pytest.mark.parametrize(
    "reading",
    ["I23-45-678S", "l23 45  67B9", "123-4S-6789", "I23 ZS 67B q", "1Z3 .... 45 ……. 6789"],
)
def test_a_confusion_garbled_ssn_is_redacted(reading):
    scan, words = _scan_with(lambda p: p.insert_text((72, 160), "123-45-6789", fontsize=11))
    result = ir.redact_identifiers(scan, ocr=_explicit_ocr(words, {"123-45-6789": reading}))
    assert len(result["redactions"]) == 1
    assert result["redactions"][0]["source"] == "ocr"


def test_a_near_identifier_digit_run_refuses():
    scan, words = _scan_with(lambda p: p.insert_text((72, 160), "123-45-6789", fontsize=11))
    with pytest.raises(ir.RedactionIncompleteError, match="misread identifier"):
        ir.redact_identifiers(scan, ocr=_explicit_ocr(words, {"123-45-6789": "123-45-678"}))


def test_lone_four_digit_numbers_on_a_read_form_do_not_refuse(monkeypatch):
    """ZIP+4, years, form and OMB numbers: read, not identifiers, not suspect."""
    def draw(p):
        p.insert_text((72, 100), "Form 1099-INT Rev. 04-2025 OMB No. 1545-0112", fontsize=10)
        p.insert_text((72, 130), "Austin, TX 78701-1234 (512) 555-0100", fontsize=10)
        p.insert_text((72, 160), "Tax year 2025 Account A-0417 Box 1 4,182.55", fontsize=10)

    scan, words = _scan_with(draw)
    lines = {}
    for text, rect in words:
        lines.setdefault(round(rect.y1), []).append((text, rect))
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: _engine([
        (" ".join(t for t, _ in ws), ws[0][1] | ws[-1][1], [(t, r, 0.95) for t, r in ws])
        for ws in lines.values()
    ]))
    assert ir.redact_identifiers(scan)["redactions"] == []


def test_a_rotated_read_identifier_is_redacted_whole(monkeypatch):
    scan, words = _scan_with(
        lambda p: p.insert_text((300, 400), "123-45-6789", fontsize=11, rotate=90))
    [(text, rect)] = words
    assert rect.height > 3 * rect.width
    result = ir.redact_identifiers(scan, ocr=_explicit_ocr(words, {text: text}))
    assert len(result["redactions"]) == 1
    with fitz.open(stream=result["pdf_bytes"], filetype="pdf") as doc:
        pix = doc[0].get_pixmap(dpi=72, clip=rect, colorspace=fitz.csGRAY)
    assert max(pix.samples) < 60  # the whole rotated number is blacked out


@pytest.mark.parametrize("reading", ["8 5 6 5 0 3 回", "B G G O Z"])
def test_a_comb_field_read_with_boxes_missing_refuses(reading):
    """Some comb boxes read (or all misread as letters): still an identifier."""
    scan, words = _scan_with(lambda p: p.insert_text((72, 160), "123-45-6789", fontsize=11))
    with pytest.raises(ir.RedactionIncompleteError, match="misread identifier"):
        ir.redact_identifiers(scan, ocr=_explicit_ocr(words, {"123-45-6789": reading}))


@pytest.mark.parametrize("kind", ["input", "result"])
def test_private_queue_read_rejects_foreign_paths_ids_and_symlinks(tmp_path, kind):
    store = ocr_queue._Store(tmp_path / "store", ocr_queue._Cipher(b"x" * 32))
    job_id = "11111111-1111-4111-8111-111111111111"
    directory = store.inputs if kind == "input" else store.results
    expected = directory / f"{job_id}.bin"
    payload = b"document" if kind == "input" else b'{"pages":[]}'
    sealed = store.cipher.seal(payload, f"{kind}:{job_id}")
    expected.write_bytes(sealed)
    read = store.read_input if kind == "input" else store.read_result
    try:
        assert read(job_id, str(expected)) == (payload if kind == "input" else {"pages": []})
        foreign = tmp_path / "outside.bin"
        foreign.write_bytes(sealed)
        with pytest.raises(ValueError, match="does not belong"):
            read(job_id, str(foreign))
        with pytest.raises(ValueError, match="Invalid OCR queue job ID"):
            read("../outside", str(foreign))
        expected.unlink()
        expected.symlink_to(foreign)
        with pytest.raises(OSError):
            read(job_id, str(expected))
    finally:
        store._db.close()
