"""Persistent, fair OCR job queue with bounded parallelism (TAX-6876).

Why it exists
-------------
Before this module every /text/ocr request ran in a FastAPI background thread
and lived in an in-memory dict. The RapidOCR path held one global lock per
document and gave up after 30 s ("RapidOCR pilot is busy"), and any restart
lost every queued and in-flight task.

What it does
------------
* **Persistent.** A submitted job's bytes are spilled to
  ``OCR_QUEUE_DIR/inputs/<id>.bin``; its state lives in
  ``OCR_QUEUE_DIR/jobs.sqlite3``. Each finished RapidOCR page is recorded, so
  after a restart ``pending`` and ``processing`` jobs are re-queued in their
  original order and resume from the first unfinished page. Results are
  written to ``OCR_QUEUE_DIR/results/<id>.bin`` and kept for
  ``OCR_RESULT_TTL_SECONDS``; an expired result reads as unknown.
* **Private at rest.** Inputs, results and checkpointed page results hold
  client documents and their OCR text (SSNs). Each is sealed with AES-256-GCM
  bound to its job id (``OCR_QUEUE_KEY``, else a key HKDF-derived from
  ``PDF_ENGINE_SECRET``). Directories are 0700, files and the ledger 0600.
  An input is deleted as soon as its job finishes; a sweeper removes expired
  jobs, files without a ledger row, stale temporaries and orphan page rows.
* **Self-healing.** A failure after a unit ran (disk full while recording,
  assembling or writing the result) fails that job with the reason; workers
  never die, and the sweeper restarts any that do.
* **Bounded parallelism.** ``OCR_WORKERS`` threads run units. A RapidOCR unit
  is one page, read on the worker's own engine (own ONNX Runtime sessions), so
  at most ``OCR_WORKERS`` inferences run at once and one large scan is spread
  over every idle worker. Page results are reassembled in request order.
* **Fair.** Up to ``OCR_MAX_ACTIVE_JOBS`` documents are active at once, admitted
  FIFO; idle workers take pages from active documents round-robin. A one-page
  document submitted behind a 50-page scan gets the next free worker instead
  of waiting for the scan.
* **Never busy.** Submission only enqueues. ``OCR_QUEUE_MAX_JOBS`` unfinished
  jobs is the one limit; past it /text/ocr answers 429 with ``Retry-After``.

Threads, not processes: ONNX Runtime's ``InferenceSession.run`` and OpenCV
release the GIL, so N threads with N engines run N inferences in parallel in
one process, share the loaded Python/PyMuPDF runtime, and need no pickling of
page images. A RapidOCR engine is never shared between threads (it mutates
per-call state). PyMuPDF calls are serialized by ``rapid_ocr.FITZ_LOCK``.

The Paddle rollback (``OCR_BACKEND=paddle``) uses the same queue, with one unit
per document: Paddle inference is serialized by ``pdf_service._PADDLE_LOCK``
anyway, so splitting pages would add nothing. Paddle units are not
checkpointed per page; they restart from the first page after a restart.
"""

from __future__ import annotations

import base64
import errno
import json
import math
import os
import sqlite3
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import structlog
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from app.config import settings
from app.services import rapid_ocr

log = structlog.get_logger()

PENDING, PROCESSING, COMPLETED, FAILED = "pending", "processing", "completed", "failed"
_WHOLE_DOCUMENT = -1
# Seconds per page assumed before any page has been timed in this process.
_INITIAL_PAGE_SECONDS = 4.0
# A temporary file older than this is a leftover of an interrupted write.
_STALE_TMP_SECONDS = 600
_MAGIC = b"OQ1\x00"
_DEFAULT_SECRET = "change-me"

_worker = threading.local()


class QueueFullError(Exception):
    """The queue holds OCR_QUEUE_MAX_JOBS unfinished jobs; retry later."""

    def __init__(self, retry_after: int):
        super().__init__(f"OCR queue is full; retry after {retry_after}s")
        self.retry_after = retry_after


class OcrTimeoutError(TimeoutError):
    """A synchronous OCR call (identifier redaction) exceeded its budget."""


class _StoppedError(Exception):
    pass


# ------------------------------------------------------------------ at rest


def queue_key() -> bytes:
    """AES-256 key for files and rows at rest.

    ``OCR_QUEUE_KEY`` may be 32 bytes in URL-safe base64 (used as is) or any
    passphrase (HKDF-SHA256). Without it the key is derived from
    ``PDF_ENGINE_SECRET``, so it is still secret and stable across restarts.
    """
    raw = settings.ocr_queue_key
    if not raw and settings.pdf_engine_secret == _DEFAULT_SECRET:
        raise RuntimeError(
            "OCR_QUEUE_KEY is not set and PDF_ENGINE_SECRET is the default; refusing to "
            "store client documents under a publicly known key. Set OCR_QUEUE_KEY."
        )
    if raw:
        try:
            key = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
            if len(key) == 32:
                return key
        except (ValueError, TypeError):
            pass
        material, info = raw.encode(), b"ocr-queue-key-v1"
    else:
        material, info = settings.pdf_engine_secret.encode(), b"ocr-queue-at-rest-v1"
    return HKDF(
        algorithm=hashes.SHA256(), length=32, salt=b"pdf-engine-toolbox/ocr-queue", info=info
    ).derive(material)


class _Cipher:
    def __init__(self, key: bytes):
        self._aead = AESGCM(key)

    def seal(self, data: bytes, context: str) -> bytes:
        nonce = os.urandom(12)
        return _MAGIC + nonce + self._aead.encrypt(nonce, data, context.encode())

    def open(self, blob: bytes, context: str) -> bytes:
        if not blob.startswith(_MAGIC):
            raise RuntimeError("OCR queue data is not sealed")
        nonce, sealed = blob[4:16], blob[16:]
        try:
            return self._aead.decrypt(nonce, sealed, context.encode())
        except InvalidTag as exc:
            raise RuntimeError("OCR queue data failed authentication") from exc


def _private_dir(path: Path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path, 0o700)


def _write_private(path: Path, data: bytes):
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if fd is not None:
            os.close(fd)
        _unlink(tmp)
        raise
    os.replace(tmp, path)


def _unlink(path):
    if path:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


class _Store:
    """SQLite job ledger. One connection, serialized by a lock."""

    def __init__(self, root: Path, cipher: _Cipher):
        self.root = root
        self.cipher = cipher
        self.inputs = root / "inputs"
        self.results = root / "results"
        for path in (root, self.inputs, self.results):
            _private_dir(path)
        self.db_path = root / "jobs.sqlite3"
        os.close(os.open(self.db_path, os.O_RDWR | os.O_CREAT, 0o600))
        self._lock = threading.Lock()
        self._db = sqlite3.connect(
            self.db_path, timeout=10, isolation_level=None, check_same_thread=False
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.execute("PRAGMA synchronous = NORMAL")
        self._db.execute("PRAGMA busy_timeout = 10000")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS ocr_jobs (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT NOT NULL UNIQUE,
                operation TEXT NOT NULL,
                backend TEXT NOT NULL,
                status TEXT NOT NULL,
                params TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                input_path TEXT,
                result_path TEXT,
                error TEXT,
                created_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                cache_key TEXT
            );
            CREATE INDEX IF NOT EXISTS ocr_jobs_status ON ocr_jobs (status, seq);
            CREATE TABLE IF NOT EXISTS ocr_job_units (
                job_id TEXT NOT NULL,
                unit INTEGER NOT NULL,
                result BLOB NOT NULL,
                PRIMARY KEY (job_id, unit)
            );
            """
        )
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(ocr_jobs)")}
        if "cache_key" not in columns:  # ledger written by an earlier build
            self._db.execute("ALTER TABLE ocr_jobs ADD COLUMN cache_key TEXT")
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS ocr_jobs_cache ON ocr_jobs (cache_key, status)"
        )
        self.restrict_permissions()

    def restrict_permissions(self):
        for suffix in ("", "-wal", "-shm"):
            path = Path(f"{self.db_path}{suffix}")
            if path.exists():
                os.chmod(path, 0o600)

    def _execute(self, sql, args=()):
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def _transaction(self, statements):
        with self._lock:
            try:
                self._db.execute("BEGIN IMMEDIATE")
                for sql, args in statements:
                    self._db.execute(sql, args)
                self._db.execute("COMMIT")
            except BaseException:
                if self._db.in_transaction:
                    self._db.execute("ROLLBACK")
                raise

    # Files ----------------------------------------------------------------

    def write_input(self, job_id: str, data: bytes) -> str:
        path = self.inputs / f"{job_id}.bin"
        _write_private(path, self.cipher.seal(data, f"input:{job_id}"))
        return str(path)

    def read_input(self, job_id: str, path: str) -> bytes:
        return self.cipher.open(Path(path).read_bytes(), f"input:{job_id}")

    def read_result(self, job_id: str, path: str) -> dict:
        return json.loads(self.cipher.open(Path(path).read_bytes(), f"result:{job_id}"))

    # Rows -----------------------------------------------------------------

    def insert(self, job: "_Job", status: str, error: str | None = None) -> int:
        now = time.time()
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO ocr_jobs (id, operation, backend, status, params, page_count,"
                " input_path, error, created_at, completed_at, cache_key)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    job.id, job.operation, job.backend, status, json.dumps(job.params),
                    job.page_count, job.source if status != FAILED else None, error,
                    job.created_at, now if status == FAILED else None,
                    job.params.get("cache_key"),
                ),
            )
            return cursor.lastrowid

    def mark_processing(self, job_id: str):
        self._execute(
            "UPDATE ocr_jobs SET status = ?, started_at = COALESCE(started_at, ?)"
            " WHERE id = ? AND status IN (?, ?)",
            (PROCESSING, time.time(), job_id, PENDING, PROCESSING),
        )

    def save_unit(self, job_id: str, unit: int, result: dict):
        sealed = self.cipher.seal(json.dumps(result).encode(), f"unit:{job_id}:{unit}")
        # Only while the job is unfinished: a late writer cannot orphan a row.
        self._execute(
            "INSERT OR REPLACE INTO ocr_job_units (job_id, unit, result)"
            " SELECT ?, ?, ? WHERE EXISTS"
            " (SELECT 1 FROM ocr_jobs WHERE id = ? AND status IN (?, ?))",
            (job_id, unit, sealed, job_id, PENDING, PROCESSING),
        )

    def units(self, job_id: str) -> dict[int, dict]:
        rows = self._execute("SELECT unit, result FROM ocr_job_units WHERE job_id = ?", (job_id,))
        return {
            row["unit"]: json.loads(self.cipher.open(row["result"], f"unit:{job_id}:{row['unit']}"))
            for row in rows
        }

    def complete(self, job_id: str, result: dict):
        path = self.results / f"{job_id}.bin"
        _write_private(path, self.cipher.seal(json.dumps(result).encode(), f"result:{job_id}"))
        try:
            self._finish(job_id, COMPLETED, result_path=str(path))
        except BaseException:
            _unlink(path)
            raise

    def fail(self, job_id: str, error: str):
        self._finish(job_id, FAILED, error=error)

    def _finish(self, job_id, status, result_path=None, error=None):
        rows = self._execute("SELECT input_path FROM ocr_jobs WHERE id = ?", (job_id,))
        self._transaction([
            (
                "UPDATE ocr_jobs SET status = ?, result_path = ?, error = ?, completed_at = ?,"
                " input_path = NULL WHERE id = ?",
                (status, result_path, error, time.time(), job_id),
            ),
            ("DELETE FROM ocr_job_units WHERE job_id = ?", (job_id,)),
        ])
        for row in rows:
            _unlink(row["input_path"])

    def cached(self, cache_key: str, ttl: int) -> tuple[str, str] | None:
        """(job id, result path) of the newest unexpired result for this request."""
        rows = self._execute(
            "SELECT id, result_path FROM ocr_jobs WHERE cache_key = ? AND status = ?"
            " AND completed_at >= ? ORDER BY completed_at DESC LIMIT 1",
            (cache_key, COMPLETED, time.time() - ttl),
        )
        return (rows[0]["id"], rows[0]["result_path"]) if rows else None

    def get(self, job_id: str) -> dict | None:
        rows = self._execute("SELECT * FROM ocr_jobs WHERE id = ?", (job_id,))
        return dict(rows[0]) if rows else None

    def unfinished(self) -> list[dict]:
        rows = self._execute(
            "SELECT * FROM ocr_jobs WHERE status IN (?, ?) ORDER BY seq", (PENDING, PROCESSING)
        )
        return [dict(row) for row in rows]

    def prune(self, ttl: int):
        cutoff = time.time() - ttl
        rows = self._execute(
            "SELECT id, result_path, input_path FROM ocr_jobs"
            " WHERE status IN (?, ?) AND completed_at < ?",
            (COMPLETED, FAILED, cutoff),
        )
        for row in rows:
            _unlink(row["result_path"])
            _unlink(row["input_path"])
            self._execute("DELETE FROM ocr_jobs WHERE id = ?", (row["id"],))

    def sweep(self, ttl: int) -> dict:
        """Prune expired jobs and remove everything no ledger row refers to."""
        self.prune(ttl)
        referenced = set()
        for row in self._execute("SELECT input_path, result_path FROM ocr_jobs"):
            referenced.update(p for p in (row["input_path"], row["result_path"]) if p)
        removed = {"files": 0, "tmp": 0}
        now = time.time()
        for folder in (self.inputs, self.results):
            for path in folder.iterdir():
                if path.name.endswith(".tmp"):
                    try:
                        stale = now - path.stat().st_mtime > _STALE_TMP_SECONDS
                    except FileNotFoundError:
                        continue
                    if stale:
                        _unlink(path)
                        removed["tmp"] += 1
                elif str(path) not in referenced:
                    try:
                        recent = now - path.stat().st_mtime <= _STALE_TMP_SECONDS
                    except FileNotFoundError:
                        continue  # finished and deleted since the listing
                    # Re-check: a submit may have written the file just now.
                    if not recent and not self._execute(
                        "SELECT 1 FROM ocr_jobs WHERE input_path = ? OR result_path = ?",
                        (str(path), str(path)),
                    ):
                        _unlink(path)
                        removed["files"] += 1
        with self._lock:
            cursor = self._db.execute(
                "DELETE FROM ocr_job_units WHERE job_id NOT IN"
                " (SELECT id FROM ocr_jobs WHERE status IN (?, ?))",
                (PENDING, PROCESSING),
            )
            removed["unit_rows"] = cursor.rowcount
        self.restrict_permissions()
        return removed


class _Job:
    __slots__ = (
        "id", "operation", "backend", "params", "source", "persistent", "targets",
        "units", "done", "inflight", "error", "created_at", "seq", "page_count",
        "started", "event", "result", "finishing", "data", "terminal",
    )

    def __init__(self, *, operation, backend, params, source, persistent, targets,
                 page_count, job_id=None, created_at=None, done=None, data=None):
        self.id = job_id or str(uuid.uuid4())
        self.operation = operation
        self.backend = backend
        self.params = params
        self.source = source  # spilled path (persistent) or bytes (in memory)
        self.persistent = persistent
        self.targets = targets
        self.done: dict[int, object] = done or {}
        self.units = deque(unit for unit in targets if unit not in self.done)
        self.inflight = 0
        self.error: BaseException | None = None
        self.created_at = created_at or time.time()
        self.seq = 0
        self.page_count = page_count
        self.started = False
        self.event = threading.Event()
        self.result = None
        self.finishing = False
        self.data = data  # decrypted input, held only while the job is active
        # (status, error) when the ledger could not record the outcome.
        self.terminal: tuple[str, str] | None = None

    def remaining_pages(self) -> int:
        if self.backend == "rapid":
            return len(self.units) + self.inflight
        return self.page_count if (self.units or self.inflight) else 0


class OcrQueue:
    def __init__(self, root: Path, workers: int, max_active: int, max_jobs: int, ttl: int,
                 key: bytes | None = None, sweep_interval: float = 300.0,
                 sync_timeout: float = 1200.0):
        self.root = Path(root)
        self.workers = max(1, workers)
        self.max_active = max(1, max_active)
        self.max_jobs = max(1, max_jobs)
        self.ttl = ttl
        self.sweep_interval = sweep_interval
        self.sync_timeout = sync_timeout
        self.store = _Store(self.root, _Cipher(key or queue_key()))
        self._cond = threading.Condition()
        self._pending: deque[_Job] = deque()
        self._active: list[_Job] = []
        self._jobs: dict[str, _Job] = {}  # unfinished persistent jobs
        self._reserved = 0  # submissions past the limit check, not yet queued
        self._rr = 0
        self._seq = 0
        self._stopped = False
        self._threads: list[threading.Thread] = []
        self._sweeper: threading.Thread | None = None
        self._page_seconds = _INITIAL_PAGE_SECONDS
        self._running = 0
        self.max_running = 0
        self.worker_restarts = 0

    # ------------------------------------------------------------------ life

    def start(self) -> "OcrQueue":
        self.store.sweep(self.ttl)
        recovered = 0
        for row in self.store.unfinished():
            try:
                params = json.loads(row["params"])
                if not row["input_path"] or not Path(row["input_path"]).exists():
                    raise RuntimeError("OCR input was lost before processing finished")
                done = self.store.units(row["id"]) if row["backend"] == "rapid" else {}
            except Exception as exc:
                self.store.fail(row["id"], str(exc))
                continue
            job = _Job(
                operation=row["operation"], backend=row["backend"], params=params,
                source=row["input_path"], persistent=True, targets=params["targets"],
                page_count=row["page_count"], job_id=row["id"], created_at=row["created_at"],
                done=done,
            )
            # A job whose every page was recorded before the restart has no
            # unit left; the scheduler finishes it like any other.
            with self._cond:
                self._enqueue(job)
            recovered += 1
        if recovered:
            log.info("ocr_queue_recovered", jobs=recovered)
        for index in range(self.workers):
            self._threads.append(self._spawn_worker(index))
        self._sweeper = threading.Thread(target=self._sweep_loop, name="ocr-sweeper", daemon=True)
        self._sweeper.start()
        log.info("ocr_queue_started", workers=self.workers, max_active=self.max_active,
                 intra_op_threads=rapid_ocr.intra_op_threads(), root=str(self.root))
        return self

    def _spawn_worker(self, index: int) -> threading.Thread:
        thread = threading.Thread(target=self._work, name=f"ocr-worker-{index}", daemon=True)
        thread.start()
        return thread

    def shutdown(self):
        """Stop dispatching. Unfinished jobs stay in the ledger for the next start.

        In-flight units are abandoned, not awaited: like a crash, nothing they
        produce after this point is recorded.
        """
        with self._cond:
            self._stopped = True
            waiting = [*self._pending, *self._active]
            self._cond.notify_all()
        for job in waiting:
            if not job.persistent:
                job.error = _StoppedError("OCR queue stopped")
                job.event.set()
        for thread in self._threads:
            thread.join(timeout=0.2)

    def supervise(self):
        """Replace any worker thread that died."""
        with self._cond:
            if self._stopped:
                return
            for index, thread in enumerate(self._threads):
                if not thread.is_alive():
                    log.error("ocr_worker_restarted", worker=index)
                    self.worker_restarts += 1
                    self._threads[index] = self._spawn_worker(index)

    def sweep(self) -> dict:
        removed = self.store.sweep(self.ttl)
        self.supervise()
        return removed

    def _sweep_loop(self):
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._stopped, timeout=self.sweep_interval)
                if self._stopped:
                    return
            try:
                removed = self.sweep()
                if any(removed.values()):
                    log.info("ocr_queue_swept", **removed)
            except Exception as exc:  # noqa: BLE001 - next sweep retries
                log.error("ocr_queue_sweep_failed", error=str(exc))

    # ---------------------------------------------------------------- submit

    def submit_ocr(self, pdf_bytes: bytes, pages, language: str, dpi: int, backend: str,
                   cache_hash: str | None = None, cache_params: dict | None = None) -> str:
        """Persist an /text/ocr request and queue it. Returns the task id."""
        with self._cond:
            if len(self._jobs) + self._reserved >= self.max_jobs:
                raise QueueFullError(self._retry_after_locked())
            self._reserved += 1
        try:
            return self._submit(pdf_bytes, pages, language, dpi, backend,
                                cache_hash, cache_params)
        finally:
            with self._cond:
                self._reserved -= 1

    def cached_result(self, cache_hash: str, cache_params: dict) -> dict | None:
        """A finished result for identical bytes and parameters, within the TTL.

        OCR results are deduplicated here, from the sealed ledger, instead of
        the plaintext file cache (cache_service): they carry client text and
        the searchable PDF, and this keeps one copy under one retention rule.
        """
        hit = self.store.cached(cache_key(cache_hash, cache_params), self.ttl)
        if hit is None:
            return None
        try:
            return self.store.read_result(*hit)
        except Exception as exc:  # unreadable: run the request again
            log.warning("ocr_cached_result_unreadable", task_id=hit[0], error=str(exc))
            return None

    def _submit(self, pdf_bytes, pages, language, dpi, backend, cache_hash, cache_params):
        job_id = str(uuid.uuid4())
        data = _as_pdf_bytes(pdf_bytes)
        params = {
            "pages": pages, "language": language, "requested_dpi": dpi, "dpi": dpi,
            "cache_key": cache_key(cache_hash, cache_params) if cache_hash else None,
        }
        error, targets, page_count = None, [_WHOLE_DOCUMENT], len(pages or [])
        try:
            if backend == "rapid":
                targets, params["dpi"] = rapid_ocr.plan(data, pages, language, dpi)
                page_count = len(targets)
            else:
                page_count = page_count or _page_count(data)
        except Exception as exc:  # the request itself is invalid: fail its task
            error = str(exc)
        params["targets"] = targets
        if error is not None:
            job = _Job(operation="ocr", backend=backend, params=params, source=None,
                       persistent=True, targets=targets, page_count=page_count, job_id=job_id)
            self.store.insert(job, FAILED, error)
            log.error("task_failed", task_id=job_id, error=error)
            return job_id
        path = self.store.write_input(job_id, data)
        job = _Job(operation="ocr", backend=backend, params=params, source=path,
                   persistent=True, targets=targets, page_count=page_count, job_id=job_id)
        try:
            self.store.insert(job, PENDING)
        except BaseException:
            _unlink(path)
            raise
        with self._cond:
            self._enqueue(job)
        log.info("task_created", task_id=job_id, operation="ocr", backend=backend,
                 pages=page_count)
        return job_id

    def run_rapid(self, pdf_bytes: bytes, pages=None, language="en", dpi=300,
                  force_ocr=False, with_pdf=True, timeout: float | None = None) -> dict:
        """OCR in memory through the shared workers and wait for the result.

        Fails closed with ``OcrTimeoutError`` after ``timeout`` seconds
        (OCR_SYNC_TIMEOUT_SECONDS): a hung inference must not hang the caller.
        """
        targets, dpi = rapid_ocr.plan(pdf_bytes, pages, language, dpi)
        params = {"dpi": dpi, "force_ocr": force_ocr, "with_pdf": with_pdf}
        job = _Job(operation="rapid_ocr", backend="rapid", params=params, source=pdf_bytes,
                   persistent=False, targets=targets, page_count=len(targets))
        if getattr(_worker, "active", False):
            # Called from a worker: run here rather than wait on ourselves.
            for unit in targets:
                job.done[unit] = self._run_unit(job, unit)
            return self._assemble(job)
        with self._cond:
            if self._stopped:
                raise RuntimeError("OCR queue stopped")
            self._enqueue(job)
        budget = self.sync_timeout if timeout is None else timeout
        if not job.event.wait(budget):
            with self._cond:
                # Stop scheduling its remaining pages; a stuck inference keeps
                # its worker until it returns, and its result is discarded.
                job.error = OcrTimeoutError(f"OCR did not finish within {budget:g}s")
                job.units.clear()
                if job in self._pending:
                    self._pending.remove(job)
                if job in self._active and job.inflight == 0:
                    self._active.remove(job)
                self._cond.notify_all()
            log.error("ocr_sync_timeout", pages=len(targets), timeout_s=budget)
            raise job.error
        if job.error is not None:
            raise job.error
        return job.result

    def _enqueue(self, job: _Job):
        """Caller holds the condition."""
        self._seq += 1
        job.seq = self._seq
        if job.persistent:
            self._jobs[job.id] = job
        self._pending.append(job)
        self._cond.notify_all()

    # ---------------------------------------------------------------- status

    def status(self, task_id: str) -> dict | None:
        with self._cond:
            job = self._jobs.get(task_id)
            if job is not None:
                if job.terminal is not None:
                    status, error = job.terminal
                    return {"task_id": job.id, "status": status, "operation": job.operation,
                            "created_at": job.created_at, "error": error}
                return {
                    "task_id": job.id,
                    "status": PROCESSING if job.started else PENDING,
                    "operation": job.operation,
                    "created_at": job.created_at,
                    "queue_position": sum(
                        1 for other in self._jobs.values() if other.seq < job.seq
                    ),
                    "eta_seconds": self._eta_locked(job),
                }
        row = self.store.get(task_id)
        if row is None:
            return None
        if row["completed_at"] and row["completed_at"] < time.time() - self.ttl:
            return None  # expired: the sweeper deletes it; never served again
        data = {
            "task_id": row["id"],
            "status": row["status"],
            "operation": row["operation"],
            "created_at": row["created_at"],
        }
        if row["completed_at"]:
            data["completed_at"] = row["completed_at"]
        if row["status"] == COMPLETED and row["result_path"]:
            try:
                data["result"] = self.store.read_result(row["id"], row["result_path"])
            except FileNotFoundError:
                return None
            except Exception as exc:  # rotated key, corrupt or truncated file
                log.error("ocr_result_unreadable", task_id=row["id"], error=str(exc))
                data["status"] = FAILED
                data["error"] = f"OCR result unreadable: {exc}"
        if row["status"] == FAILED and row["error"]:
            data["error"] = row["error"]
        if row["status"] in (PENDING, PROCESSING):
            # In the ledger but not in this process's queue (a second process);
            # report what is known.
            data["queue_position"] = None
            data["eta_seconds"] = None
        return data

    def _eta_locked(self, job: _Job) -> int:
        mine = max(job.remaining_pages(), 1)
        if job in self._active:
            # Round-robin: until this job finishes, every active job gets up to
            # as many pages as this one still needs.
            work = sum(min(other.remaining_pages(), mine) for other in self._active)
        else:
            ahead = [other for other in (*self._active, *self._pending) if other.seq < job.seq]
            work = sum(other.remaining_pages() for other in ahead) + mine
        return int(math.ceil(work * self._page_seconds / self.workers))

    def _retry_after_locked(self) -> int:
        work = sum(job.remaining_pages() for job in (*self._active, *self._pending))
        return int(min(300, max(5, math.ceil(work * self._page_seconds / self.workers))))

    # --------------------------------------------------------------- workers

    def _next_unit_locked(self):
        """Next (job, unit, first) to run; unit None means "finish this job"."""
        while len(self._active) < self.max_active and self._pending:
            self._active.append(self._pending.popleft())
        # A job with nothing left to run (no pages requested, every page
        # recorded before a restart, or cancelled) is finished at once.
        for job in self._active:
            if not job.units and job.inflight == 0 and not job.finishing:
                job.finishing = True
                self._active.remove(job)
                return job, None, False
        count = len(self._active)
        for offset in range(count):
            index = (self._rr + offset) % count
            job = self._active[index]
            if job.units and job.error is None:
                self._rr = index + 1
                unit = job.units.popleft()
                job.inflight += 1
                first = not job.started
                job.started = True
                return job, unit, first
        return None

    def _work(self):
        _worker.active = True
        while True:
            with self._cond:
                picked = None
                while not self._stopped:
                    picked = self._next_unit_locked()
                    if picked is not None:
                        break
                    self._cond.wait()
                if self._stopped:
                    return
            job, unit, first = picked
            try:
                if unit is None:
                    self._finish(job)
                else:
                    self._run_and_record(job, unit, first)
            except BaseException as exc:  # noqa: BLE001 - a worker never dies
                log.error("ocr_worker_error", task_id=job.id, error=str(exc))
                self._fail_job(job, exc)

    def _run_and_record(self, job: _Job, unit: int, first: bool):
        with self._cond:
            self._running += 1
            self.max_running = max(self.max_running, self._running)
        result, error = None, None
        started = time.monotonic()
        try:
            if first and job.persistent:
                self.store.mark_processing(job.id)
            result = self._run_unit(job, unit)
            # Record before the job can be seen as finished, so a checkpoint
            # never lands after the job's rows were cleared.
            if job.persistent and job.backend == "rapid" and not self._stopped:
                self.store.save_unit(job.id, unit, result)
        except BaseException as exc:  # noqa: BLE001 - recorded on the job
            error = exc
        elapsed = time.monotonic() - started
        with self._cond:
            self._running -= 1
            if self._stopped:
                return
            job.inflight -= 1
            if error is not None:
                job.error = job.error or error
                job.units.clear()
            else:
                job.done[unit] = result
                pages = 1 if job.backend == "rapid" else max(job.page_count, 1)
                self._page_seconds = 0.8 * self._page_seconds + 0.2 * (elapsed / pages)
            # The last unit back hands the job to the scheduler to finish.
            self._cond.notify_all()

    def _job_bytes(self, job: _Job) -> bytes:
        if isinstance(job.source, (bytes, bytearray)):
            return job.source
        data = job.data
        if data is None:
            data = job.data = self.store.read_input(job.id, job.source)
        return data

    def _run_unit(self, job: _Job, unit: int):
        data = self._job_bytes(job)
        if job.backend == "rapid":
            return rapid_ocr.read_page(
                data, unit, job.params["dpi"], job.params.get("force_ocr", False)
            )
        from app.services import pdf_service

        return pdf_service.ocr_pages(
            data, job.params["pages"], job.params["language"], job.params["dpi"],
            backend="paddle",
        )

    def _assemble(self, job: _Job) -> dict:
        if job.backend == "rapid":
            return rapid_ocr.assemble(
                self._job_bytes(job), [job.done[unit] for unit in job.targets],
                job.params["dpi"], with_pdf=job.params.get("with_pdf", True),
            )
        return job.done[_WHOLE_DOCUMENT]

    def _finish(self, job: _Job):
        try:
            if job.error is not None:
                raise job.error
            result = self._assemble(job)
            if not job.persistent:
                job.result = result
                job.event.set()
                return
            result = self._ocr_response(job, result)
            if self._stopped:
                return
            self.store.complete(job.id, result)
        except BaseException as exc:  # noqa: BLE001 - becomes the task's error
            self._fail_job(job, exc)
            return
        finally:
            job.data = None
        with self._cond:
            self._jobs.pop(job.id, None)
        log.info("task_completed", task_id=job.id)

    def _fail_job(self, job: _Job, exc: BaseException):
        """Record a job as failed; never raises."""
        reason = str(exc) or type(exc).__name__
        if isinstance(exc, OSError) and exc.errno is not None:
            reason = f"{errno.errorcode.get(exc.errno, exc.errno)}: {reason}"
        job.data = None
        with self._cond:
            if job in self._active:
                self._active.remove(job)
            job.units.clear()
            job.finishing = True
            stopped = self._stopped
        if not job.persistent:
            job.error = job.error or exc
            job.event.set()
            return
        if stopped:
            return
        try:
            self.store.fail(job.id, reason)
        except BaseException as store_exc:  # noqa: BLE001 - keep the outcome in memory
            log.error("ocr_ledger_write_failed", task_id=job.id, error=str(store_exc))
            with self._cond:
                job.terminal = (FAILED, reason)
            return
        with self._cond:
            self._jobs.pop(job.id, None)
        log.error("task_failed", task_id=job.id, error=reason)

    def _ocr_response(self, job: _Job, result: dict) -> dict:
        return {
            "pages": result["pages"],
            "total_words": result["total_words"],
            "avg_confidence": result["avg_confidence"],
            "pdf_base64": base64.b64encode(result["pdf_bytes"]).decode("ascii"),
            "provenance": result.get("provenance"),
            "coverage": result.get("coverage"),
        }

    # ----------------------------------------------------------------- tests

    def wait(self, task_id: str, timeout: float = 30.0) -> dict | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            data = self.status(task_id)
            if data is None or data["status"] in (COMPLETED, FAILED):
                return data
            time.sleep(0.01)
        raise TimeoutError(f"OCR task {task_id} did not finish in {timeout}s")


def cache_key(cache_hash: str, cache_params: dict | None) -> str:
    """Opaque dedupe key: content hash plus every parameter that shapes a result."""
    import hashlib

    material = json.dumps({"hash": cache_hash, "params": cache_params}, sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()


def _as_pdf_bytes(data: bytes) -> bytes:
    """Spill a PDF as is; convert an image (TAX-5613) once, at submission."""
    import fitz

    from app.services.pdf_service import _open_pdf

    with rapid_ocr.FITZ_LOCK:
        try:
            with fitz.open(stream=data, filetype="pdf") as doc:
                if doc.is_pdf:
                    return data
        except Exception:
            return data  # unreadable: planning fails the task with the reason
        with _open_pdf(data) as doc:
            return doc.tobytes()


def _page_count(data: bytes) -> int:
    import fitz

    with rapid_ocr.FITZ_LOCK:
        with fitz.open(stream=data, filetype="pdf") as doc:
            return len(doc)


# --------------------------------------------------------------- singleton

_queue: OcrQueue | None = None
_queue_lock = threading.Lock()


def get_queue() -> OcrQueue:
    """The process's queue, started (with recovery) on first use."""
    global _queue
    with _queue_lock:
        if _queue is None:
            workers = rapid_ocr.worker_count()
            max_active = settings.ocr_max_active_jobs or max(2, workers)
            _queue = OcrQueue(
                Path(settings.ocr_queue_dir), workers, max_active,
                settings.ocr_queue_max_jobs, settings.ocr_result_ttl_seconds,
                sweep_interval=settings.ocr_sweep_interval_seconds,
                sync_timeout=settings.ocr_sync_timeout_seconds,
            ).start()
        return _queue


def shutdown():
    global _queue
    with _queue_lock:
        queue, _queue = _queue, None
    if queue is not None:
        queue.shutdown()


def lookup(task_id: str) -> dict | None:
    return get_queue().status(task_id)
