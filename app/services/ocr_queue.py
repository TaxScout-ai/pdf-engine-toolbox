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
  ``OCR_QUEUE_DIR/inputs/<id>.pdf``; its state lives in
  ``OCR_QUEUE_DIR/jobs.sqlite3``. Each finished RapidOCR page is recorded, so
  after a restart ``pending`` and ``processing`` jobs are re-queued in their
  original order and resume from the first unfinished page. Results are
  written to ``OCR_QUEUE_DIR/results/<id>.json`` and kept for
  ``OCR_RESULT_TTL_SECONDS``.
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

from app.config import settings
from app.services import rapid_ocr

log = structlog.get_logger()

PENDING, PROCESSING, COMPLETED, FAILED = "pending", "processing", "completed", "failed"
_WHOLE_DOCUMENT = -1
# Seconds per page assumed before any page has been timed in this process.
_INITIAL_PAGE_SECONDS = 4.0

_worker = threading.local()


class QueueFullError(Exception):
    """The queue holds OCR_QUEUE_MAX_JOBS unfinished jobs; retry later."""

    def __init__(self, retry_after: int):
        super().__init__(f"OCR queue is full; retry after {retry_after}s")
        self.retry_after = retry_after


class _Stopped(Exception):
    pass


class _Store:
    """SQLite job ledger. One connection, serialized by a lock."""

    def __init__(self, root: Path):
        self.root = root
        self.inputs = root / "inputs"
        self.results = root / "results"
        for path in (self.inputs, self.results):
            path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(
            root / "jobs.sqlite3", timeout=10, isolation_level=None, check_same_thread=False
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
                completed_at REAL
            );
            CREATE INDEX IF NOT EXISTS ocr_jobs_status ON ocr_jobs (status, seq);
            CREATE TABLE IF NOT EXISTS ocr_job_units (
                job_id TEXT NOT NULL,
                unit INTEGER NOT NULL,
                result TEXT NOT NULL,
                PRIMARY KEY (job_id, unit)
            );
            """
        )

    def close(self):
        with self._lock:
            self._db.close()

    def _execute(self, sql, args=()):
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def insert(self, job: "_Job", status: str, error: str | None = None) -> int:
        now = time.time()
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO ocr_jobs (id, operation, backend, status, params, page_count,"
                " input_path, error, created_at, completed_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    job.id, job.operation, job.backend, status, json.dumps(job.params),
                    job.page_count, str(job.source) if status != FAILED else None, error,
                    job.created_at, now if status == FAILED else None,
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
        self._execute(
            "INSERT OR REPLACE INTO ocr_job_units (job_id, unit, result) VALUES (?,?,?)",
            (job_id, unit, json.dumps(result)),
        )

    def units(self, job_id: str) -> dict[int, dict]:
        rows = self._execute("SELECT unit, result FROM ocr_job_units WHERE job_id = ?", (job_id,))
        return {row["unit"]: json.loads(row["result"]) for row in rows}

    def complete(self, job_id: str, result: dict):
        path = self.results / f"{job_id}.json"
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as handle:
            json.dump(result, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        self._finish(job_id, COMPLETED, result_path=str(path))

    def fail(self, job_id: str, error: str):
        self._finish(job_id, FAILED, error=error)

    def _finish(self, job_id, status, result_path=None, error=None):
        rows = self._execute("SELECT input_path FROM ocr_jobs WHERE id = ?", (job_id,))
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            self._db.execute(
                "UPDATE ocr_jobs SET status = ?, result_path = ?, error = ?, completed_at = ?,"
                " input_path = NULL WHERE id = ?",
                (status, result_path, error, time.time(), job_id),
            )
            self._db.execute("DELETE FROM ocr_job_units WHERE job_id = ?", (job_id,))
            self._db.execute("COMMIT")
        for row in rows:
            _unlink(row["input_path"])

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


def _unlink(path):
    if path:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


class _Job:
    __slots__ = (
        "id", "operation", "backend", "params", "source", "persistent", "targets",
        "units", "done", "inflight", "error", "created_at", "seq", "page_count",
        "started", "event", "result", "completed_at",
    )

    def __init__(self, *, operation, backend, params, source, persistent, targets,
                 page_count, job_id=None, created_at=None, done=None):
        self.id = job_id or str(uuid.uuid4())
        self.operation = operation
        self.backend = backend
        self.params = params
        self.source = source
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
        self.completed_at = None

    def remaining_pages(self) -> int:
        if self.backend == "rapid":
            return len(self.units) + self.inflight
        return self.page_count if (self.units or self.inflight) else 0


class OcrQueue:
    def __init__(self, root: Path, workers: int, max_active: int, max_jobs: int, ttl: int):
        self.root = Path(root)
        self.workers = max(1, workers)
        self.max_active = max(1, max_active)
        self.max_jobs = max(1, max_jobs)
        self.ttl = ttl
        self.store = _Store(self.root)
        self._cond = threading.Condition()
        self._pending: deque[_Job] = deque()
        self._active: list[_Job] = []
        self._jobs: dict[str, _Job] = {}  # unfinished persistent jobs
        self._rr = 0
        self._seq = 0
        self._stopped = False
        self._threads: list[threading.Thread] = []
        self._page_seconds = _INITIAL_PAGE_SECONDS
        self._running = 0
        self.max_running = 0

    # ------------------------------------------------------------------ life

    def start(self) -> "OcrQueue":
        self.store.prune(self.ttl)
        recovered = 0
        for row in self.store.unfinished():
            params = json.loads(row["params"])
            if not row["input_path"] or not Path(row["input_path"]).exists():
                self.store.fail(row["id"], "OCR input was lost before processing finished")
                continue
            done = self.store.units(row["id"]) if row["backend"] == "rapid" else {}
            job = _Job(
                operation=row["operation"], backend=row["backend"], params=params,
                source=row["input_path"], persistent=True, targets=params["targets"],
                page_count=row["page_count"], job_id=row["id"], created_at=row["created_at"],
                done=done,
            )
            with self._cond:
                self._enqueue(job)
            recovered += 1
        if recovered:
            log.info("ocr_queue_recovered", jobs=recovered)
        for index in range(self.workers):
            thread = threading.Thread(
                target=self._work, name=f"ocr-worker-{index}", daemon=True
            )
            thread.start()
            self._threads.append(thread)
        log.info("ocr_queue_started", workers=self.workers, max_active=self.max_active,
                 intra_op_threads=rapid_ocr.intra_op_threads(), root=str(self.root))
        return self

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
                job.error = _Stopped("OCR queue stopped")
                job.event.set()
        for thread in self._threads:
            thread.join(timeout=0.2)

    # ---------------------------------------------------------------- submit

    def submit_ocr(self, pdf_bytes: bytes, pages, language: str, dpi: int, backend: str,
                   cache_hash: str | None = None, cache_params: dict | None = None) -> str:
        """Persist an /text/ocr request and queue it. Returns the task id."""
        self.store.prune(self.ttl)
        with self._cond:
            if len(self._jobs) >= self.max_jobs:
                raise QueueFullError(self._retry_after_locked())
        job_id = str(uuid.uuid4())
        path = self.store.inputs / f"{job_id}.pdf"
        _write_durably(path, _as_pdf_bytes(pdf_bytes))
        params = {
            "pages": pages, "language": language, "requested_dpi": dpi, "dpi": dpi,
            "cache_hash": cache_hash, "cache_params": cache_params,
        }
        error, targets, page_count = None, [_WHOLE_DOCUMENT], len(pages or [])
        try:
            if backend == "rapid":
                targets, params["dpi"] = rapid_ocr.plan(str(path), pages, language, dpi)
                page_count = len(targets)
            else:
                page_count = page_count or _page_count(str(path))
        except Exception as exc:  # the request itself is invalid: fail its task
            error = str(exc)
        params["targets"] = targets
        job = _Job(operation="ocr", backend=backend, params=params, source=str(path),
                   persistent=True, targets=targets, page_count=page_count, job_id=job_id)
        if error is not None:
            self.store.insert(job, FAILED, error)
            _unlink(path)
            log.error("task_failed", task_id=job_id, error=error)
            return job_id
        self.store.insert(job, PENDING)
        with self._cond:
            self._enqueue(job)
        log.info("task_created", task_id=job_id, operation="ocr", backend=backend,
                 pages=page_count)
        return job_id

    def run_rapid(self, pdf_bytes: bytes, pages=None, language="en", dpi=300,
                  force_ocr=False, with_pdf=True) -> dict:
        """OCR in memory through the shared workers and wait for the result."""
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
        job.event.wait()
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
                with open(row["result_path"]) as handle:
                    data["result"] = json.load(handle)
            except FileNotFoundError:
                return None
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
        while len(self._active) < self.max_active and self._pending:
            self._active.append(self._pending.popleft())
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
                self._running += 1
                self.max_running = max(self.max_running, self._running)
            job, unit, first = picked
            if first and job.persistent:
                self.store.mark_processing(job.id)
            started = time.monotonic()
            result, error = None, None
            try:
                result = self._run_unit(job, unit)
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
                finished = job.inflight == 0 and not job.units
                if finished:
                    self._active.remove(job)
                self._cond.notify_all()
            if error is None and job.persistent and job.backend == "rapid":
                self.store.save_unit(job.id, unit, result)
            if finished:
                self._finish(job)

    def _run_unit(self, job: _Job, unit: int):
        if job.backend == "rapid":
            return rapid_ocr.read_page(
                job.source, unit, job.params["dpi"], job.params.get("force_ocr", False)
            )
        from app.services import pdf_service

        data = Path(job.source).read_bytes() if isinstance(job.source, str) else job.source
        return pdf_service.ocr_pages(
            data, job.params["pages"], job.params["language"], job.params["dpi"],
            backend="paddle",
        )

    def _assemble(self, job: _Job) -> dict:
        if job.backend == "rapid":
            return rapid_ocr.assemble(
                job.source, [job.done[unit] for unit in job.targets], job.params["dpi"],
                with_pdf=job.params.get("with_pdf", True),
            )
        return job.done[_WHOLE_DOCUMENT]

    def _finish(self, job: _Job):
        try:
            if job.error is not None:
                raise job.error
            result = self._assemble(job)
            if job.persistent:
                result = self._ocr_response(job, result)
        except BaseException as exc:  # noqa: BLE001 - becomes the task's error
            if job.persistent:
                with self._cond:
                    if self._stopped:
                        return
                self.store.fail(job.id, str(exc))
                with self._cond:
                    self._jobs.pop(job.id, None)
                log.error("task_failed", task_id=job.id, error=str(exc))
            else:
                job.error = exc
                job.event.set()
            return
        if not job.persistent:
            job.result = result
            job.event.set()
            return
        with self._cond:
            if self._stopped:
                return
        self.store.complete(job.id, result)
        with self._cond:
            self._jobs.pop(job.id, None)
        log.info("task_completed", task_id=job.id)

    def _ocr_response(self, job: _Job, result: dict) -> dict:
        from app.services import cache_service

        data = {
            "pages": result["pages"],
            "total_words": result["total_words"],
            "avg_confidence": result["avg_confidence"],
            "pdf_base64": base64.b64encode(result["pdf_bytes"]).decode("ascii"),
            "provenance": result.get("provenance"),
            "coverage": result.get("coverage"),
        }
        if job.params.get("cache_hash"):
            cache_service.put_cached(
                job.params["cache_hash"], "ocr", data, job.params.get("cache_params")
            )
        return data

    # ----------------------------------------------------------------- tests

    def wait(self, task_id: str, timeout: float = 30.0) -> dict | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            data = self.status(task_id)
            if data is None or data["status"] in (COMPLETED, FAILED):
                return data
            time.sleep(0.01)
        raise TimeoutError(f"OCR task {task_id} did not finish in {timeout}s")


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


def _page_count(path: str) -> int:
    import fitz

    with rapid_ocr.FITZ_LOCK:
        with fitz.open(path, filetype="pdf") as doc:
            return len(doc)


def _write_durably(path: Path, data: bytes):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


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
