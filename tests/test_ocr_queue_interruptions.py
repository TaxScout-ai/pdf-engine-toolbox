"""Actual process interruptions, with fake inference and isolated sealed inputs."""

import json
import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path

import fitz
import pytest

from app.services.ocr_queue import OcrQueue

KEY = b"i" * 32
ROOT = Path(__file__).resolve().parents[1]
CHILD = """
import json, os, signal, sys
from pathlib import Path
import fitz
from app.services.ocr_queue import OcrQueue

root, task, mode = sys.argv[1:]
def read(self, job, unit):
    print(f'page:{job.id}:{unit}', flush=True)
    if mode == 'page' and job.id == task and unit == 1:
        os.kill(os.getpid(), signal.SIGKILL)
    return {'page': unit + 1}
def assemble(self, job):
    print(f'assemble:{job.id}', flush=True)
    if mode == 'assembly' and job.id == task:
        os.kill(os.getpid(), signal.SIGKILL)
    return {'pages': [job.done[i] for i in job.targets], 'total_words': 0,
            'avg_confidence': 0, 'pdf_bytes': b''}
OcrQueue._run_unit = read
OcrQueue._assemble = assemble
queue = OcrQueue(Path(root), 1, 1, 10, 3600, key=b'i' * 32).start()
result = queue.wait(task, timeout=10)
# The failed/recovered document must release its only active slot.
with fitz.open() as doc:
    doc.new_page()
    another = queue.submit_ocr(doc.tobytes(), None, 'en', 150)
following = queue.wait(another, timeout=10)
print(json.dumps({'task': result, 'following': following}), flush=True)
queue.shutdown()
"""


def queue_at(root):
    return OcrQueue(root, 1, 1, 10, 3600, key=KEY)


def submit(queue, pages=3):
    with fitz.open() as doc:
        for _ in range(pages):
            doc.new_page()
        return queue.submit_ocr(doc.tobytes(), None, "en", 150)


def child(root, task, mode):
    return subprocess.run(
        [sys.executable, "-c", CHILD, str(root), task, mode], cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True, text=True, timeout=20,
    )


def receipt(process):
    assert process.returncode == 0, process.stdout + process.stderr
    return next(json.loads(line) for line in process.stdout.splitlines()
                if line.startswith('{"task":'))


@pytest.mark.parametrize("stage,unit,label", [("page", 1, "page 2"),
                                             ("assembly", -1, "result assembly")])
def test_repeated_sigkill_is_bounded_and_terminal_survives_restart(tmp_path, stage, unit, label):
    root = tmp_path / "queue"
    queue = queue_at(root)
    try:
        task = submit(queue)
        for attempt in range(1, 4):
            killed = child(root, task, stage)
            assert killed.returncode == -signal.SIGKILL, killed.stdout + killed.stderr
            if attempt > 1:
                assert f"page:{task}:0" not in killed.stdout  # checkpoint never rerun
            row = queue.store._execute(
                "SELECT attempts FROM ocr_job_attempts WHERE job_id=? AND unit=?", (task, unit)
            )[0]
            assert row["attempts"] == attempt
            assert queue.store.get(task)["status"] == "processing"
            assert set(queue.store.units(task)) == ({0} if stage == "page" else {0, 1, 2})

        recovered = child(root, task, stage)
        # No fourth attempt even when the same operation would SIGKILL again.
        if stage == "page":
            assert f"page:{task}:1" not in recovered.stdout
        else:
            # Only the following clean task reaches assembly.
            assert f"assemble:{task}" not in recovered.stdout
        result = receipt(recovered)
        assert result["task"]["status"] == "failed"
        assert f"OCR {label} could not finish after 3 attempts" in result["task"]["error"]
        assert result["following"]["status"] == "completed"
        assert queue.store.units(task) == {}
        assert queue.store._execute("SELECT * FROM ocr_job_attempts WHERE job_id=?", (task,)) == []
        assert queue.store.get(task)["input_path"] is None
        assert not (root / "inputs" / f"{task}.bin").exists()
        assert not (root / "results" / f"{task}.bin").exists()
        assert receipt(child(root, task, "success"))["task"]["status"] == "failed"
    finally:
        queue.store._db.close()


def test_third_attempt_can_complete_and_preserves_completed_pages(tmp_path):
    queue = queue_at(tmp_path / "queue")
    try:
        task = submit(queue, pages=6)
        for _ in range(2):
            assert child(queue.root, task, "page").returncode == -signal.SIGKILL
        success = child(queue.root, task, "success")
        assert f"page:{task}:0" not in success.stdout
        result = receipt(success)
        assert result["task"]["status"] == "completed"
        assert [p["page"] for p in result["task"]["result"]["pages"]] == list(range(1, 7))
        assert result["following"]["status"] == "completed"
        assert queue.store._execute("SELECT * FROM ocr_job_attempts") == []
    finally:
        queue.store._db.close()


def test_checkpoint_and_attempt_cleanup_are_atomic(tmp_path):
    queue = queue_at(tmp_path / "queue")
    try:
        task = submit(queue)
        assert queue.store._execute("SELECT * FROM ocr_job_attempts") == []
        queue.store.begin_unit(task, 0)
        queue.store._execute(
            "CREATE TRIGGER refuse_attempt_cleanup BEFORE DELETE ON ocr_job_attempts "
            "BEGIN SELECT RAISE(ABORT, 'interrupted checkpoint'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="interrupted checkpoint"):
            queue.store.save_unit(task, 0, {"page": 1})
        assert queue.store.units(task) == {}
        assert queue.store._execute("SELECT attempts FROM ocr_job_attempts")[0][0] == 1
        queue.store._execute("DROP TRIGGER refuse_attempt_cleanup")
        queue.store.save_unit(task, 0, {"page": 1})
        assert queue.store.units(task) == {0: {"page": 1}}
        assert queue.store._execute("SELECT * FROM ocr_job_attempts") == []
    finally:
        queue.store._db.close()


def test_legacy_ledger_upgrade_and_orphan_cleanup(tmp_path):
    root = tmp_path / "queue"
    before = queue_at(root)
    task = submit(before)
    before.store.save_unit(task, 0, {"page": 1})
    before.store._execute("DROP TABLE ocr_job_attempts")  # ledger before this upgrade
    before.store._db.close()
    after = queue_at(root)
    try:
        assert after.store.units(task) == {0: {"page": 1}}
        assert after.store._execute("SELECT * FROM ocr_job_attempts") == []
        after.store._execute("INSERT INTO ocr_job_attempts VALUES ('orphan', 0, 3)")
        assert after.store.sweep(3600)["attempt_rows"] == 1
        assert receipt(child(root, task, "success"))["task"]["status"] == "completed"
    finally:
        after.store._db.close()
