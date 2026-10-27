"""Mixed-lane scheduling must bound work and preserve persistent progress."""

import pytest

from app.services.ocr_queue import OcrQueue, _Job


def job(persistent, pages=3):
    return _Job(operation="rapid_ocr" if persistent else "tables", backend="rapid",
                params={}, source=b"fixture", persistent=persistent,
                targets=list(range(pages)), page_count=pages)


def enqueue(queue, item):
    with queue._cond:
        queue._enqueue(item)


def finish_page(picked):
    item, unit, _ = picked
    assert unit is not None
    item.inflight -= 1
    item.done[unit] = {"page_index": unit}


@pytest.mark.parametrize("slots", [1, 2, 4])
def test_sync_starts_at_page_boundary_before_backlog_and_ocr_progresses(tmp_path, slots):
    queue = OcrQueue(tmp_path, workers=2, max_active=slots, max_jobs=50,
                     ttl=3600, key=b"a" * 32)
    persistent = [job(True, pages=5) for _ in range(20)]
    for item in persistent:
        enqueue(queue, item)
    first = queue._next_unit_locked()
    first[0].data = b"loaded input"
    sync = [job(False), job(False)]
    for item in sync:
        enqueue(queue, item)
    finish_page(first)
    sequence = []
    completed = set()
    for _ in range(160):
        picked = queue._next_unit_locked()
        assert picked is not None
        item, unit, _ = picked
        assert len(queue._active) <= slots
        if unit is None:
            completed.add(item)
        else:
            sequence.append(item)
            finish_page(picked)
        if len(completed) == 22:
            break
    assert len(completed) == 22
    assert sequence.index(sync[0]) < 5, "interactive work waited for the OCR backlog"
    assert any(item.persistent for item in sequence[:sequence.index(sync[1]) + 1])
    assert all(sorted(item.done) == list(range(5)) for item in persistent)
    assert all(sorted(item.done) == [0, 1, 2] for item in sync)


def test_yield_waits_for_all_inflight_pages_and_releases_input(tmp_path):
    queue = OcrQueue(tmp_path, workers=2, max_active=1, max_jobs=50,
                     ttl=3600, key=b"a" * 32)
    persistent = job(True, pages=5)
    persistent.data = b"loaded input"
    enqueue(queue, persistent)
    page1 = queue._next_unit_locked()
    page2 = queue._next_unit_locked()
    sync = job(False)
    enqueue(queue, sync)
    assert queue._next_unit_locked() is None
    finish_page(page1)
    assert queue._next_unit_locked() is None
    assert persistent.data is not None
    finish_page(page2)
    assert queue._next_unit_locked()[0] is sync
    assert persistent.data is None
    assert persistent in queue._pending
    assert list(persistent.units) == [2, 3, 4]


def test_cancelled_sync_waiter_does_not_leave_ocr_unschedulable(tmp_path):
    queue = OcrQueue(tmp_path, workers=2, max_active=1, max_jobs=50,
                     ttl=3600, key=b"a" * 32)
    persistent = job(True)
    enqueue(queue, persistent)
    page = queue._next_unit_locked()
    sync = job(False)
    enqueue(queue, sync)
    assert queue._next_unit_locked() is None
    # _run_sync removes a timed-out pending request under the condition.
    queue._pending.remove(sync)
    finish_page(page)
    assert queue._next_unit_locked()[0] is persistent


def test_persistent_work_progresses_during_continuous_sync_arrivals(tmp_path):
    queue = OcrQueue(tmp_path, workers=2, max_active=2, max_jobs=50,
                     ttl=3600, key=b"a" * 32)
    persistent = job(True, pages=30)
    enqueue(queue, persistent)
    for _ in range(100):
        enqueue(queue, job(False, pages=1))
        picked = queue._next_unit_locked()
        assert len(queue._active) <= 2
        if picked[1] is not None:
            finish_page(picked)
    assert len(persistent.done) >= 20


@pytest.mark.parametrize("recovered", [False, True])
def test_single_worker_finishes_just_admitted_empty_job(tmp_path, recovered):
    queue = OcrQueue(tmp_path, workers=1, max_active=1, max_jobs=50,
                     ttl=3600, key=b"a" * 32)
    item = job(True, pages=1 if recovered else 0)
    if recovered:
        item.done[0] = {"page_index": 0}
        item.units.clear()
    enqueue(queue, item)
    assert queue._next_unit_locked() == (item, None, False)
    assert item.finishing


def test_mixed_ready_pages_give_two_starts_to_sync_then_one_to_persistent(tmp_path):
    queue = OcrQueue(tmp_path, workers=2, max_active=2, max_jobs=50,
                     ttl=3600, key=b"a" * 32)
    enqueue(queue, job(True, pages=30))
    enqueue(queue, job(False, pages=30))
    lane = []
    for _ in range(18):
        picked = queue._next_unit_locked()
        lane.append(picked[0].persistent)
        finish_page(picked)
    assert lane == [False, False, True] * 6


def test_short_ocr_job_is_not_starved_by_weighted_sync_and_long_ocr(tmp_path):
    queue = OcrQueue(tmp_path, workers=2, max_active=3, max_jobs=50,
                     ttl=3600, key=b"a" * 32)
    long = job(True, pages=100)
    short = job(True, pages=1)
    sync = job(False, pages=100)
    queue._active = [long, short, sync]
    starts = []
    for _ in range(6):
        picked = queue._next_unit_locked()
        starts.append(picked[0])
        finish_page(picked)
    assert starts.count(long) == 1
    assert starts.count(short) == 1
    assert starts.count(sync) == 4


def test_persistent_cursor_survives_sync_lane_inflight_gap(tmp_path):
    queue = OcrQueue(tmp_path, workers=2, max_active=3, max_jobs=50,
                     ttl=3600, key=b"a" * 32)
    long = job(True, pages=100)
    short = job(True, pages=1)
    sync = job(False, pages=1)
    queue._active = [long, short, sync]
    sync_page = queue._next_unit_locked()
    assert sync_page[0] is sync
    long_page = queue._next_unit_locked()
    assert long_page[0] is long
    # A new synchronous job takes the same active position after the old
    # one finishes, while the long OCR page remains in flight.
    finish_page(sync_page)
    assert queue._next_unit_locked()[1] is None
    enqueue(queue, job(False, pages=1))
    next_sync = queue._next_unit_locked()
    assert not next_sync[0].persistent
    finish_page(long_page)
    assert queue._next_unit_locked()[0] is short
