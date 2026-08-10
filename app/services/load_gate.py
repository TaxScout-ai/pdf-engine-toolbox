"""Say "busy" when busy, instead of failing as if the document were broken.

── Why this exists ─────────────────────────────────────────────────────────

Eleven documents in a corpus run failed with `invalid status code: 500`. They
shared no size, type or content: a four-page engagement letter sat in the set
beside a consolidated 1099. Replayed against an idle engine every one of them
returned 200. The failures clustered in the same window as 155 watchdog
starvations — one 258-document burst, one cause.

The caller cannot tell that from a 500. A 500 reads as "this document is
broken", so it retries into the same wall, which is exactly what happened.

Measured downstream on 2026-08-10: of the 25 OCR failures in production that
record a reason, roughly 19 are the engine being unavailable, unreachable, or
so busy that a 100 KB document hit a five-minute timeout. Documents are not the
problem; contention is.

── What this does, and what it deliberately does not ───────────────────────

A counter, not a queue. Requests over the limit are refused immediately with
503 and a `Retry-After`, so a caller backs off instead of holding a connection
open behind work that will not start for minutes.

Queueing would be worse here. The expensive endpoints run for minutes, so a
queued request outlives the client's own timeout and finishes into nobody —
paying full inference cost for an answer that is discarded. Refusing early is
cheaper for both sides and, unlike a timeout, it is *legible*.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager

from fastapi import HTTPException

#: How many expensive reads may run at once.
#:
#: Two, not one: a single slot serialises a page sweep behind an unrelated
#: OCR request and doubles the latency a caller sees for no memory benefit.
#: Not many more either — the readers hold the rendered page and a full model
#: forward pass at the same time, and this engine has been OOM-killed on a
#: shared host with a smaller footprint than that suggests.
MAX_CONCURRENT_READS = int(os.environ.get("ENGINE_MAX_CONCURRENT_READS", "2"))

#: What to tell a refused caller to wait. A read takes minutes, so anything
#: shorter invites the same stampede a second later.
RETRY_AFTER_SECONDS = int(os.environ.get("ENGINE_RETRY_AFTER_SECONDS", "60"))

_lock = threading.Lock()
_in_flight = 0


def in_flight() -> int:
    """Reads currently holding a slot. For /health and for tests."""
    with _lock:
        return _in_flight


def acquire_read_slot(operation: str) -> None:
    """Take a slot at accept time, or refuse now with 503.

    Accept time, not run time: the point is to answer the caller before it
    commits to waiting. A request admitted and then queued behind minutes of
    inference outlives its own client timeout and finishes into nobody.
    """
    global _in_flight

    with _lock:
        if _in_flight >= MAX_CONCURRENT_READS:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"{operation}: engine is at capacity "
                    f"({_in_flight}/{MAX_CONCURRENT_READS} reads in flight). "
                    "This is load, not a problem with the document — retry later."
                ),
                headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
            )
        _in_flight += 1


def release_read_slot() -> None:
    """Give the slot back. Must run even when the read raised."""
    global _in_flight

    with _lock:
        # Never below zero: a double release would otherwise let the counter
        # drift negative and silently disable the gate for the process's life.
        _in_flight = max(0, _in_flight - 1)


@contextmanager
def reserve_read_slot(operation: str):
    """Hold a slot for the duration of an expensive read, or refuse now.

    Raises ``HTTPException(503)`` rather than blocking. Blocking is what the
    engine already did implicitly, and it is what turned load into 500s and
    five-minute timeouts on small documents.
    """
    global _in_flight

    with _lock:
        if _in_flight >= MAX_CONCURRENT_READS:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"{operation}: engine is at capacity "
                    f"({_in_flight}/{MAX_CONCURRENT_READS} reads in flight). "
                    "This is load, not a problem with the document — retry later."
                ),
                headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
            )
        _in_flight += 1

    try:
        yield
    finally:
        with _lock:
            _in_flight -= 1
