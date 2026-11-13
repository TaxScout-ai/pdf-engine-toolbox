"""File-hash based cache for expensive PDF operations.

Caches results of CPU-intensive operations (OCR, thumbnails, render,
classify, detect-blank, office conversion) keyed by SHA-256 of the
source content + operation parameters. Results are stored on disk.

This avoids re-processing the same document when the same operation
is requested multiple times (e.g., regenerating thumbnails after
navigating away and back).
"""

import hashlib
import fcntl
import math
from contextlib import contextmanager
from functools import wraps
import json
import os
import threading
import time
from typing import Any

import structlog

from app.config import settings

log = structlog.get_logger()


def _ensure_cache_dir() -> str:
    """Create cache directory if it doesn't exist."""
    os.makedirs(settings.cache_dir, exist_ok=True)
    return settings.cache_dir


# Linux runtime: the directory inode is the common lock across worker processes.
# RLock also serializes threads; nested put -> sweep calls reuse the same flock.
_cache_lock = threading.RLock()
_lock_state = threading.local()


@contextmanager
def _cache_guard():
    with _cache_lock:
        if getattr(_lock_state, 'held', False):
            yield
            return
        fd = os.open(_ensure_cache_dir(), os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            _lock_state.held = True
            yield
        finally:
            _lock_state.held = False
            os.close(fd)


def _serialized(default=None):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if not settings.cache_enabled:
                return default
            try:
                with _cache_guard():
                    return fn(*args, **kwargs)
            except OSError as exc:
                log.warning('cache_io_failed', error_type=type(exc).__name__)
                return default
        return wrapped
    return decorate


def _meta_expired(meta, now):
    if not isinstance(meta, dict):
        return True
    try:
        created = float(meta.get('created_at', 0))
        ttl = float(meta.get('ttl', settings.cache_ttl_seconds))
    except (TypeError, ValueError, OverflowError):
        return True
    return (not math.isfinite(created) or not math.isfinite(ttl)
            or created < 0 or ttl < 0 or now - created > ttl)


def _cache_key(content_hash: str, operation: str, params: dict | None = None) -> str:
    """Build a deterministic cache key from content hash + operation + params."""
    key_parts = f"{content_hash}:{operation}"
    if params:
        # Sort keys for deterministic ordering
        key_parts += ":" + json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(key_parts.encode()).hexdigest()


def content_hash(data: bytes) -> str:
    """Compute SHA-256 hash of raw content bytes."""
    return hashlib.sha256(data).hexdigest()


@_serialized()
def get_cached(
    source_hash: str,
    operation: str,
    params: dict | None = None,
) -> bytes | dict | None:
    """Look up a cached result.

    Returns:
        Cached bytes (for binary results) or dict (for JSON results),
        or None if not cached / expired / cache disabled.
    """
    if not settings.cache_enabled:
        return None

    cache_dir = _ensure_cache_dir()
    key = _cache_key(source_hash, operation, params)
    data_path = os.path.join(cache_dir, key + ".dat")
    meta_path = os.path.join(cache_dir, key + ".meta")

    if not os.path.exists(meta_path) or not os.path.exists(data_path):
        return None

    # Check TTL
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except (ValueError, UnicodeError, OSError):
        return None

    if _meta_expired(meta, time.time()):
        # Expired - clean up
        _remove_entry(data_path, meta_path)
        return None

    # Read cached data
    try:
        with open(data_path, "rb") as f:
            raw = f.read()
    except OSError:
        return None

    result_type = meta.get("type", "bytes")
    if result_type == "json":
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    log.debug("cache_hit", operation=operation, key=key[:12])
    return raw


@_serialized()
def put_cached(
    source_hash: str,
    operation: str,
    result: bytes | dict,
    params: dict | None = None,
    ttl: int | None = None,
) -> None:
    """Store a result in the cache.

    Args:
        source_hash: SHA-256 of source content
        operation: Operation name (e.g. "thumbnails", "ocr", "classify")
        result: Bytes (binary) or dict (JSON-serializable)
        params: Operation parameters for cache key
        ttl: Override TTL in seconds (defaults to config)
    """
    if not settings.cache_enabled:
        return

    cache_dir = _ensure_cache_dir()
    key = _cache_key(source_hash, operation, params)
    data_path = os.path.join(cache_dir, key + ".dat")
    meta_path = os.path.join(cache_dir, key + ".meta")

    result_type = "bytes"
    if isinstance(result, dict):
        raw = json.dumps(result).encode()
        result_type = "json"
    elif isinstance(result, list):
        raw = json.dumps(result).encode()
        result_type = "json"
    else:
        raw = result

    meta = {
        "created_at": time.time(),
        "ttl": ttl or settings.cache_ttl_seconds,
        "type": result_type,
        "operation": operation,
        "size": len(raw),
    }

    try:
        with open(data_path, "wb") as f:
            f.write(raw)
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        log.debug("cache_put", operation=operation, key=key[:12], size=len(raw))
    except OSError as e:
        log.warning("cache_write_failed", error_type=type(e).__name__)

    # Prune if over size limit
    _prune_if_needed(cache_dir)
    sweep_if_due()


def _remove_entry(data_path: str, meta_path: str) -> None:
    """Remove a cache entry (data + meta files)."""
    for path in (data_path, meta_path):
        try:
            os.remove(path)
        except OSError:
            pass


def _prune_if_needed(cache_dir: str) -> None:
    """Remove oldest entries if total cache size exceeds limit."""
    max_bytes = settings.cache_max_size_mb * 1024 * 1024

    try:
        entries = []
        total_size = 0
        for fname in os.listdir(cache_dir):
            if not fname.endswith(".dat"):
                continue
            path = os.path.join(cache_dir, fname)
            stat = os.stat(path)
            entries.append((path, stat.st_mtime, stat.st_size))
            total_size += stat.st_size

        if total_size <= max_bytes:
            return

        # Sort by modification time (oldest first)
        entries.sort(key=lambda e: e[1])

        for path, _, size in entries:
            if total_size <= max_bytes:
                break
            meta_path = path.replace(".dat", ".meta")
            _remove_entry(path, meta_path)
            total_size -= size
            log.debug("cache_pruned", path=os.path.basename(path))
    except OSError:
        pass


_sweep_lock = threading.Lock()
_last_sweep_at = 0.0


def _entry_expired(meta_path: str, now: float) -> bool:
    """True when the entry's TTL has passed or its meta cannot be read."""
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except (ValueError, UnicodeError, OSError):
        return True
    return _meta_expired(meta, now)


@_serialized(default=0)
def sweep_expired(now: float | None = None) -> int:
    """Delete every expired entry from disk (TAX-9528, audit T12).

    The TTL used to be applied only when the same key was read again, so the
    page images and thumbnails of a document nobody opened again stayed on
    disk until the size cap pushed them out. This removes:
      * entries whose TTL has passed (data + meta);
      * entries whose meta is unreadable;
      * data files without a meta file and meta files without a data file,
        once they are older than the default TTL (a write in progress has
        both files within milliseconds).

    Returns the number of files removed. Never raises.
    """
    if not os.path.isdir(settings.cache_dir):
        return 0
    now = time.time() if now is None else now
    removed = 0
    try:
        names = os.listdir(settings.cache_dir)
    except OSError:
        return 0
    present = set(names)
    for fname in names:
        path = os.path.join(settings.cache_dir, fname)
        if fname.endswith(".meta"):
            data_path = path[: -len(".meta")] + ".dat"
            if fname[: -len(".meta")] + ".dat" not in present:
                if _older_than(path, now, settings.cache_ttl_seconds):
                    removed += _remove_counted(path)
                continue
            if _entry_expired(path, now):
                removed += _remove_counted(data_path) + _remove_counted(path)
        elif fname.endswith(".dat"):
            if fname[: -len(".dat")] + ".meta" not in present and _older_than(
                path, now, settings.cache_ttl_seconds
            ):
                removed += _remove_counted(path)
    if removed:
        log.info("cache_swept", removed_files=removed)
    return removed


def sweep_if_due(now: float | None = None) -> int:
    """Run `sweep_expired` when the sweep interval has passed since the last one."""
    global _last_sweep_at
    if not settings.cache_enabled:
        return 0
    now = time.time() if now is None else now
    if now - _last_sweep_at < settings.cache_sweep_interval_seconds:
        return 0
    if not _sweep_lock.acquire(blocking=False):
        return 0
    try:
        _last_sweep_at = now
        return sweep_expired(now)
    finally:
        _sweep_lock.release()


def _older_than(path: str, now: float, seconds: float) -> bool:
    try:
        return now - os.stat(path).st_mtime > seconds
    except OSError:
        return False


def _remove_counted(path: str) -> int:
    try:
        os.remove(path)
        return 1
    except FileNotFoundError:
        return 0
    except OSError:
        return 0
