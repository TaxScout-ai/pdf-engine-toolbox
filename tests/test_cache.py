"""Test the cache service."""

import json
import os
import tempfile
import time

from app.services import cache_service
from app.config import settings


def test_cache_put_and_get_bytes():
    """Should store and retrieve binary data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = settings.cache_dir
        settings.cache_dir = tmpdir
        settings.cache_enabled = True
        try:
            data = b"fake PDF content for cache test"
            src_hash = cache_service.content_hash(data)

            # Cache miss initially
            assert cache_service.get_cached(src_hash, "test_op") is None

            # Store
            cache_service.put_cached(src_hash, "test_op", data)

            # Cache hit
            result = cache_service.get_cached(src_hash, "test_op")
            assert result == data
        finally:
            settings.cache_dir = original_dir


def test_cache_put_and_get_json():
    """Should store and retrieve JSON (dict) data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = settings.cache_dir
        settings.cache_dir = tmpdir
        settings.cache_enabled = True
        try:
            data = {"suggested_label": "W-2", "confidence": 0.9}
            src_hash = cache_service.content_hash(b"some pdf content")

            cache_service.put_cached(src_hash, "classify", data)

            result = cache_service.get_cached(src_hash, "classify")
            assert isinstance(result, dict)
            assert result["suggested_label"] == "W-2"
            assert result["confidence"] == 0.9
        finally:
            settings.cache_dir = original_dir


def test_cache_params_differentiation():
    """Different params should produce different cache entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = settings.cache_dir
        settings.cache_dir = tmpdir
        settings.cache_enabled = True
        try:
            src_hash = cache_service.content_hash(b"test content")

            data_a = b"result with width=200"
            data_b = b"result with width=400"

            cache_service.put_cached(src_hash, "thumbnails", data_a, {"width": 200})
            cache_service.put_cached(src_hash, "thumbnails", data_b, {"width": 400})

            result_a = cache_service.get_cached(src_hash, "thumbnails", {"width": 200})
            result_b = cache_service.get_cached(src_hash, "thumbnails", {"width": 400})

            assert result_a == data_a
            assert result_b == data_b
        finally:
            settings.cache_dir = original_dir


def test_cache_ttl_expiry():
    """Expired entries should return None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = settings.cache_dir
        settings.cache_dir = tmpdir
        settings.cache_enabled = True
        try:
            src_hash = cache_service.content_hash(b"expiring data")

            # Store with a very short TTL
            cache_service.put_cached(src_hash, "short_lived", b"data", ttl=1)

            # Should be cached immediately
            assert cache_service.get_cached(src_hash, "short_lived") == b"data"

            # Wait for expiry
            time.sleep(1.5)

            # Should be expired
            assert cache_service.get_cached(src_hash, "short_lived") is None
        finally:
            settings.cache_dir = original_dir


def test_cache_disabled():
    """When cache is disabled, should always return None."""
    original_enabled = settings.cache_enabled
    settings.cache_enabled = False
    try:
        src_hash = cache_service.content_hash(b"test data")
        cache_service.put_cached(src_hash, "test_op", b"data")
        assert cache_service.get_cached(src_hash, "test_op") is None
    finally:
        settings.cache_enabled = original_enabled


def test_cache_list_data():
    """Should store and retrieve list data (e.g., blank page detection results)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = settings.cache_dir
        settings.cache_dir = tmpdir
        settings.cache_enabled = True
        try:
            data = [
                {"page_index": 0, "is_blank": False, "ink_ratio": 0.15},
                {"page_index": 1, "is_blank": True, "ink_ratio": 0.002},
            ]
            src_hash = cache_service.content_hash(b"pdf with blanks")

            cache_service.put_cached(src_hash, "detect_blank", data)

            result = cache_service.get_cached(src_hash, "detect_blank")
            assert isinstance(result, list)
            assert len(result) == 2
            assert result[1]["is_blank"] is True
        finally:
            settings.cache_dir = original_dir


def test_content_hash_deterministic():
    """Same content should always produce the same hash."""
    data = b"deterministic content for hashing"
    h1 = cache_service.content_hash(data)
    h2 = cache_service.content_hash(data)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex digest


def test_content_hash_different():
    """Different content should produce different hashes."""
    h1 = cache_service.content_hash(b"content A")
    h2 = cache_service.content_hash(b"content B")
    assert h1 != h2


def test_sweep_deletes_expired_entries_that_are_never_read_again():
    """TAX-9528 (audit T12): expiry removes files from disk without a read."""
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = settings.cache_dir
        settings.cache_dir = tmpdir
        settings.cache_enabled = True
        try:
            old = cache_service.content_hash(b"page images nobody opens again")
            fresh = cache_service.content_hash(b"page images still in use")
            cache_service.put_cached(old, "render", b"png-bytes", {"page": 0}, ttl=60)
            cache_service.put_cached(fresh, "render", b"png-bytes", {"page": 0}, ttl=3600)
            assert len(os.listdir(tmpdir)) == 4

            # Ten minutes later: the 60 s entry is past its TTL, the other is not.
            removed = cache_service.sweep_expired(now=time.time() + 600)

            assert removed == 2
            assert cache_service.get_cached(fresh, "render", {"page": 0}) == b"png-bytes"
            remaining = sorted(os.listdir(tmpdir))
            assert len(remaining) == 2
            key = cache_service._cache_key(old, "render", {"page": 0})
            assert key + ".dat" not in remaining and key + ".meta" not in remaining
        finally:
            settings.cache_dir = original_dir


def test_sweep_removes_orphans_and_unreadable_meta():
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = settings.cache_dir
        settings.cache_dir = tmpdir
        settings.cache_enabled = True
        try:
            orphan_dat = os.path.join(tmpdir, "a" * 64 + ".dat")
            orphan_meta = os.path.join(tmpdir, "b" * 64 + ".meta")
            broken_dat = os.path.join(tmpdir, "c" * 64 + ".dat")
            broken_meta = os.path.join(tmpdir, "c" * 64 + ".meta")
            for path, body in [(orphan_dat, b"x"), (orphan_meta, b"{}"), (broken_dat, b"x"), (broken_meta, b"not json")]:
                with open(path, "wb") as f:
                    f.write(body)

            # A fresh orphan may be a write in progress: kept.
            assert cache_service.sweep_expired(now=time.time()) == 2  # the broken pair only
            assert os.path.exists(orphan_dat) and os.path.exists(orphan_meta)

            later = time.time() + settings.cache_ttl_seconds + 10
            assert cache_service.sweep_expired(now=later) == 2
            assert os.listdir(tmpdir) == []
        finally:
            settings.cache_dir = original_dir


def test_put_sweeps_once_the_interval_has_passed():
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = settings.cache_dir
        settings.cache_dir = tmpdir
        settings.cache_enabled = True
        try:
            stale = cache_service.content_hash(b"stale")
            cache_service.put_cached(stale, "thumbnails", b"webp", ttl=1)
            # Make the entry old and the last sweep long ago.
            meta = os.path.join(tmpdir, cache_service._cache_key(stale, "thumbnails") + ".meta")
            with open(meta) as f:
                data = json.load(f)
            data["created_at"] -= 120
            with open(meta, "w") as f:
                json.dump(data, f)
            cache_service._last_sweep_at = 0.0

            cache_service.put_cached(cache_service.content_hash(b"new"), "thumbnails", b"webp")

            names = os.listdir(tmpdir)
            assert cache_service._cache_key(stale, "thumbnails") + ".dat" not in names
            assert len(names) == 2
        finally:
            settings.cache_dir = original_dir
