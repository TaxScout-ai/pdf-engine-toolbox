"""Cache expiry never breaks a request or removes a concurrent fresh write."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.config import settings
from app.services import cache_service as cache


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, 'cache_dir', str(tmp_path))
    monkeypatch.setattr(settings, 'cache_enabled', True)
    monkeypatch.setattr(cache, '_last_sweep_at', 0)
    return tmp_path


@pytest.mark.parametrize('metadata', [[], None, 7, 'bad',
    {'created_at': float('nan'), 'ttl': 3600},
    {'created_at': 1, 'ttl': float('inf')},
    {'created_at': float('inf'), 'ttl': 3600},
    {'created_at': 1, 'ttl': 'bad'},
    {'created_at': 1, 'ttl': -1}])
def test_invalid_metadata_is_removed_without_breaking_other_writes(cache_dir, metadata):
    key = cache._cache_key('bad-source', 'render')
    (cache_dir / (key + '.dat')).write_bytes(b'old')
    (cache_dir / (key + '.meta')).write_text(json.dumps(metadata))
    cache.put_cached('new-source', 'render', b'fresh')
    assert cache.get_cached('new-source', 'render') == b'fresh'
    assert not (cache_dir / (key + '.dat')).exists()
    assert not (cache_dir / (key + '.meta')).exists()


def test_read_invalid_metadata_is_a_cache_miss(cache_dir):
    key = cache._cache_key('source', 'render')
    (cache_dir / (key + '.dat')).write_bytes(b'data')
    (cache_dir / (key + '.meta')).write_text('[]')
    assert cache.get_cached('source', 'render') is None


def test_sweep_expiry_decision_cannot_delete_fresh_rewrite(cache_dir, monkeypatch):
    cache.put_cached('source', 'render', b'old')
    key = cache._cache_key('source', 'render')
    (cache_dir / (key + '.meta')).write_text(json.dumps({'created_at': 1, 'ttl': 1}))
    checked = threading.Event(); proceed = threading.Event(); writer_started = threading.Event()
    original = cache._entry_expired
    def pause_after_check(path, now):
        result = original(path, now)
        checked.set()
        assert proceed.wait(5)
        return result
    monkeypatch.setattr(cache, '_entry_expired', pause_after_check)
    def rewrite():
        writer_started.set()
        cache.put_cached('source', 'render', b'fresh')
    with ThreadPoolExecutor(2) as pool:
        sweep = pool.submit(cache.sweep_expired)
        assert checked.wait(5)
        write = pool.submit(rewrite)
        assert writer_started.wait(5)
        try:
            assert not write.done()
        finally:
            proceed.set()
        sweep.result(5); write.result(5)
    assert cache.get_cached('source', 'render') == b'fresh'


def test_partial_metadata_publication_is_not_seen_by_sweeper(cache_dir, monkeypatch):
    entered = threading.Event(); proceed = threading.Event(); sweep_started = threading.Event()
    original = json.dump
    def slow_dump(*args, **kwargs):
        entered.set()
        assert proceed.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(cache.json, 'dump', slow_dump)
    def sweep():
        sweep_started.set()
        return cache.sweep_expired()
    with ThreadPoolExecutor(2) as pool:
        write = pool.submit(cache.put_cached, 'source', 'render', b'fresh')
        assert entered.wait(5)
        swept = pool.submit(sweep)
        assert sweep_started.wait(5)
        try:
            assert not swept.done()
        finally:
            proceed.set()
        write.result(5); swept.result(5)
    assert cache.get_cached('source', 'render') == b'fresh'


def test_worker_process_waits_for_shared_directory_lock(cache_dir):
    fd = os.open(cache_dir, os.O_RDONLY)
    fcntl.flock(fd, fcntl.LOCK_EX)
    script = '''
import sys
from pathlib import Path
from app.config import settings
from app.services import cache_service as cache
settings.cache_dir = sys.argv[1]
settings.cache_enabled = True
Path(sys.argv[1], 'started').touch()
cache.put_cached('source', 'render', b'fresh')
'''
    child = subprocess.Popen([sys.executable, '-c', script, str(cache_dir)],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 10
        while not (cache_dir / 'started').exists() and time.monotonic() < deadline:
            assert child.poll() is None
            time.sleep(.01)
        assert (cache_dir / 'started').exists()
        with pytest.raises(subprocess.TimeoutExpired):
            child.wait(.2)
        fcntl.flock(fd, fcntl.LOCK_UN)
        child.communicate(timeout=10)
        assert child.returncode == 0
        assert cache.get_cached('source', 'render') == b'fresh'
    finally:
        os.close(fd)
        if child.poll() is None:
            child.kill(); child.communicate()
