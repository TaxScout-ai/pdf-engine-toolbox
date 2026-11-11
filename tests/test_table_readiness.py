"""Readiness must precede traffic and recovery when the optional model is on."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
import sys
import threading

import pytest

from app.config import settings
from app.main import app, lifespan
from app.services import ocr_queue, rapid_ocr, table_docling


@pytest.fixture
def startup(monkeypatch):
    events=[]
    monkeypatch.setattr(rapid_ocr,'validate_runtime',lambda:events.append('validate'))
    monkeypatch.setattr(ocr_queue,'get_queue',lambda:events.append('recover'))
    monkeypatch.setattr(ocr_queue,'shutdown',lambda:events.append('shutdown'))
    return events


def test_disabled_optional_model_does_not_prepare(startup,monkeypatch):
    monkeypatch.setattr(settings,'table_fallback','none')
    def unexpected():raise AssertionError('Disabled reader loaded')
    monkeypatch.setattr(table_docling,'prepare',unexpected)
    async def run():
        async with lifespan(app):startup.append('serve')
    asyncio.run(run())
    assert startup==['validate','recover','serve','shutdown']


def test_enabled_pipeline_finishes_before_recovery_and_http(startup,monkeypatch):
    monkeypatch.setattr(settings,'table_fallback','docling')
    started,release=threading.Event(),threading.Event()
    def prepare():
        startup.append('prepare');started.set()
        assert release.wait(5)
        startup.append('prepared')
    monkeypatch.setattr(table_docling,'prepare',prepare)
    async def run():
        async def enter():
            async with lifespan(app):startup.append('serve')
        task=asyncio.create_task(enter())
        try:
            assert await asyncio.to_thread(started.wait,5)
            assert startup==['validate','prepare']
        finally:release.set()
        await task
    asyncio.run(run())
    assert startup==['validate','prepare','prepared','recover','serve','shutdown']


def test_preparation_failure_aborts_before_recovery_or_serving(startup,monkeypatch):
    monkeypatch.setattr(settings,'table_fallback','docling')
    def fail():raise RuntimeError('Offline pipeline unavailable')
    monkeypatch.setattr(table_docling,'prepare',fail)
    async def run():
        async with lifespan(app):startup.append('serve')
    with pytest.raises(RuntimeError,match='Offline pipeline unavailable'):asyncio.run(run())
    assert startup==['validate']


@pytest.fixture
def preparation(monkeypatch):
    # The standard image intentionally has no Docling package.
    module=ModuleType('docling.datamodel.base_models')
    module.InputFormat=SimpleNamespace(PDF=object())
    monkeypatch.setitem(sys.modules,'docling.datamodel.base_models',module)
    monkeypatch.setattr(table_docling,'_prepared',False)
    return module.InputFormat.PDF


def test_pipeline_initialization_is_serialized_once(preparation,monkeypatch):
    calls=[]
    monkeypatch.setattr(table_docling,'_get_converter',lambda:SimpleNamespace(
        initialize_pipeline=lambda kind:calls.append(kind)))
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda _:table_docling.prepare(),range(8)))
    assert calls==[preparation]


def test_failed_preparation_is_not_cached_as_ready(preparation,monkeypatch):
    calls=[]
    def initialize(kind):
        calls.append(kind)
        if len(calls)==1:raise FileNotFoundError('Missing offline weight')
    monkeypatch.setattr(table_docling,'_get_converter',lambda:SimpleNamespace(initialize_pipeline=initialize))
    with pytest.raises(FileNotFoundError):table_docling.prepare()
    assert table_docling._prepared is False
    table_docling.prepare()
    assert calls==[preparation,preparation] and table_docling._prepared is True
