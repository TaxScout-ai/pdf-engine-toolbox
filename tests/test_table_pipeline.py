from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.config import settings
from app.services import table_docling, table_ocr, table_pipeline
from app.services.table_evidence import TableStructureError


def test_optional_fallback_only_runs_for_unresolved_pages(monkeypatch):
    monkeypatch.setattr(settings, "table_fallback", "docling")
    primary = {"page_index": 0, "status": "complete", "tables": [{"warnings": []}], "source": "rapidtable"}
    monkeypatch.setattr(table_ocr, "read_page_result", lambda *args, **kwargs: primary.copy())
    fallback = Mock(return_value={"status": "complete", "tables": [{"source": "docling_tableformer"}],
                                  "page_index": 0, "source": "docling_tableformer", "reason": None})
    monkeypatch.setattr(table_docling, "read_page_result", fallback)
    # A digital-only branch is tested with real PDFs in test_tables.py; here
    # skip native detection and exercise fallback policy independently.
    import fitz
    doc = fitz.open()
    doc.new_page()
    pdf = doc.tobytes()
    assert table_pipeline.read_page(pdf, 0, {"strategy": "auto"})["source"] == "rapidtable"
    fallback.assert_not_called()
    primary.update(status="unsupported", tables=[], reason="no_ruled_table_region")
    assert table_pipeline.read_page(pdf, 0, {"strategy": "auto"})["source"] == "docling_tableformer"
    assert fallback.call_count == 1
    result = table_pipeline.read_page(pdf, 0, {"strategy": "auto", "allow_fallback": False})
    assert result["status"] == "unsupported" and result["reason"] == "fallback_page_limit"
    assert fallback.call_count == 1
    table_pipeline.read_page(pdf, 0, {"strategy": "rapidtable"})
    assert fallback.call_count == 1


def _bbox(l, t, r, b):
    return SimpleNamespace(to_top_left_origin=lambda height: SimpleNamespace(l=l, t=t, r=r, b=b))


def test_fallback_budget_counts_only_unresolved_pages_under_parallel_workers(monkeypatch):
    import fitz
    from app.services import ocr_queue
    monkeypatch.setattr(settings, "table_fallback", "docling")
    monkeypatch.setattr(settings, "table_fallback_max_pages", 1)
    monkeypatch.setattr(table_ocr, "read_page_result", lambda source, index, dpi, **kwargs: {
        "page_index": index, "status": "empty" if index < 2 else "unsupported",
        "tables": [], "reason": "blank_page" if index < 2 else "no_ruled_table_region",
        "source": "rapidtable",
    })
    fallback = Mock(side_effect=lambda source, index: {
        "page_index": index, "status": "complete", "source": "docling_tableformer",
        "tables": [{"warnings": []}], "reason": None,
    })
    monkeypatch.setattr(table_docling, "read_page_result", fallback)
    with fitz.open() as doc:
        for _ in range(4):
            doc.new_page()
        results = ocr_queue.get_queue().run_pages("tables", doc.tobytes(), [0, 1, 2, 3])
    assert fallback.call_count == 1
    assert sum(r.get("reason") == "fallback_page_limit" for r in results) == 1
    assert [r["status"] for r in results[:2]] == ["empty", "empty"]


def test_tableformer_preserves_merged_cells_and_document_page():
    cell = SimpleNamespace(start_row_offset_idx=0, start_col_offset_idx=0, row_span=2,
                           col_span=1, text="(100.00)", column_header=False, bbox=_bbox(10, 20, 80, 40))
    item = SimpleNamespace(prov=[SimpleNamespace(page_no=1, bbox=_bbox(0, 0, 100, 100))],
                           data=SimpleNamespace(num_rows=2, num_cols=1, table_cells=[cell]))
    doc = SimpleNamespace(tables=[item], pages={1: SimpleNamespace(size=SimpleNamespace(width=600, height=800))})
    table = table_docling.map_document(doc, 7)[0]
    assert table["rows"] == [["(100.00)"], [""]]
    assert table["cells"][0]["page_index"] == 7
    assert table["cells"][0]["row_span"] == 2
    assert table["cells"][0]["bbox"]["y0"] == 20
    item.data.table_cells.append(cell)
    with pytest.raises(TableStructureError, match="Overlapping"):
        table_docling.map_document(doc, 7)
    doc.tables.append(SimpleNamespace(
        prov=item.prov, data=SimpleNamespace(num_rows=2, num_cols=1, table_cells=[cell])))
    partial = table_docling.map_document(doc, 7, strict=False)
    assert len(partial) == 1 and partial.unresolved_tables == 1


def _oriented_native_table(content_rotation, display_rotation=0):
    import fitz
    with fitz.open() as source, fitz.open() as output:
        page = source.new_page(width=400, height=500)
        for row, texts in enumerate([
            ['Description', 'Proceeds', 'Cost basis'],
            ['FIRST TRANSACTION', '123.45', '100.00'],
            ['SECOND TRANSACTION', '678.90', '600.00'],
        ]):
            for col, text in enumerate(texts):
                rect = fitz.Rect(20 + col * 120, 100 + row * 25,
                                 140 + col * 120, 125 + row * 25)
                page.draw_rect(rect)
                page.insert_textbox(rect, text, fontsize=9)
        width, height = ((500, 400) if content_rotation % 180 else (400, 500))
        target = output.new_page(width=width, height=height)
        target.show_pdf_page(target.rect, source, 0, rotate=content_rotation)
        target.set_rotation(display_rotation)
        return output.tobytes()


@pytest.mark.parametrize('rotation', [90, 180, 270])
@pytest.mark.parametrize('strategy', ['auto', 'pymupdf'])
def test_sideways_native_table_is_not_complete_or_replaced_by_fallback(monkeypatch, rotation, strategy):
    monkeypatch.setattr(settings, 'table_fallback', 'docling')
    fallback = Mock(side_effect=AssertionError('unqualified fallback must not run'))
    monkeypatch.setattr(table_docling, 'read_page_result', fallback)
    monkeypatch.setattr(table_ocr, 'read_page_result', fallback)
    result = table_pipeline.read_page(_oriented_native_table(rotation), 0, {'strategy': strategy})
    assert result['status'] == 'unsupported'
    assert result['reason'] == 'native_text_orientation_unresolved'
    assert result['tables'] == []
    fallback.assert_not_called()


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
def test_pdf_display_rotation_already_corrects_native_text(rotation):
    import fitz
    with fitz.open(stream=_oriented_native_table(rotation, rotation), filetype='pdf') as doc:
        assert not table_pipeline._native_text_needs_orientation(doc[0])
    result = table_pipeline.read_page(_oriented_native_table(rotation, rotation), 0,
                                      {'strategy': 'pymupdf'})
    assert result['status'] == 'complete'
    assert result['tables'][0]['rows'][1] == ['FIRST TRANSACTION', '123.45', '100.00']


def test_small_vertical_margin_label_does_not_reject_upright_table():
    import fitz
    with fitz.open(stream=_oriented_native_table(0), filetype='pdf') as doc:
        doc[0].insert_text((390, 480), 'COPY FOR RECIPIENT', rotate=90, fontsize=8)
        assert not table_pipeline._native_text_needs_orientation(doc[0])
        assert table_pipeline.read_page(doc.tobytes(), 0, {'strategy': 'pymupdf'})['status'] == 'complete'
