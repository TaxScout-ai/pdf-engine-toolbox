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
