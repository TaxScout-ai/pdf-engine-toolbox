"""Actual PDF source highlights: signs, repeated values, spans and page rotation."""
from copy import deepcopy
from types import SimpleNamespace

import fitz
import pytest

from app.services.pdf_service import _extract_tables_pymupdf
from app.services.table_native_geometry import refine_native_highlights


def cell(text, bounds, row=0, col=0, **extra):
    return dict(text=text, raw_text=text, bbox=dict(zip(('x0', 'y0', 'x1', 'y1'), bounds)),
                geometry_status='available', page_index=0, row_index=row, column_index=col,
                row_span=1, column_span=1, source='native_pdf', **extra)


def test_repeated_amounts_stay_in_their_own_row_and_keep_whole_signed_words():
    with fitz.open() as doc:
        p = doc.new_page()
        p.insert_text((80, 80), '-100.00')
        p.insert_text((80, 120), '-100.00')
        cells = [cell('-100.00', (40, 60, 220, 95)),
                 cell('-100.00', (40, 100, 220, 135), row=1)]
        original = deepcopy(cells)
        refine_native_highlights(p, cells)
        for i, c in enumerate(cells):
            assert c['geometry_kind'] == 'native_text'
            assert c['cell_bbox'] == original[i]['bbox']
            assert c['raw_text'] == c['text'] == '-100.00'
            assert c['row_index'] == i
            b = fitz.Rect(list(c['bbox'].values()))
            assert b.width < 60 and b.x0 == 80
            assert p.get_textbox(b + (-.01, -.01, .01, .01)).strip() == '-100.00'
        assert cells[0]['bbox']['y1'] < cells[1]['bbox']['y0']


@pytest.mark.parametrize('rotation', [0, 90, 180, 270])
def test_rotation_maps_to_the_displayed_page_without_clipping(rotation):
    with fitz.open() as doc:
        p = doc.new_page(width=612, height=792)
        p.insert_text((300, 300), '(123.45)', rotate=rotation)
        p.set_rotation(rotation)
        word = fitz.Rect(p.get_text('words')[0][:4]) * p.rotation_matrix
        region = word + (-20, -5, 40, 8)
        cells = [cell('(123.45)', region)]
        refine_native_highlights(p, cells)
        assert cells[0]['geometry_kind'] == 'native_text'
        got = list(cells[0]['bbox'].values())
        assert got == pytest.approx(list(word), abs=.0001)


@pytest.mark.parametrize('text', ['100.00', '100,00', '-100.0'])
def test_missing_sign_or_changed_value_never_gets_a_precise_match(text):
    with fitz.open() as doc:
        p = doc.new_page(); p.insert_text((80, 80), '-100.00')
        cells = [cell(text, (40, 60, 220, 95))]
        before = deepcopy(cells[0]['bbox'])
        refine_native_highlights(p, cells)
        assert cells[0]['bbox'] == before
        assert cells[0]['geometry_kind'] == 'cell_region'
        assert cells[0]['text'] == text


def test_separately_painted_minus_is_not_omitted():
    with fitz.open() as doc:
        p = doc.new_page(); p.insert_text((76, 80), '-', fontsize=7)
        p.insert_text((90, 80), '100.00', fontsize=11)
        cells = [cell('100.00', (40, 60, 220, 95))]
        refine_native_highlights(p, cells)
        assert cells[0]['geometry_kind'] == 'cell_region'


def test_overlapping_cells_cannot_reuse_the_same_word():
    with fitz.open() as doc:
        p = doc.new_page(); p.insert_text((80, 80), '100.00')
        cells = [cell('100.00', (40, 60, 220, 95)), cell('100.00', (60, 60, 240, 95), col=1)]
        refine_native_highlights(p, cells)
        assert all(c['geometry_kind'] == 'cell_region' for c in cells)


@pytest.mark.parametrize('mode', [3, 0])
def test_invisible_overlay_or_transparent_text_keeps_only_grid_evidence(mode):
    with fitz.open() as doc:
        p = doc.new_page()
        p.insert_text((80, 80), '100.00', render_mode=mode, fill_opacity=0 if mode == 0 else 1)
        cells = [cell('100.00', (40, 60, 220, 95))]
        refine_native_highlights(p, cells)
        assert cells[0]['geometry_kind'] == 'cell_region'


def test_crossing_word_is_not_clipped_into_the_next_column():
    with fitz.open() as doc:
        p = doc.new_page(); p.insert_text((80, 80), '-100.00')
        cells = [cell('-100.00', (85, 60, 220, 95))]
        refine_native_highlights(p, cells)
        assert cells[0]['geometry_kind'] == 'cell_region'


def test_multiline_merged_cell_keeps_spans_and_text():
    with fitz.open() as doc:
        p = doc.new_page(); p.insert_text((80, 80), 'First line\nSecond line')
        c = cell('First line\nSecond line', (40, 60, 250, 120))
        c.update(row_span=2, column_span=2)
        refine_native_highlights(p, [c])
        assert c['geometry_kind'] == 'native_text'
        assert (c['row_span'], c['column_span']) == (2, 2)
        assert c['raw_text'] == 'First line\nSecond line'


def test_native_table_integration_preserves_structure_and_page_index():
    with fitz.open() as doc:
        doc.new_page(); p = doc.new_page()
        for r, row in enumerate([['Account', 'Amount'], ['First', '-100.00'], ['Second', '-100.00']]):
            for col, text in enumerate(row):
                b = fitz.Rect(40 + col*150, 40 + r*35, 190 + col*150, 75 + r*35)
                p.draw_rect(b); p.insert_text((b.x0+8, b.y0+20), text)
        tables = _extract_tables_pymupdf(doc, 1)
        assert len(tables) == 1 and tables[0]['row_count'] == 3 and tables[0]['col_count'] == 2
        amounts = [c for c in tables[0]['cells'] if c['text'] == '-100.00']
        assert len(amounts) == 2
        assert all(c['page_index'] == 1 and c['geometry_kind'] == 'native_text' for c in amounts)
        assert amounts[0]['bbox']['y1'] < amounts[1]['bbox']['y0']
        assert not tables[0]['warnings']


def test_geometry_read_failure_preserves_the_extracted_table():
    def fail():
        raise RuntimeError('Synthetic native reader failure')
    c = cell('100.00', (40, 60, 220, 95))
    before = deepcopy(c)
    refine_native_highlights(SimpleNamespace(rotation=0, get_texttrace=fail), [c])
    assert c['bbox'] == before['bbox'] and c['text'] == before['text']
    assert c['geometry_kind'] == 'cell_region'


def test_dense_ambiguous_grid_has_a_bounded_comparison_budget(monkeypatch):
    from app.services import table_native_geometry as geometry
    monkeypatch.setattr(geometry, 'MAX_CANDIDATE_CHECKS', 1)
    with fitz.open() as doc:
        p = doc.new_page(); p.insert_text((80, 80), '100.00 200.00')
        c = cell('100.00 200.00', (40, 60, 220, 95))
        refine_native_highlights(p, [c])
        assert c['geometry_kind'] == 'cell_region'
