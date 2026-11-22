"""Synthetic regressions for header-only false completeness (TAX-11151)."""
from copy import deepcopy
from unittest.mock import Mock

import fitz
import pytest

from app.config import settings
from app.services import table_docling, table_pipeline, pdf_service
from app.services.table_native_coverage import coverage_reason


def statement(*, rows=5, columns=2, hidden=False, covered=False, ruled_body=False):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for y in (40, 65, 90):
        page.draw_line((40, y), (550, y))
    for x in (40, 295, 550):
        page.draw_line((x, 40), (x, 90))
    for row, labels in enumerate([['Statement', 'Year'], ['SYNTHETIC ACCOUNT', '2025']]):
        for col, label in enumerate(labels):
            page.insert_text((50 + col * 255, 57 + row * 25), label, fontsize=10)
    for row in range(rows):
        y = 200 + row * 25
        page.insert_text((50, y), 'Financial line', fontsize=10)
        for col in range(columns):
            page.insert_text((300 + col * 110, y), '-123.45' if col else '0.00', fontsize=10,
                             render_mode=3 if hidden else 0)
    if covered:
        page.draw_rect(fitz.Rect(290, 180, 550, 210 + rows * 25), color=None, fill=(1, 1, 1))
    if ruled_body:
        for row in range(rows + 1):
            page.draw_line((40, 183 + row * 25), (550, 183 + row * 25))
        for x in (40, 290, 400, 550):
            page.draw_line((x, 183), (x, 183 + rows * 25))
    return doc


@pytest.mark.parametrize('strategy', ['auto', 'pymupdf'])
@pytest.mark.parametrize('columns', [1, 2])
def test_header_only_is_partial_without_changing_extracted_evidence(monkeypatch, strategy, columns):
    with statement(columns=columns) as doc:
        before = deepcopy(pdf_service._extract_tables_pymupdf(doc, 0))
        assert before and len(before) == 1
        monkeypatch.setattr(settings, 'table_fallback', 'docling')
        fallback = Mock(side_effect=AssertionError('unqualified fallback'))
        monkeypatch.setattr(table_docling, 'read_page_result', fallback)
        result = table_pipeline.read_page(doc.tobytes(), 0, {'strategy': strategy})
        assert result['status'] == 'unsupported'
        assert result['reason'] == 'native_financial_rows_omitted'
        assert result['tables'] == before
        fallback.assert_not_called()


@pytest.mark.parametrize('options', [{'rows': 1}, {'rows': 2}, {'hidden': True}, {'covered': True}, {'ruled_body': True}])
def test_incidental_hidden_covered_or_already_extracted_amounts_do_not_trigger(options):
    with statement(**options) as doc:
        result = table_pipeline.read_page(doc.tobytes(), 0, {'strategy': 'pymupdf'})
        assert result['status'] == 'complete'
        assert result['reason'] is None


@pytest.mark.parametrize('angle', [0, 90, 180, 270])
def test_corrected_pdf_display_rotation_preserves_missing_row_detection(angle):
    with statement() as source, fitz.open() as doc:
        target = doc.new_page(width=792 if angle % 180 else 612,
                              height=612 if angle % 180 else 792)
        target.show_pdf_page(target.rect, source, 0, rotate=angle)
        target.set_rotation(angle)
        result = table_pipeline.read_page(doc.tobytes(), 0, {'strategy': 'pymupdf'})
        assert result['reason'] == 'native_financial_rows_omitted'


def test_large_inputs_abstain_without_unbounded_pairwise_work(monkeypatch):
    from app.services import table_native_coverage
    with statement() as doc:
        tables = pdf_service._extract_tables_pymupdf(doc, 0)
        monkeypatch.setattr(table_native_coverage, 'MAX_CANDIDATE_CHECKS', 1)
        assert coverage_reason(doc[0], tables) == 'native_financial_coverage_limit'


def test_no_tables_does_not_create_any_evidence():
    with statement() as doc:
        assert coverage_reason(doc[0], []) is None


def test_three_fee_sentences_are_prose_not_missing_columns():
    with statement(rows=0) as doc:
        for i, text in enumerate([
            'We charge $10.00 for delivery and $20.00 for expedited service.',
            'Replacement cards are $5.00 each; overseas replacements cost $15.00 per card.',
            'A returned payment incurs $25.00 plus a $3.00 processing fee.',
            'We charge $10.00 and $20.00',
            'We charge $10.00 and $20.00',
            'We charge $10.00 and $20.00',
            'We charge $10.00 and $20.00',
        ]):
            doc[0].insert_text((50, 200 + 25*i), text, fontsize=9)
        result = table_pipeline.read_page(doc.tobytes(), 0, {'strategy': 'pymupdf'})
        assert result['status'] == 'complete'


@pytest.mark.parametrize('decoration', ['underlines', 'transparent_fill', 'hidden_duplicates'])
def test_visible_missing_amounts_survive_non_occluding_objects(decoration):
    with statement() as doc:
        page = doc[0]
        for row in range(5):
            y = 200 + row * 25
            if decoration == 'underlines':
                page.draw_line((295, y+1), (465, y+1), width=.5)
            elif decoration == 'transparent_fill':
                page.draw_rect(fitz.Rect(290, y-15, 550, y+5), color=None, fill=(1, 1, 1), fill_opacity=0)
            else:
                for col in range(2):
                    page.insert_text((300 + col * 110, y), '-123.45' if col else '0.00', fontsize=10, render_mode=3)
        result = table_pipeline.read_page(doc.tobytes(), 0, {'strategy': 'pymupdf'})
        assert len(result['tables']) == 1
        assert result['reason'] == 'native_financial_rows_omitted'


@pytest.mark.parametrize('mode', ['hidden', 'transparent', 'covered'])
def test_invisible_prose_does_not_hide_omitted_financial_rows(mode):
    with statement() as doc:
        for row in range(5):
            y = 200 + row * 25
            kwargs = {'render_mode': 3} if mode == 'hidden' else {'fill_opacity': 0} if mode == 'transparent' else {}
            doc[0].insert_text((470, y), 'USD', fontsize=10, **kwargs)
            if mode == 'covered':
                doc[0].draw_rect(fitz.Rect(468, y-15, 500, y+5), color=None, fill=(1, 1, 1))
        result = table_pipeline.read_page(doc.tobytes(), 0, {'strategy': 'pymupdf'})
        assert result['reason'] == 'native_financial_rows_omitted'
