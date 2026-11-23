"""Native bank columns must retain blanks, signs and exact source regions."""
import fitz
import pytest

from app.config import settings
from app.services import pdf_service, table_pipeline, table_native_bank


def bank(rows=None, header=True, hidden=False):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    labels = ['Date', 'Description', 'Money out', 'Money in', 'Balance']
    xs = [40, 125, 330, 415, 525]
    data = rows if rows is not None else [
        ['Nov 13, 2025', 'Deposit', '', '$20.00', '$20.00'],
        ['Nov 18, 2025', 'Deposit', '', '$5,000.00', '$5,020.00'],
        ['Dec 1, 2025', 'Interest', '', '$6.49', '$5,026.49']]
    if header:
        data = [labels, *data]
    for r, row in enumerate(data):
        for x, text in zip(xs, row):
            if text:
                if x == 525:
                    x = 555 - fitz.get_text_length(text, fontsize=8)
                page.insert_text((x, 80 + r*24), text, fontsize=8, render_mode=3 if hidden else 0)
    return doc


def test_bank_table_preserves_empty_column_exact_amounts_and_tight_geometry():
    with bank() as doc:
        result = table_pipeline.read_page(doc.tobytes(), 0, {'strategy': 'pymupdf'})
        assert result['status'] == 'complete'
        table, = result['tables']
        assert table['structure_method'] == 'native_bank_header'
        assert len(table['cells']) == 20
        assert table['rows'][3] == ['Dec 1, 2025', 'Interest', '', '$6.49', '$5,026.49']
        for cell in table['cells']:
            assert cell['page_index'] == 0
            assert cell['row_span'] == cell['column_span'] == 1
            if cell['text']:
                assert cell['geometry_kind'] == 'native_text'
                clip = fitz.Rect(*cell['bbox'].values())
                assert ''.join(doc[0].get_textbox(clip).split()) == ''.join(cell['raw_text'].split())
            else:
                assert cell['text'] == cell['raw_text'] == ''


@pytest.mark.parametrize('amount', ['-$2.50', '($2.50)', '$0.00'])
def test_outflow_sign_and_zero_preserved(amount):
    rows = [['Nov 13, 2025', 'Fee', amount, '', '$20.00'],
            ['Nov 18, 2025', 'Fee', amount, '', '$20.00']]
    with bank(rows) as doc:
        tables = pdf_service._extract_tables_pymupdf(doc, 0)
        assert tables[0]['rows'][1][2] == amount
        assert tables[0]['rows'][1][3] == ''


@pytest.mark.parametrize('row', [
    ['Nov 18, 2025', 'Deposit', '', '$999', '$20.00'],
    ['Nov 18, 2025', 'Deposit', '', '$999.9', '$20.00'],
    ['Nov 18, 2025', 'Fee', '($1234,567.89)', '', '$20.00'],
    ['Nov 99, 2025', 'Deposit', '', '$20.00', '$20.00'],
    ['Nov 18, 2025', 'Deposit', '$1.00', '$20.00', '$20.00'],
    ['Nov 18, 2025', '', '', '$20.00', '$20.00'],
    ['', 'Continued transaction', '', '$20.00', '$20.00'],
])
def test_ambiguous_or_incomplete_row_abstains_instead_of_disappearing(row):
    with bank([['Nov 13, 2025', 'Deposit', '', '$20.00', '$20.00'], row]) as doc:
        assert table_native_bank.recover_bank_table(doc[0], 0) == []


def test_kill_switch_restores_previous_native_result(monkeypatch):
    with bank() as doc:
        monkeypatch.setattr(settings, 'native_bank_tables_enabled', False)
        assert pdf_service._extract_tables_pymupdf(doc, 0) == []


@pytest.mark.parametrize('mode', ['no_header', 'hidden', 'covered', 'rotated', 'interrupted', 'double_header'])
def test_unqualified_layout_is_not_recovered(mode):
    with bank(header=mode != 'no_header', hidden=mode == 'hidden') as doc:
        p = doc[0]
        if mode == 'covered':
            p.draw_rect(fitz.Rect(400, 90, 600, 170), fill=(1, 1, 1))
        if mode == 'rotated':
            p.set_rotation(90)
        if mode == 'interrupted':
            p.insert_text((40, 116), 'Notes between rows', fontsize=8)
        if mode == 'double_header':
            for x, t in zip([40,125,330,415,525], ['Date','Description','Money out','Money in','Balance']):
                p.insert_text((x, 200), t, fontsize=8)
        assert table_native_bank.recover_bank_table(p, 0) == []


def test_word_limit_abstains(monkeypatch):
    with bank() as doc:
        monkeypatch.setattr(table_native_bank, 'MAX_WORDS', 10)
        assert table_native_bank.recover_bank_table(doc[0], 0) == []


def test_row_limit_abstains(monkeypatch):
    with bank() as doc:
        monkeypatch.setattr(table_native_bank, 'MAX_ROWS', 2)
        assert table_native_bank.recover_bank_table(doc[0], 0) == []


@pytest.mark.parametrize('mode', ['wrapped', 'damaged'])
def test_trailing_transaction_content_cannot_be_discarded_as_footer(mode):
    with bank() as doc:
        if mode == 'wrapped':
            doc[0].insert_text((125, 164), 'Adjustment for previous month', fontsize=8)
        else:
            for x, text in [(40,'N0v 20, 2025'), (125,'Deposit'), (415,'$999'), (525,'$999')]:
                doc[0].insert_text((x, 176), text, fontsize=8)
        assert table_native_bank.recover_bank_table(doc[0], 0) == []


def test_full_width_footer_prose_is_outside_transaction_table():
    with bank() as doc:
        doc[0].insert_text((40, 180), 'This statement is provided by the bank in relation to your account transactions and account activity.', fontsize=8)
        assert len(table_native_bank.recover_bank_table(doc[0], 0)) == 1


def test_overlapping_description_words_do_not_establish_visible_source():
    with bank() as doc:
        doc[0].insert_text((130, 104), 'Other', fontsize=8)
        assert table_native_bank.recover_bank_table(doc[0], 0) == []
