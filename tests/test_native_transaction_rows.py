"""Original PDF coordinates must support whole rows, not just matching digits."""
from copy import deepcopy

import fitz
import pytest

from app.services.table_native_rows import refine_native_tables

HEADERS = ['Description of property CUSIP (Box 1a)', 'Quantity Sold',
           'Date Acquired (Box 1b)', 'Date Sold or Disposed (Box 1c)',
           'Proceeds (Box 1d)', 'Cost or Other Basis (Box 1e)',
           'Accrued Market Discount (Box 1f)', 'Wash Sale Loss Disallowed (Box 1g)',
           'Gain/Loss Amount', 'Additional Information']
VALUES = [['EXAMPLE FUND', 'CUSIP: 111111111'], ['1.000', '2.000'],
          ['01/01/2025', '02/02/2025'], ['10/01/2025', '10/01/2025'],
          ['$100.00', '$100.00'], ['$90.00', '$95.00'], ['$0.00', '$0.00'],
          ['$0.00', '$0.00'], ['$10.00', '$5.00'], []]


def fixture(*, values=None, shifts=None, annotation=False, hidden=False):
    values = values or deepcopy(VALUES)
    shifts = shifts or {}
    doc = fitz.open()
    page = doc.new_page(width=1300, height=200)
    cells = []
    for col, header in enumerate(HEADERS):
        x = 10 + col * 128
        page.insert_text((x, 50), header, fontsize=5)
        for i, value in enumerate(values[col]):
            page.insert_text((x, 100 + i * 15 + shifts.get(col, 0)), value, fontsize=8)
    page.insert_text((10, 160), 'TOTAL', fontsize=8)
    page.insert_text((10 + 4 * 128, 160), '$200.00', fontsize=8)
    words = page.get_text('words')
    for row, (y0, y1) in enumerate([(40, 55), (85, 140), (145, 165)]):
        for col in range(len(HEADERS)):
            x = 10 + col * 128
            selected = [w for w in words if x <= w[0] < x + 128 and y0 <= w[1] <= y1]
            if not selected:
                continue
            box = fitz.Rect(selected[0][:4])
            for w in selected[1:]:
                box |= fitz.Rect(w[:4])
            text = ' '.join(w[4] for w in selected)
            cells.append(dict(page_index=7, row_index=row, column_index=col,
                              row_span=1, column_span=1, raw_text=text, text=text,
                              bbox=dict(zip(('x0', 'y0', 'x1', 'y1'), box)),
                              is_header=row == 0, source='docling_tableformer',
                              geometry_status='available'))
    if annotation:
        page.insert_text((1200, 108), 'RESTRICTION', fontsize=8)
    if hidden:
        page.insert_text((1200, 108), 'HIDDEN', fontsize=8, render_mode=3)
    rows = [[''] * len(HEADERS) for _ in range(3)]
    for cell in cells:
        rows[cell['row_index']][cell['column_index']] = cell['text']
    return doc, dict(page_index=7, cells=cells, rows=rows, row_count=3,
                     col_count=len(HEADERS), page_width=1300, page_height=200,
                     bbox=dict(x0=10, y0=40, x1=1290, y1=165), warnings=[])


def run(table, doc):
    return refine_native_tables(doc[0], [table])


def test_recovers_dates_amounts_and_shared_security_as_one_valid_grid():
    doc, table = fixture()
    snapshot = deepcopy(table)
    result, unresolved = run(table, doc)
    assert unresolved == 0
    actual = result[0]
    assert actual['row_count'] == 4
    assert actual['rows'][1][2:6] == ['01/01/2025', '10/01/2025', '$100.00', '$90.00']
    assert actual['rows'][2][2:6] == ['02/02/2025', '10/01/2025', '$100.00', '$95.00']
    assert actual['rows'][2][0] == ''  # CSV does not invent a duplicate description.
    assert actual['rows'][3][0] == 'TOTAL'
    security = next(c for c in actual['cells'] if c['row_index'] == 1 and c['column_index'] == 0)
    assert security['row_span'] == 2
    assert security['bbox'] == next(c['bbox'] for c in table['cells'] if c['row_index'] == 1 and c['column_index'] == 0)
    occupied = set()
    for c in actual['cells']:
        assert c['page_index'] == 7
        slots = {(r, col) for r in range(c['row_index'], c['row_index'] + c['row_span'])
                 for col in range(c['column_index'], c['column_index'] + c['column_span'])}
        assert not slots & occupied
        occupied |= slots
    assert table == snapshot


@pytest.mark.parametrize('case', ['amount_missing', 'date_shift', 'different_security',
                                  'annotation', 'date_invalid', 'swapped_amount_columns'])
def test_ambiguous_transaction_is_removed_not_exposed_as_partial_table(case):
    values = deepcopy(VALUES)
    if case == 'different_security':
        values[0] = ['CUSIP: 111111111', 'CUSIP: 222222222']
    if case == 'date_invalid':
        values[2][1] = '02/30/2025'
    doc, table = fixture(values=values, shifts={2: 5} if case == 'date_shift' else {},
                         annotation=case == 'annotation')
    if case == 'amount_missing':
        c = next(c for c in table['cells'] if c['row_index'] == 1 and c['column_index'] == 4)
        c.update(raw_text='$100.00', text='$100.00')
    if case == 'swapped_amount_columns':
        for c in table['cells']:
            if c['row_index'] == 1 and c['column_index'] in (4, 5):
                c['column_index'] = 9 - c['column_index']
    assert run(table, doc) == ([], 1)


def test_wrong_physical_header_order_removes_table_even_without_recovery():
    doc, table = fixture()
    for c in table['cells']:
        if c['column_index'] in (2, 3):
            c['column_index'] = 5 - c['column_index']
    assert run(table, doc) == ([], 1)


def test_preserves_other_valid_tables_when_one_mapping_is_unsafe():
    doc, table = fixture()
    unsafe = deepcopy(table)
    for c in unsafe['cells']:
        if c['column_index'] in (2, 3):
            c['column_index'] = 5 - c['column_index']
    result, unresolved = refine_native_tables(doc[0], [table, unsafe])
    assert unresolved == 1 and len(result) == 1
    assert result[0]['row_count'] == 4


def test_hidden_native_overlay_is_not_used_to_reconstruct_transactions():
    doc, table = fixture(hidden=True)
    assert run(table, doc) == ([table], 0)


def test_single_transaction_with_wrapped_description_is_unchanged():
    values = [[v[0]] if i != 0 and v else v for i, v in enumerate(deepcopy(VALUES))]
    doc, table = fixture(values=values)
    assert run(table, doc) == ([table], 0)


def test_existing_spans_do_not_silently_turn_merged_rows_into_success():
    doc, table = fixture()
    table['cells'][-1]['row_span'] = 2
    assert run(table, doc) == ([], 1)


def test_adapter_quarantines_unsafe_tables_and_preserves_original_page(monkeypatch):
    from types import SimpleNamespace
    from app.services import table_docling
    doc, table = fixture()
    for cell in table['cells']:
        if cell['column_index'] in (2, 3):
            cell['column_index'] = 5 - cell['column_index']
    mapped = table_docling.MappedTables([table])
    converted = SimpleNamespace(status=SimpleNamespace(value='success'), document=object())
    monkeypatch.setattr(table_docling, '_get_converter', lambda: SimpleNamespace(convert=lambda *a, **kw: converted))
    monkeypatch.setattr(table_docling, 'map_document', lambda *a, **kw: mapped)
    result = table_docling.read_page_result(doc.tobytes(), 0)
    assert result['status'] == 'unsupported'
    assert result['reason'] == 'fallback_invalid_table_structure'
    assert result['tables'] == []


def test_duplicate_printed_box_label_does_not_qualify_as_two_date_roles():
    from app.services.table_native_rows import _profile
    headers = [dict(text=h, column_index=i) for i, h in enumerate(HEADERS)]
    headers[3]['text'] = 'Date Sold (Box 1b)'
    assert _profile(headers) is None


def test_blank_information_box_cannot_conceal_a_printed_restriction():
    doc, table = fixture(annotation=True)
    table['cells'].append(dict(page_index=7, row_index=1, column_index=9,
                              row_span=1, column_span=1, text='', raw_text='', is_header=False,
                              bbox=dict(x0=1160, y0=90, x1=1290, y1=130)))
    assert run(table, doc) == ([], 1)


@pytest.mark.parametrize('unrelated', ['hidden', 'missing_header'])
def test_unrelated_missing_or_hidden_text_cannot_erase_known_header_conflict(unrelated):
    doc, table = fixture(hidden=unrelated == 'hidden')
    for c in table['cells']:
        if c['column_index'] in (2, 3):
            c['column_index'] = 5 - c['column_index']
    if unrelated == 'missing_header':
        table['cells'] = [c for c in table['cells'] if not (c['is_header'] and c['column_index'] == 9)]
    assert run(table, doc) == ([], 1)


def test_shared_security_cannot_cross_a_printed_transaction_separator():
    doc, table = fixture()
    doc[0].draw_line((0, 106), (1290, 106))
    assert run(table, doc) == ([], 1)


def test_unresolved_second_security_is_not_a_shared_description():
    values = deepcopy(VALUES)
    values[0] = ['CUSIP: 111111111', 'CUSIP: UNKNOWN']
    doc, table = fixture(values=values)
    assert run(table, doc) == ([], 1)


def test_ordered_body_columns_shifted_under_different_headers_are_rejected():
    doc, table = fixture()
    # Shift both source glyphs and claimed source boxes, preserving body order.
    source = fitz.open()
    source.insert_pdf(doc)
    shifted = fitz.open()
    page = shifted.new_page(width=1430, height=200)
    page.show_pdf_page(fitz.Rect(0, 0, 1300, 80), source, 0, clip=fitz.Rect(0, 0, 1300, 80))
    page.show_pdf_page(fitz.Rect(128, 80, 1428, 140), source, 0, clip=fitz.Rect(0, 80, 1300, 140))
    for c in table['cells']:
        if c['row_index'] == 1:
            c['bbox']['x0'] += 128
            c['bbox']['x1'] += 128
    assert run(table, shifted) == ([], 1)
