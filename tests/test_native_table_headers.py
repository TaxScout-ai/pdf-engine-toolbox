from copy import deepcopy
from types import SimpleNamespace

import fitz
import pytest

from app.services.table_docling import map_document
from app.services.table_evidence import TableStructureError
from app.services.table_native_headers import recover_headers


HEADERS = ['1c- Date sold', 'Quantity', '1d- Proceeds', '1b- Date acquired', '1e- Cost basis']
VALUES = [['01/01/25', '2', '120.00', '01/01/24', '100.00'],
          ['02/01/25', '3', '150.00', '02/01/24', '110.00']]


def fixture():
    doc = fitz.open()
    page = doc.new_page(width=600, height=300)
    cells = []
    for row, texts in enumerate([HEADERS] + VALUES):
        for col, text in enumerate(texts):
            point = (20 + col*110, 50 + row*25)
            page.insert_text(point, text, fontsize=7)
            words = [w for w in page.get_text('words') if abs(w[0]-point[0]) < 100
                     and point[1]-10 < w[1] < point[1]]
            # Bound the current physical cell, not neighbouring words.
            words = [w for w in words if point[0] <= w[0] < point[0]+100]
            b = fitz.Rect(min(w[0] for w in words), min(w[1] for w in words),
                          max(w[2] for w in words), max(w[3] for w in words))
            cells.append(dict(row_index=row, column_index=col, row_span=1, column_span=1,
                              is_header=row == 0, text=text, raw_text=text, page_index=4,
                              bbox=dict(x0=b.x0,y0=b.y0,x1=b.x1,y1=b.y1),
                              source='docling_tableformer',geometry_status='available'))
    cells[2]['column_span'] = 2
    cells[2]['text'] = cells[2]['raw_text'] = '1d- Proceeds 1b- Date'
    cells[2]['bbox']['x1'] = cells[3]['bbox']['x1']
    cells[3]['text'] = cells[3]['raw_text'] = 'acquired'
    cells[3]['is_header'] = False
    return doc, cells


def test_source_headers_recover_without_body_or_input_mutation():
    doc, cells = fixture()
    before = deepcopy(cells)
    result = recover_headers(doc[0], cells, 5)
    assert result is not cells
    assert cells == before
    assert result[5:] == before[5:]
    assert [c['text'] for c in result[:5]] == HEADERS
    assert all(c['page_index'] == 4 and c['column_span'] == 1 and c['is_header'] for c in result[:5])
    assert all(c['raw_text'] == c['text'] for c in result[:5])


@pytest.mark.parametrize('change', ['missing_word', 'extra_word', 'missing_geometry', 'header_rowspan',
                                  'body_column', 'body_text', 'body_row', 'body_rowspan', 'no_overlap',
                                  'missing_date_marker', 'rotated', 'hidden', 'extra_source'])
def test_uncertain_source_or_grid_does_not_bypass_overlap_validation(change):
    doc, cells = fixture()
    if change == 'missing_word': cells[3]['raw_text'] = ''
    elif change == 'extra_word': cells[2]['raw_text'] += ' OTHER'
    elif change == 'missing_geometry': cells[2]['bbox'] = None
    elif change == 'header_rowspan': cells[2]['row_span'] = 2
    elif change == 'body_column': cells[7]['column_index'] = 1
    elif change == 'body_text': cells[7]['raw_text'] = '999.00'
    elif change == 'body_row': cells[7]['row_index'] = 2
    elif change == 'body_rowspan': cells[7]['row_span'] = 2
    elif change == 'no_overlap': cells[2]['column_span'] = 1
    elif change == 'missing_date_marker':
        cells[2]['raw_text'] = cells[2]['raw_text'].replace('1b-', 'Date')
    elif change == 'rotated': doc[0].set_rotation(90)
    elif change == 'hidden': doc[0].insert_text((30,200), 'HIDDEN', render_mode=3)
    elif change == 'extra_source': doc[0].insert_text((100,50), 'EXTRA', fontsize=7)
    assert recover_headers(doc[0], cells, 5) is cells


def as_document(cells):
    def bbox(b):
        return SimpleNamespace(to_top_left_origin=lambda height: SimpleNamespace(
            l=b['x0'],t=b['y0'],r=b['x1'],b=b['y1']))
    mapped = [SimpleNamespace(start_row_offset_idx=c['row_index'],start_col_offset_idx=c['column_index'],
                              row_span=c['row_span'],col_span=c['column_span'],text=c['raw_text'],
                              column_header=c['is_header'],bbox=bbox(c['bbox'])) for c in cells]
    return SimpleNamespace(pages={1:SimpleNamespace(size=SimpleNamespace(width=600,height=300))},
        tables=[SimpleNamespace(prov=[SimpleNamespace(page_no=1,bbox=bbox(dict(x0=10,y0=30,x1=590,y1=110)))],
                                data=SimpleNamespace(num_rows=3,num_cols=5,table_cells=mapped))])


def test_mapper_requires_original_source_and_still_rejects_body_overlaps():
    pdf, cells = fixture()
    model = as_document(cells)
    with pytest.raises(TableStructureError, match='Overlapping'):
        map_document(model,4)
    result = map_document(model,4,native_page=pdf[0])
    assert result[0]['rows'] == [HEADERS] + VALUES
    assert result[0]['cells'][3]['bbox']['x0'] > result[0]['cells'][2]['bbox']['x1']
    model.tables[0].data.table_cells.append(model.tables[0].data.table_cells[7])
    with pytest.raises(TableStructureError, match='Overlapping'):
        map_document(model,4,native_page=pdf[0])
