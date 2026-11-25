"""Conservative native-only recovery for one unmerged bank transaction layout.

Headers establish columns; dates establish rows. No missing values, continued
rows, merged cells or financial meaning are inferred. Uncertainty abstains.
"""
from datetime import datetime
import re

from app.services.table_native_geometry import refine_native_highlights

MAX_WORDS = 5000
MAX_ROWS = 200
HEADER = ['Date', 'Description', 'Money', 'out', 'Money', 'in', 'Balance']
NUMBER = r'(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}'
MONEY = re.compile(rf'(?:-?\$?{NUMBER}|\(\$?{NUMBER}\))')
DATE_START = re.compile(r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d+[/-])')


def _lines(words):
    lines = []
    for word in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        center = (word[1] + word[3]) / 2
        if not lines or abs(center - lines[-1][0]) > 2:
            lines.append((center, []))
        lines[-1][1].append(word)
    return [sorted(line, key=lambda w: w[0]) for _, line in lines]


def recover_bank_table(page, page_index):
    # Display rotation / invisible overlays need a separately qualified reader.
    if page.rotation:
        return []
    words = page.get_text('words')
    if not words or len(words) > MAX_WORDS:
        return []
    lines = _lines(words)
    headers = [i for i, line in enumerate(lines) if [w[4] for w in line] == HEADER]
    if len(headers) != 1:
        return []
    start = headers[0]
    h = lines[start]
    if any(a[2] > b[0]+.1 for a, b in zip(h, h[1:])):
        return []
    edges = [h[0][0]-2, h[1][0]-2, h[2][0]-2,
             (h[3][2]+h[4][0])/2, (h[5][2]+h[6][0])/2, h[6][2]+2]
    if any(b <= a for a, b in zip(edges, edges[1:])):
        return []
    groups = [[h[:1], h[1:2], h[2:4], h[4:6], h[6:]]]
    ended = False
    for line in lines[start+1:]:
        date_words = [w for w in line if w[0] < edges[1]]
        date_text = ' '.join(w[4] for w in date_words)
        try:
            datetime.strptime(date_text, '%b %d, %Y')
        except ValueError:
            # A malformed date, amount-only continuation, or split financial
            # row cannot be silently dropped as if it were footer prose.
            if DATE_START.match(date_text) or any(MONEY.fullmatch(w[4]) for w in line):
                return []
            # Only a continuous, full-width prose line establishes the end.
            # A final wrapped description or damaged last transaction must
            # not disappear simply because no later date follows it.
            height = max(w[3]-w[1] for w in line)
            if not ended and (line[0][0] > edges[0]+4 or line[-1][2] <= edges[2]
                              or any(b[0]-a[2] > 2*height for a, b in zip(line, line[1:]))):
                return []
            ended = True
            continue
        if ended or len(groups) > MAX_ROWS:
            return []
        # Two words painted over one another are not a readable description,
        # even if both fit inside the same inferred cell rectangle.
        if any(a[2] > b[0]+.1 for a, b in zip(line, line[1:])):
            return []
        row = [[] for _ in range(5)]
        for w in line:
            owners = [c for c in range(5) if edges[c] <= w[0] and w[2] <= edges[c+1]]
            if len(owners) != 1:
                return []
            row[owners[0]].append(w)
        texts = [' '.join(w[4] for w in col) for col in row]
        if (not texts[1] or not MONEY.fullmatch(texts[4])
                or sum(bool(texts[c]) for c in (2, 3)) != 1
                or any(texts[c] and not MONEY.fullmatch(texts[c]) for c in (2, 3))):
            return []
        groups.append(row)
    if len(groups) < 3:
        return []
    rows, cells = [], []
    for r, row in enumerate(groups):
        rows.append([' '.join(w[4] for w in col) for col in row])
        all_words = [w for col in row for w in col]
        y0, y1 = min(w[1] for w in all_words)-.1, max(w[3] for w in all_words)+.1
        for c, col in enumerate(row):
            bounds = dict(zip(('x0', 'y0', 'x1', 'y1'), (edges[c], y0, edges[c+1], y1)))
            cells.append({'page_index': page_index, 'row_index': r, 'column_index': c,
                          'row_span': 1, 'column_span': 1, 'is_header': r == 0,
                          'text': rows[r][c], 'raw_text': rows[r][c], 'bbox': bounds,
                          'source': 'native_pdf', 'geometry_status': 'available'})
    refine_native_highlights(page, cells)
    # This also rejects invisible, overlapping or painted-over native text.
    if any(c['text'] and c['geometry_kind'] != 'native_text' for c in cells):
        return []
    return [{'page_index': page_index, 'table_index': 0, 'page_width': page.rect.width,
             'page_height': page.rect.height, 'bbox': {
                 'x0': edges[0], 'y0': cells[0]['cell_bbox']['y0'],
                 'x1': edges[-1], 'y1': cells[-1]['cell_bbox']['y1']},
             'rows': rows, 'cells': cells, 'source': 'native_pdf',
             'structure_method': 'native_bank_header', 'warnings': [],
             'row_count': len(rows), 'col_count': 5}]
