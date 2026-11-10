"""Recover source-supported native transaction rows, never isolated amounts.

TableFormer can join two transactions while preserving every digit. Splitting
only their amounts leaves dates/security ownership attached to the wrong row.
This conservative adapter requires two date columns, every numeric column and
one explicitly shared CUSIP description before expanding a transaction row.
It neither changes values nor establishes taxpayer/year/fact approval.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import math
import re

import fitz

from app.services.table_evidence import MAX_CELLS, pdf_bbox

_NUMBER = re.compile(r"\(?-?\$?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?\)?")
_CUSIP = re.compile(r"\bCUSIP\s*:\s*[A-Z0-9]{9}\b", re.I)


def _compact(text):
    return re.sub(r"\s+", "", text).casefold()


def _number(text):
    text = text.strip()
    return bool(_NUMBER.fullmatch(text) and text.startswith('(') == text.endswith(')')
                and not (text.startswith('(') and '-' in text))


def _date(text):
    if not re.fullmatch(r"\d{2}/\d{2}/(?:\d{2}|\d{4})", text):
        return False
    try:
        datetime.strptime(text, '%m/%d/%Y' if len(text) == 10 else '%m/%d/%y')
        return True
    except ValueError:
        return False


def _rect(bounds):
    if not bounds:
        return None
    try:
        values = [bounds[k] for k in ('x0', 'y0', 'x1', 'y1')]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            return None
        box = fitz.Rect(values)
        return box if box.width > 0 and box.height > 0 else None
    except (KeyError, TypeError, ValueError):
        return None


def _lines(cell, words):
    bounds = _rect(cell.get('bbox'))
    if bounds is None:
        return []
    groups = defaultdict(list)
    for rect, text, block, line in words:
        if bounds.contains((rect.tl + rect.br) / 2):
            groups[(block, line)].append((rect, text))
    lines = []
    for group in groups.values():
        group.sort(key=lambda w: w[0].x0)
        box = fitz.Rect(group[0][0])
        for rect, _ in group[1:]:
            box |= rect
        lines.append((box, ' '.join(text for _, text in group)))
    lines.sort(key=lambda line: (line[0].y0, line[0].x0))
    if _compact(' '.join(text for _, text in lines)) != _compact(cell.get('raw_text', cell['text'])):
        return []
    return lines


def _profile(headers):
    roles = {}
    boxes = []
    for cell in headers:
        text = ' '.join(cell['text'].lower().split())
        role = None
        match = re.search(r'\(box (1[a-g])\)', text)
        box = match.group(1) if match else None
        if box:
            boxes.append(box)
        if 'cusip' in text and box == '1a':
            role = 'security'
        elif 'date' in text and box in ('1b', '1c'):
            role = 'date'
        elif box in ('1d', '1e', '1f', '1g') or text in ('quantity sold', 'gain/loss amount'):
            role = 'number'
        elif text == 'additional information':
            role = 'information'
        if role is None:
            return None
        roles[cell['column_index']] = role
    if (len(boxes) != len(set(boxes)) or not set(('1a', '1b', '1c', '1d', '1e')).issubset(boxes) or
            list(roles.values()).count('security') != 1 or
            list(roles.values()).count('date') != 2 or
            list(roles.values()).count('number') < 2):
        return None
    return roles


def _separates_group(page, bounds, baselines):
    for path in page.get_drawings():
        for item in path['items']:
            if item[0] == 'l':
                segments = [(item[1], item[2])]
            elif item[0] == 're':
                rect = item[1]
                segments = [(rect.tl, rect.tr), (rect.bl, rect.br)]
            else:
                continue
            for start, end in segments:
                start, end = start * page.rotation_matrix, end * page.rotation_matrix
                if (abs(start.y - end.y) < .5 and min(baselines) < start.y < max(baselines)
                        and min(start.x, end.x) <= bounds.x0 + 1
                        and max(start.x, end.x) >= bounds.x1 - 1):
                    return True
    return False


def _split_row(cells, roles, words, page, headers):
    """Return expanded logical cells, or None when any association is unknown."""
    by_col = {c['column_index']: c for c in cells}
    if len(by_col) != len(cells) or any(c.get('row_span', 1) != 1 or c.get('column_span', 1) != 1 for c in cells):
        return None
    ordered = sorted((c for c in cells if c['text'].strip()), key=lambda c: c['column_index'])
    bounds = [_rect(c.get('bbox')) for c in ordered]
    if any(b is None for b in bounds) or any(a.x1 >= b.x0 for a, b in zip(bounds, bounds[1:])):
        return None
    # Bind body cells to the same printed columns. Header-center nearest-
    # neighbour matching is unsafe for wide description columns. Use the gaps
    # between disjoint source header extents; decline ambiguous layouts.
    head_bounds = [_rect(c['bbox']) for c in sorted(headers, key=lambda c: c['column_index'])]
    if any(a.x1 >= b.x0 for a, b in zip(head_bounds, head_bounds[1:])):
        return None
    edges = [0] + [(a.x1 + b.x0) / 2 for a, b in zip(head_bounds, head_bounds[1:])] + [page.rect.width]
    for cell in cells:
        box = _rect(cell.get('bbox'))
        col = cell['column_index']
        # A wrapped security name can extend into the whitespace before the
        # next heading. It still must end before that heading and body column.
        right = head_bounds[col + 1].x0 if roles[col] == 'security' and col + 1 < len(head_bounds) else edges[col + 1]
        if box and (box.x0 < edges[col] - 1 or box.x1 > right + 1):
            return None
    split = {}
    shared = []
    baselines = None
    for col, role in roles.items():
        cell = by_col.get(col)
        if role == 'information':
            # Do not replicate a restriction/annotation across transactions.
            if cell and cell['text'].strip():
                return None
            bounds = _rect(cell.get('bbox')) if cell else None
            if bounds and any(bounds.contains((b.tl + b.br) / 2) for b, _, _, _ in words):
                return None
            continue
        if cell is None:
            return None
        lines = _lines(cell, words)
        if not lines:
            return None
        if role == 'security':
            if len(_CUSIP.findall(cell['text'])) != 1 or len(re.findall(r'\bCUSIP\b', cell['text'], re.I)) != 1:
                return None
            # Only the observed description-then-terminal-CUSIP grammar proves
            # one shared block. Text after the identifier could start another
            # security whose identifier was omitted; never guess ownership.
            if len(lines) < 2 or not _CUSIP.fullmatch(lines[-1][1]):
                return None
            shared.append(cell)
            continue
        validator = _date if role == 'date' else _number
        if not 2 <= len(lines) <= 4 or not all(validator(text) for _, text in lines):
            return None
        centers = [(box.y0 + box.y1) / 2 for box, _ in lines]
        if any(b - a < 4 for a, b in zip(centers, centers[1:])):
            return None
        if baselines is None:
            baselines = centers
        elif len(centers) != len(baselines) or any(abs(a - b) > 2 for a, b in zip(centers, baselines)):
            return None
        split[col] = lines
    if baselines is None:
        return None
    count = len(baselines)
    for cell in shared:
        source_lines = _lines(cell, words)
        top = min(b.y0 for b, _ in source_lines)
        bottom = max(b.y1 for b, _ in source_lines)
        if min(baselines) < top - 1 or max(baselines) > bottom + 1:
            return None
    if any(_separates_group(page, _rect(c['bbox']), baselines) for c in shared):
        return None
    # Unrepresented text is never silently discarded. Source words in the
    # entire row band must belong to one of its original physical cells.
    rects = [_rect(c.get('bbox')) for c in cells if roles[c['column_index']] != 'information']
    rects = [b for b in rects if b is not None]
    if not rects:
        return None
    band = fitz.Rect(0, min(b.y0 for b in rects), page.rect.width, max(b.y1 for b in rects))
    for rect, _, _, _ in words:
        center = (rect.tl + rect.br) / 2
        if band.contains(center) and sum(b.contains(center) for b in rects) != 1:
            return None
    result = []
    for cell in shared:
        result.append({**deepcopy(cell), 'row_index': 0, 'row_span': count})
    for col, lines in split.items():
        cell = by_col[col]
        for offset, (bounds, text) in enumerate(lines):
            bbox = pdf_bbox(list(bounds), (0, 0), (1, 1), page.rect.width, page.rect.height)
            if bbox is None:
                return None
            result.append({**deepcopy(cell), 'row_index': offset, 'row_span': 1,
                           'raw_text': text, 'text': text, 'bbox': bbox,
                           'geometry_status': 'available'})
    return count, result


def refine_native_tables(page, tables):
    """Return refined tables and count of native structures requiring review.

    Scanned/hidden/rotated source text is not used to manufacture row evidence.
    Header-order contradictions are reported even when row recovery abstains.
    """
    trace = page.get_texttrace()
    angle = math.radians(page.rotation)
    if not trace:
        return tables, 0
    unsafe_spans = [s for s in trace if s['type'] == 3 or s['opacity'] == 0 or
                    s['dir'][0] * math.cos(angle) - s['dir'][1] * math.sin(angle) < .9]
    unsafe_bounds = [fitz.Rect(s['bbox']) * page.rotation_matrix for s in unsafe_spans]
    words = [(fitz.Rect(w[:4]) * page.rotation_matrix, w[4], w[5], w[6])
             for w in page.get_text('words')]
    words = [w for w in words if not any(b.contains((w[0].tl + w[0].br) / 2) for b in unsafe_bounds)]
    output, unresolved = [], 0
    for original in tables:
        table = deepcopy(original)
        headers = [c for c in table['cells'] if c.get('is_header')]
        supported = [c for c in headers if c.get('row_span', 1) == c.get('column_span', 1) == 1 and _lines(c, words)]
        # A demonstrable contradiction survives missing unrelated headers or
        # hidden text elsewhere. Unknown headers cannot erase known evidence.
        if any(a['row_index'] == b['row_index'] and a['column_index'] < b['column_index']
               and _rect(a['bbox']).x0 >= _rect(b['bbox']).x1 for a in supported for b in supported):
            unresolved += 1
            continue
        if unsafe_spans:
            output.append(table)
            continue
        if (len(headers) != table['col_count'] or
                {c['column_index'] for c in headers} != set(range(table['col_count'])) or
                len({c['row_index'] for c in headers}) != 1 or
                any(c.get('row_span', 1) != 1 or c.get('column_span', 1) != 1 or not _lines(c, words) for c in headers)):
            output.append(table)
            continue
        ordered = sorted(headers, key=lambda c: (_rect(c['bbox']).x0 + _rect(c['bbox']).x1) / 2)
        if [c['column_index'] for c in ordered] != list(range(table['col_count'])):
            unresolved += 1
            continue
        roles = _profile(headers)
        if roles is None:
            output.append(table)
            continue
        rows = defaultdict(list)
        for cell in table['cells']:
            rows[cell['row_index']].append(cell)
        merged_rows = {row for row, cells in rows.items() if any(
            roles.get(c['column_index']) == 'date' and not c.get('is_header')
            and len(_lines(c, words)) > 1 for c in cells)}
        if merged_rows and any(c.get('row_span', 1) != 1 for c in table['cells']):
            unresolved += 1
            continue
        rebuilt, next_row, recovered = [], 0, 0
        for row in range(table['row_count']):
            cells = rows[row]
            # Recover only a demonstrably merged date row. Totals and a single
            # transaction with a wrapped description are left unchanged.
            merged = row in merged_rows
            expansion = _split_row(cells, roles, words, page, headers) if merged else None
            if merged and expansion is None:
                unresolved += 1
                recovered = -2
                break
            if expansion:
                count, new_cells = expansion
                rebuilt.extend({**c, 'row_index': next_row + c['row_index']} for c in new_cells)
                next_row += count
                recovered += 1
            else:
                if any(c.get('row_span', 1) != 1 for c in cells):
                    recovered = -1
                    break
                rebuilt.extend({**c, 'row_index': next_row} for c in cells)
                next_row += 1
        if recovered > 0 and next_row * table['col_count'] > MAX_CELLS:
            unresolved += 1
            continue
        if recovered > 0:
            grid = [[''] * table['col_count'] for _ in range(next_row)]
            for cell in rebuilt:
                grid[cell['row_index']][cell['column_index']] = cell['text']
            table.update(cells=rebuilt, rows=grid, row_count=next_row)
        if recovered == -2:
            continue
        output.append(table)
    return output, unresolved
