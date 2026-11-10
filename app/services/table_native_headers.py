"""Recover overlapping 1099-B headers only from complete native PDF evidence.

No model retries, inferred values or body-grid changes. Missing or ambiguous
source proof leaves the original overlap for the mapper to reject.
"""
from collections import Counter
from statistics import median
import re

import fitz

from app.services.table_native_rows import _rect, _compact


def recover_headers(page, cells, columns):
    headers = [c for c in cells if c['row_index'] == 0]
    body = [c for c in cells if c['row_index'] != 0]
    if (page.rotation or not 4 <= columns <= 32 or not 1 <= len(body) <= 1000 or
            not any(c['is_header'] for c in headers) or
            any(c['row_span'] != 1 for c in headers)):
        return cells
    occupied = set()
    overlap = False
    for c in headers:
        slots = set(range(c['column_index'], c['column_index'] + c['column_span']))
        overlap |= bool(slots & occupied)
        occupied |= slots
    if not overlap or occupied != set(range(columns)):
        return cells
    bounds = [_rect(c.get('bbox')) for c in headers]
    if not bounds or any(b is None for b in bounds):
        return cells
    # Searchable scan overlays or non-upright text are not printed proof.
    if any(s['type'] == 3 or s['opacity'] == 0 or s['dir'][0] < .99
           or abs(s['dir'][1]) > .01 for s in page.get_texttrace()):
        return cells
    band = fitz.Rect(min(b.x0 for b in bounds), min(b.y0 for b in bounds),
                     max(b.x1 for b in bounds), max(b.y1 for b in bounds))
    words = page.get_text('words')
    if len(words) > 5000:
        return cells
    selected = [w for w in words if band.contains(fitz.Point((w[0]+w[2])/2, (w[1]+w[3])/2))]
    if not selected or len(selected) > 2000:
        return cells
    # No source word may be dropped, duplicated, or supplied by the model.
    if Counter(w[4] for w in selected) != Counter(' '.join(c['raw_text'] for c in headers).split()):
        return cells
    gap = median(w[3] - w[1] for w in selected) * .35
    groups = []
    right = 0
    for w in sorted(selected, key=lambda w: w[0]):
        if groups and w[0] - right <= gap:
            groups[-1].append(w)
            right = max(right, w[2])
        else:
            groups.append([w])
            right = w[2]
    if len(groups) != columns:
        return cells
    boxes = [fitz.Rect(min(w[0] for w in g), min(w[1] for w in g),
                       max(w[2] for w in g), max(w[3] for w in g)) for g in groups]
    texts = [' '.join(w[4] for w in sorted(g, key=lambda w: (round(w[1], 1), w[0]))) for g in groups]
    markers = [re.findall(r'\b1[b-g](?=\s*[-–:])', t, re.I) for t in texts]
    markers = [[m.lower() for m in ms] for ms in markers]
    counts = Counter(m for ms in markers for m in ms)
    if (any(counts[m] != 1 for m in ('1b', '1c', '1d', '1e')) or
            any(n != 1 for n in counts.values()) or
            any(len(ms) > 1 and set(ms) != {'1f', '1g'} for ms in markers)):
        return cells
    edges = [0] + [(a.x1+b.x0)/2 for a, b in zip(boxes, boxes[1:])] + [page.rect.width]
    witnesses = [set() for _ in groups]
    row_lines = {}
    for c in body:
        if not c['text'].strip():
            continue
        b = _rect(c.get('bbox'))
        if b is None or b.y0 < band.y1:
            return cells
        if c['column_span'] != 1:
            continue  # Shared security/context rows stay exactly as supplied.
        if c['row_span'] != 1:
            return cells
        col = c['column_index']
        source = [w for w in words if b.contains(fitz.Point((w[0]+w[2])/2, (w[1]+w[3])/2))]
        if (not source or any(not edges[col] < (w[0]+w[2])/2 < edges[col+1] for w in source) or
                _compact(' '.join(w[4] for w in sorted(source, key=lambda w: (round(w[1], 1), w[0]))))
                != _compact(c['raw_text'])):
            return cells
        witnesses[col].add(c['row_index'])
        row_lines.setdefault(c['row_index'], []).extend(source)
    previous_bottom = None
    for row in sorted(row_lines):
        evidence = row_lines[row]
        centers = [(w[1] + w[3])/2 for w in evidence]
        if max(centers)-min(centers) > median(w[3]-w[1] for w in evidence)*.75:
            return cells
        if previous_bottom is not None and min(centers) <= previous_bottom:
            return cells
        previous_bottom = max(centers)
    for col, ms in enumerate(markers):
        required = 2 if set(ms) & {'1d', '1e'} else 1 if set(ms) & {'1b', '1c'} else 0
        if len(witnesses[col]) < required:
            return cells
    replacement = []
    for col, (b, text) in enumerate(zip(boxes, texts)):
        replacement.append({**headers[0], 'column_index': col, 'column_span': 1,
                            'row_span': 1, 'is_header': True, 'text': text, 'raw_text': text,
                            'bbox': dict(x0=b.x0, y0=b.y0, x1=b.x1, y1=b.y1),
                            'geometry_status': 'available'})
    return replacement + body
