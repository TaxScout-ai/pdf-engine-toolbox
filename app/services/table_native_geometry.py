"""Highlight complete native words within an already identified logical cell.

The original grid rectangle remains available as cell_bbox. This only refines
source geometry: it never changes values, logical spans, or tax-field approval.
"""
from bisect import bisect_left
from collections import Counter
import math

import fitz
import structlog

from app.services.table_evidence import MAX_CELLS, pdf_bbox

MAX_WORDS = 50_000
MAX_CANDIDATE_CHECKS = 1_000_000
log = structlog.get_logger()


def refine_native_highlights(page, cells):
    for cell in cells:
        cell['cell_bbox'] = dict(cell['bbox']) if cell.get('bbox') else None
        cell['geometry_kind'] = 'cell_region' if cell.get('bbox') else 'unavailable'
    if not cells or len(cells) > MAX_CELLS:
        return
    try:
        _refine(page, cells)
    except MemoryError:
        raise
    except Exception as exc:
        # Refinement failure must not erase an otherwise usable native table.
        for cell in cells:
            cell['bbox'] = cell['cell_bbox']
            cell['geometry_kind'] = 'cell_region' if cell['bbox'] else 'unavailable'
        log.warning('native_cell_geometry_unresolved', error_type=type(exc).__name__)


def _refine(page, cells):
    # Hidden overlays and text that is not upright on the displayed page cannot
    # establish a precise visible source. Keep the existing grid evidence.
    angle = math.radians(page.rotation)
    for span in page.get_texttrace():
        dx, dy = span['dir']
        if (span['type'] == 3 or span['opacity'] != 1 or
                dx * math.cos(angle) - dy * math.sin(angle) < .999):
            return
    words = page.get_text('words')
    if len(words) > MAX_WORDS:
        return
    displayed = [(fitz.Rect(w[:4]) * page.rotation_matrix, w[4]) for w in words]
    displayed.sort(key=lambda w: ((w[0].y0 + w[0].y1) / 2, w[0].x0))
    centers = [(b.y0 + b.y1) / 2 for b, _ in displayed]
    assignments = []
    candidate_checks = 0
    for cell in cells:
        b = cell.get('cell_bbox')
        if not b or not cell['text'].strip():
            assignments.append([])
            continue
        # Include crossing words rather than clipping their signs or digits at
        # the cell edge. Y uses the baseline band because font boxes can extend
        # above/below a tightly ruled row; the final highlight retains them whole.
        lo, hi = bisect_left(centers, b['y0']), bisect_left(centers, b['y1'])
        candidate_checks += hi - lo
        if candidate_checks > MAX_CANDIDATE_CHECKS:
            return
        assignments.append([i for i in range(lo, hi)
                            if displayed[i][0].x0 < b['x1'] and displayed[i][0].x1 > b['x0']])
    owners = Counter(i for assignment in assignments for i in assignment)
    for cell, indices in zip(cells, assignments):
        if not indices or any(owners[i] != 1 for i in indices):
            continue
        words = [displayed[i] for i in indices]
        # Ignore whitespace only. Currency, signs, decimal punctuation and
        # repeated tokens must all survive this exact comparison.
        text = ''.join(''.join(t.split()) for _, t in words)
        if text != ''.join(cell['raw_text'].split()):
            continue
        region = cell['cell_bbox']
        bounds = (min(b.x0 for b, _ in words), min(b.y0 for b, _ in words),
                  max(b.x1 for b, _ in words), max(b.y1 for b, _ in words))
        if bounds[0] < region['x0'] or bounds[2] > region['x1']:
            continue
        bbox = pdf_bbox(bounds, (0, 0), (1, 1), page.rect.width, page.rect.height)
        if bbox is not None:
            cell['bbox'] = bbox
            cell['geometry_kind'] = 'native_text'
