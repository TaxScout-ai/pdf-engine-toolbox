"""Detect strong evidence of financial rows outside recognized native tables.

This is an abstention check, not another table extractor or proof of complete
coverage. Preserve every recognized cell; never infer a missing amount or zero.
"""
import re
from collections import Counter

import fitz

from app.services.table_native_geometry import MAX_WORDS, MAX_CANDIDATE_CHECKS

MONEY = re.compile(r"(?:[-−–]?\$?(?:\d+|\d{1,3}(?:,\d{3})+)\.\d{2}|\(\$?(?:\d+|\d{1,3}(?:,\d{3})+)\.\d{2}\))[†*]?")
REASON = 'native_financial_rows_omitted'


def coverage_reason(page, tables) -> str | None:
    if not tables:
        return None
    words = page.get_text('words')
    if len(words) > MAX_WORDS:
        return 'native_financial_coverage_limit'
    rotation = page.rotation_matrix
    bounds = [fitz.Rect(*(t['bbox'][k] for k in ('x0', 'y0', 'x1', 'y1')))
              for t in tables]
    money = [(fitz.Rect(w[:4]) * rotation, w[4]) for w in words if MONEY.fullmatch(w[4])]
    if not money:
        return None
    # A hidden duplicate must not erase a visible original. Match displayed
    # text objects, and ignore only a later opaque rectangle that covers the
    # whole amount. Thin rules and transparent paint do not hide digits.
    traces = [(fitz.Rect(s['bbox']) * rotation, s['seqno'],
               ''.join(chr(c[0]) for c in s['chars']))
              for s in page.get_texttrace() if s['type'] != 3 and s['opacity'] > 0]
    opaque = [(fitz.Rect(d['items'][0][1]) * rotation, d['seqno'])
              for d in page.get_drawings()
              if d.get('fill') is not None and d.get('fill_opacity') == 1
              and len(d['items']) == 1 and d['items'][0][0] == 're']
    prose_words = [(fitz.Rect(w[:4]) * rotation, w[4]) for w in words
                   if any(c.isalpha() for c in w[4])]
    work = ((len(money) + len(prose_words)) * (len(traces) + len(opaque))
            + len(money) * (len(bounds) + len(prose_words)))
    if work > MAX_CANDIDATE_CHECKS:
        return 'native_financial_coverage_limit'
    def visible_word(box, text):
        center = (box.tl + box.br) / 2
        visible = [seq for region, seq, content in traces if center in region and text in content]
        latest_visible = max(visible, default=-1)
        return (latest_visible >= 0 and center in page.rect
                and not any(seq > latest_visible and box in region for region, seq in opaque))

    outside = []
    seen = set()
    for box, text in money:
        center = (box.tl + box.br) / 2
        key = (*box, text)
        if (any(center in b for b in bounds) or key in seen or not visible_word(box, text)):
            continue
        seen.add(key)
        outside.append(box)
    # PDF creators often put each column in its own text block. Group by the
    # displayed baseline band, not PDF block/line numbers or matching values.
    rows = []
    for box in sorted(outside, key=lambda b: ((b.y0 + b.y1) / 2, b.x0)):
        cy = (box.y0 + box.y1) / 2
        if not rows or abs(cy - rows[-1][0]) > max(2, box.height * .4):
            rows.append((cy, []))
        rows[-1][1].append(box)
    # Table amounts form trailing columns. Words between or after the amounts
    # indicate prose; require separation from the preceding description too.
    displayed_words = [(box, text) for box, text in prose_words if visible_word(box, text)]
    candidates = []
    for cy, boxes in rows:
        left = min(b.x0 for b in boxes)
        height = max(b.height for b in boxes)
        prose = [b for b, text in displayed_words if any(c.isalpha() for c in text)
                 and abs((b.y0 + b.y1) / 2 - cy) <= height * .4]
        if any(b.x1 > left for b in prose):
            continue
        if prose and left - max(b.x1 for b in prose) < height:
            continue
        candidates.append((cy, boxes))
    rows = candidates
    # Three rows with two aligned trailing amounts are strong tabular evidence.
    pairs = [(boxes[0].x1, boxes[-1].x1, row)
             for row, (_, boxes) in enumerate(rows) if len(boxes) >= 2]
    pairs.sort()
    checks = 0
    for start, (left, right, row) in enumerate(pairs):
        aligned = {row}
        for j in range(start + 1, len(pairs)):
            x, end, other_row = pairs[j]
            checks += 1
            if checks > MAX_CANDIDATE_CHECKS:
                return 'native_financial_coverage_limit'
            if x - left > 3:
                break
            if abs(end - right) <= 3:
                aligned.add(other_row)
            if len(aligned) >= 3:
                return REASON
    # One-column financial summaries also matter. Require four different rows
    # with aligned amount ends; one incidental fee in a paragraph is not a table.
    ends = sorted((b.x1, row) for row, (_, boxes) in enumerate(rows) for b in boxes)
    aligned = Counter()
    start = 0
    for right, row in ends:
        aligned[row] += 1
        while right - ends[start][0] > 3:
            old_row = ends[start][1]
            aligned[old_row] -= 1
            if not aligned[old_row]:
                del aligned[old_row]
            start += 1
        if len(aligned) >= 4:
            return REASON
    return None
