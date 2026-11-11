"""Source-only financial boundary repairs; uncertain cells still require review."""
from collections import Counter
from copy import deepcopy
import math
import re
from statistics import median

import fitz

from app.services.table_native_rows import _rect, _compact, _number
from app.services.table_evidence import MAX_CELLS


def _box(words):
    return fitz.Rect(min(w[0].x0 for w in words), min(w[0].y0 for w in words),
                     max(w[0].x1 for w in words), max(w[0].y1 for w in words))


def _inside(bounds, words):
    return [w for w in words if bounds.contains((w[0].tl + w[0].br) / 2)] if bounds else []


def _text(words):
    return ' '.join(w[1] for w in sorted(words, key=lambda w: (round(w[0].y0, 1), w[0].x0)))


def _cell(cell, words):
    b = _box(words)
    return dict(cell, text=_text(words), raw_text=_text(words),
                bbox=dict(x0=b.x0, y0=b.y0, x1=b.x1, y1=b.y1), geometry_status='available')


def _lines(words):
    groups = []
    for w in sorted(words, key=lambda w: (w[0].y0+w[0].y1)/2):
        center = (w[0].y0+w[0].y1)/2
        if groups and abs(center - groups[-1][0]) <= .8:
            groups[-1][1].append(w)
        else:
            groups.append((center, [w]))
    return [g[1] for g in groups]


def _split_empty_terminal_column(table, words):
    headers = [c for c in table['cells'] if c.get('is_header')]
    combined = [c for c in headers if ' '.join(c['text'].split()) ==
                'Wash Sale Loss Disallowed (Box 1g) Proceeds from C/Q']
    if not combined:
        return
    c = combined[0]
    fail = 'native_merged_financial_columns'
    if (table['row_count']*(table['col_count']+1)>MAX_CELLS or
            len(combined) != 1 or c['column_index'] != table['col_count']-1 or
            c['column_span'] != 1 or c['row_span'] != 1):
        table['warnings'].append(fail)
        return
    source = _inside(_rect(c['bbox']), words)
    if Counter(w[1] for w in source) != Counter(c['raw_text'].split()):
        table['warnings'].append(fail)
        return
    # The second printed heading is exactly these three words; its column must
    # be wholly empty below the heading. No body value is moved or invented.
    right = [w for w in source if w[1] in ('Proceeds', 'from', 'C/Q')]
    left = [w for w in source if w not in right]
    if len(right) != 3 or not left or _box(left).x1 >= _box(right).x0:
        table['warnings'].append(fail)
        return
    boundary = (_box(left).x1 + _box(right).x0)/2
    bounds = _rect(table['bbox'])
    below = [w for w in _inside(bounds, words) if w[0].y0 > _box(source).y1]
    body = [x for x in table['cells'] if x is not c and
            x['column_index']+x['column_span'] > c['column_index']]
    if any((w[0].x0+w[0].x1)/2 >= boundary for w in below):
        table['warnings'].append(fail)
        return
    replacements = []
    for x in body:
        evidence = _inside(_rect(x['bbox']),words)
        if (x['column_span'] != 1 or not evidence or
                _compact(_text(evidence)) != _compact(x['raw_text']) or
                _box(evidence).x1 >= boundary):
            table['warnings'].append(fail)
            return
        replacements.append((x,_cell(x,evidence)))
    for old,new in replacements:
        table['cells'][table['cells'].index(old)] = new
    table['cells'].remove(c)
    table['cells'].extend([_cell(c,left), _cell(dict(c,column_index=table['col_count']),right)])
    table['col_count'] += 1


def _remove_proven_rules(table, words):
    bounds = _rect(table['bbox'])
    rules = []
    for line in _lines(_inside(bounds, words)):
        if len(line) < 20 or any(w[1] != '-' for w in line):
            continue
        ordered = sorted(line,key=lambda w:w[0].x0)
        heights = median(w[0].height for w in line)
        if (_box(line).width < bounds.width*.6 or
                any(b[0].x0-a[0].x1 > heights*1.5 for a,b in zip(ordered,ordered[1:]))):
            continue
        rules.extend(line)
    if not rules:
        return
    for i,c in enumerate(table['cells']):
        if c.get('is_header') or c['row_span'] != 1 or c['column_span'] != 1:
            continue
        source = _inside(_rect(c['bbox']),words)
        if Counter(w[1] for w in source) != Counter(c['raw_text'].split()):
            continue
        kept = [w for w in source if w not in rules]
        removed = [w for w in source if w in rules]
        if not removed or not kept or not _number(_text(kept)):
            continue
        # A minus on the amount's own baseline is never a separator.
        if len(_lines(kept)) != 1 or abs((_box(kept).y0+_box(kept).y1-
                                       _box(removed).y0-_box(removed).y1)/2) < _box(kept).height*.5:
            continue
        table['cells'][i] = _cell(c,kept)


def _recover_summary(table, words):
    pattern = r'(Box [A-F] (?:Short|Long)-Term Realized) (Gain|Loss)'
    summary = [(c,re.fullmatch(pattern,c['text'])) for c in table['cells']]
    summary = [(c,m) for c,m in summary if m]
    if (table['row_count']+1)*table['col_count']>MAX_CELLS or len(summary) != 1:
        return
    c, match = summary[0]
    if c['row_index'] != table['row_count']-1 or c['row_span'] != 1:
        return
    amounts = [x for x in table['cells'] if x['row_index']==c['row_index'] and _number(x['text'])]
    if len(amounts) != 1 or amounts[0]['column_span'] != 1:
        return
    amount = amounts[0]
    gain_headers = [h for h in table['cells'] if h.get('is_header') and
                    h['column_index']==amount['column_index'] and
                    re.fullmatch(r'Gain/Loss(?: \(-\))?',h['text'])]
    if len(gain_headers)!=1:
        return
    header_source=_inside(_rect(gain_headers[0]['bbox']),words)
    if _compact(_text(header_source))!=_compact(gain_headers[0]['raw_text']):
        return
    source_label = _inside(_rect(c['bbox']),words)
    source_amount = _inside(_rect(amount['bbox']),words)
    if (_compact(_text(source_label)) != _compact(c['raw_text']) or
            _compact(_text(source_amount)) != _compact(amount['raw_text']) or not source_label or not source_amount):
        return
    prior = _box(source_label+source_amount)
    target = match[1] + (' Loss' if match[2]=='Gain' else ' Gain')
    candidates = []
    for line in _lines(_inside(_rect(table['bbox']),words)):
        b = _box(line)
        if not 0 < b.y0-prior.y0 < prior.height*1.5:
            continue
        numbers = [w for w in line if _number(w[1])]
        label = [w for w in line if w not in numbers]
        if len(numbers)==1 and _text(label)==target:
            candidates.append((label,numbers))
    if len(candidates)!=1:
        return
    label,number = candidates[0]
    lb,nb = _box(label),_box(number)
    if (abs(lb.x0-_box(source_label).x0)>1 or
            abs(nb.x1-_box(source_amount).x1)>1 or lb.x1>=nb.x0):
        return
    # Never duplicate evidence already assigned elsewhere in this table.
    if any(_inside(_rect(x['bbox']), label+number) for x in table['cells']):
        return
    next_row=table['row_count']
    table['cells'].extend([_cell(dict(c,row_index=next_row),label),
                           _cell(dict(amount,row_index=next_row),number)])
    table['row_count']+=1
    b = _rect(table['bbox']) | lb | nb
    table['bbox']=dict(x0=b.x0,y0=b.y0,x1=b.x1,y1=b.y1)


def refine_financial_boundaries(page, tables):
    angle=math.radians(page.rotation)
    trace=page.get_texttrace()
    if not trace or any(s['type']==3 or s['opacity']==0 or
                       s['dir'][0]*math.cos(angle)-s['dir'][1]*math.sin(angle)<.9 for s in trace):
        return tables,0
    words=[(fitz.Rect(w[:4])*page.rotation_matrix,w[4]) for w in page.get_text('words')]
    if len(words)>5000:
        return tables,0
    output,unresolved=[],0
    for original in tables:
        table=deepcopy(original)
        headers=[c for c in table['cells'] if c.get('is_header')]
        if not any(re.search(r'\b1d\b',c['text'],re.I) for c in headers):
            output.append(original)
            continue
        table.setdefault('warnings',[])
        _split_empty_terminal_column(table,words)
        _remove_proven_rules(table,words)
        _recover_summary(table,words)
        # Quantity/date columns containing header prose are not transactions,
        # even when their associated monetary values happen to be correct.
        scalar_cols={c['column_index'] for c in headers if c['text']=='Quantity' or
                     re.search(r'\b1[bc]\b',c['text'],re.I)}
        if any(c['column_index'] in scalar_cols and not c.get('is_header') and
               c['column_span']==1 and c['text'].strip() and
               len(_lines(_inside(_rect(c['bbox']),words)))>1 for c in table['cells']):
            table['warnings'].append('native_financial_row_contaminated')
        table['warnings']=list(dict.fromkeys(table['warnings']))
        if any(w.startswith('native_') for w in table['warnings']):
            unresolved+=1
        table['rows']=[['']*table['col_count'] for _ in range(table['row_count'])]
        for c in table['cells']:
            table['rows'][c['row_index']][c['column_index']]=c['text']
        output.append(table)
    return output,unresolved
