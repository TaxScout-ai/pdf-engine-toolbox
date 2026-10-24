#!/usr/bin/env python3
"""Evaluate privately labelled table regions without writing their values to a report.

Golden cells carry row/column, text, tight bbox and a manually marked cell
region. Match geometry FIRST, then compare values; matching by value would
hide the very wrong-column/duplicate-amount failures this evaluation targets.
This is a labelled-region evaluation, not whole-document precision.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re


def number(text):
    s = str(text).strip().replace("−", "-").replace("–", "-")
    s = re.sub(r"[tD†*]+$", "", s).strip()
    negative = s.startswith("(") and s.endswith(")")
    if negative:
        s = s[1:-1]
    # A merged cell containing two numbers is not one correctly read value.
    if re.search(r"\d\s+\d", s):
        return None
    s = re.sub(r"[$,\s]", "", s)
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return None
    try:
        value = Decimal(s)
        return -value if negative else value
    except InvalidOperation:
        return None


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def overlap(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def box(value):
    if not value:
        return None
    return [value[k] for k in ("x0", "y0", "x1", "y1")] if isinstance(value, dict) else value


def evaluate(golden, result):
    predictions = []
    for ti, table in enumerate(result.get("tables", [])):
        for ci, cell in enumerate(table.get("cells", [])):
            bounds = box(cell.get("bbox"))
            if bounds and area(bounds) > 0:
                predictions.append((ti, ci, cell, bounds))
    counts = Counter(pages=1, labelled_rows=golden["rows"])
    used = set()
    located_rows = set()
    columns = defaultdict(list)
    for expected in golden["cells"]:
        value = number(expected["text"])
        if value is None:
            continue
        counts["numeric_cells"] += 1
        monetary = bool(re.search(
            r"amount|proceeds|basis|gain|loss|discount|wash|dividend|withheld|interest|"
            r"total|value|income|tax|expenses|adjustment|period|year-to-date|change|price|^20\d\d",
            expected["column_label"], re.I))
        counts["money_cells"] += monetary
        candidates = []
        for ti, ci, cell, bounds in predictions:
            if (ti, ci) in used:
                continue
            region, tight = expected["region"], expected["bbox"]
            cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
            inside = region[0] - 1 <= cx <= region[2] + 1 and region[1] - 1 <= cy <= region[3] + 1
            coverage = overlap(bounds, tight) / max(area(tight), 1)
            if (inside or coverage >= .7) and area(bounds) <= 4 * max(area(region), 1):
                score = coverage + (1 if inside else 0) - area(bounds) / max(area(region), 1) * .01
                candidates.append((score, ti, ci, cell))
        if not candidates:
            counts["missing_numeric_cells"] += 1
            continue
        _, ti, ci, candidate = max(candidates, key=lambda c: c[0])
        used.add((ti, ci))
        located_rows.add(expected["row"])
        counts["located_numeric_cells"] += 1
        columns[(ti, candidate["column_index"])].append(expected["column"])
        if candidate.get("column_span", 1) > 1:
            counts["numeric_cells_in_merged_columns"] += 1
        found = number(candidate.get("raw_text", candidate.get("text", "")))
        if found == value:
            counts["exact_numeric_cells"] += 1
            counts["exact_money_cells"] += monetary
        else:
            counts["wrong_numeric_cells"] += 1
            counts["wrong_money_cells"] += monetary
        if found is not None and abs(found) == abs(value) and value != 0:
            counts["sign_comparable_cells"] += 1
            counts["wrong_sign_cells"] += (found < 0) != (value < 0)
    numeric_rows = {c["row"] for c in golden["cells"] if number(c["text"]) is not None}
    counts["labelled_numeric_rows"] = len(numeric_rows)
    counts["missing_numeric_rows"] = len(numeric_rows - located_rows)
    # A predicted logical column assigned to different labelled columns is
    # inconsistent even if all digits happen to be right. Absolute column
    # indices are not comparable across split tables and repeated headers.
    counts["column_conflicts"] = sum(len(cs) - Counter(cs).most_common(1)[0][1]
                                      for cs in columns.values())
    return dict(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("golden", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    totals = Counter()
    pages = []
    for expected in json.loads(args.golden.read_text()):
        if not expected.get("reviewed"):
            raise ValueError(f"Unreviewed ground truth: {expected['id']}")
        result = json.loads((args.results / f"{expected['id']}{args.suffix}.json").read_text())
        metrics = evaluate(expected, result)
        totals.update(metrics)
        pages.append({"id": expected["id"], **metrics})
    args.output.write_text(json.dumps({"scope": "labelled financial regions only",
                                      "totals": dict(totals), "pages": pages}, indent=2) + "\n")
    print(json.dumps(dict(totals), sort_keys=True))


if __name__ == "__main__":
    main()
