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
import math
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
    # Reject malformed grouping and double negatives instead of silently
    # turning e.g. 1,2 into 12 or (-100) into positive 100.
    s = s.removeprefix("$").strip()
    if s.startswith("-$"):
        s = "-" + s[2:]
    if negative and s.startswith("-"):
        return None
    if not re.fullmatch(r"-?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", s):
        return None
    s = s.replace(",", "")
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
    try:
        coords = [value[k] for k in ("x0", "y0", "x1", "y1")] if isinstance(value, dict) else value
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            return None
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in coords):
            return None
        bounds = [float(v) for v in coords]
        return bounds if all(math.isfinite(v) for v in bounds) and area(bounds) > 0 else None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def monetary_cell(cell):
    """Explicit labels win; old annotations retain the published denominator."""
    if "is_monetary" in cell:
        if not isinstance(cell["is_monetary"], bool):
            raise ValueError("is_monetary must be boolean")
        return cell["is_monetary"]
    return bool(re.search(
        r"amount|proceeds|basis|gain|loss|discount|wash|dividend|withheld|interest|"
        r"total|value|income|tax|expenses|adjustment|period|year-to-date|change|price|^20\d\d",
        cell["column_label"], re.I))


def page_index(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def evaluate(golden, result, *, details=None):
    predictions = []
    counts = Counter(pages=1, labelled_rows=golden["rows"])
    expected_page = golden.get("page_index", 0)
    if not page_index(expected_page):
        raise ValueError("Invalid labelled page index")
    for ti, table in enumerate(result.get("tables", [])):
        for ci, cell in enumerate(table.get("cells", [])):
            bounds = box(cell.get("bbox"))
            page = cell.get("page_index", table.get("page_index", 0))
            table_page = table.get("page_index", page)
            if not page_index(page) or not page_index(table_page):
                counts["invalid_page_predictions"] += 1
            elif page != expected_page or table_page != page:
                counts["wrong_page_predictions"] += 1
            elif bounds is None:
                counts["invalid_geometry_predictions"] += 1
            else:
                predictions.append((ti, ci, cell, bounds))
    used = set()
    located_rows = set()
    row_counts = defaultdict(Counter)
    columns = defaultdict(list)
    for expected in golden["cells"]:
        value = number(expected["text"])
        if value is None:
            continue
        counts["numeric_cells"] += 1
        monetary = monetary_cell(expected)
        row_counts[expected["row"]]["expected"] += 1
        region, tight = box(expected["region"]), box(expected["bbox"])
        if region is None or tight is None:
            raise ValueError("Invalid labelled cell geometry")
        detail = {"row": expected["row"], "column": expected["column"],
                  "page_index": expected_page, "monetary": monetary}
        counts["money_cells"] += monetary
        candidates = []
        for ti, ci, cell, bounds in predictions:
            if (ti, ci) in used:
                continue
            cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
            inside = region[0] - 1 <= cx <= region[2] + 1 and region[1] - 1 <= cy <= region[3] + 1
            coverage = overlap(bounds, tight) / max(area(tight), 1)
            if (inside or coverage >= .7) and area(bounds) <= 4 * max(area(region), 1):
                score = coverage + (1 if inside else 0) - area(bounds) / max(area(region), 1) * .01
                candidates.append((score, ti, ci, cell))
        if not candidates:
            counts["missing_numeric_cells"] += 1
            counts["missing_money_cells"] += monetary
            if details is not None:
                details.append({**detail, "outcome": "missing"})
            continue
        _, ti, ci, candidate = max(candidates, key=lambda c: c[0])
        ambiguous = len(candidates) > 1
        counts["ambiguous_numeric_associations"] += ambiguous
        bounds = box(candidate["bbox"])
        # Unlike the legacy association window, containment has no tolerance.
        contained = (bounds[0] >= region[0] and bounds[1] >= region[1]
                     and bounds[2] <= region[2] and bounds[3] <= region[3])
        counts["located_bbox_outside_cell_region"] += not contained
        used.add((ti, ci))
        located_rows.add(expected["row"])
        row_counts[expected["row"]]["located"] += 1
        counts["located_numeric_cells"] += 1
        columns[(ti, candidate["column_index"])].append(expected["column"])
        if candidate.get("column_span", 1) > 1:
            counts["numeric_cells_in_merged_columns"] += 1
        found = number(candidate.get("raw_text", candidate.get("text", "")))
        if found == value:
            counts["exact_numeric_cells"] += 1
            counts["exact_money_cells"] += monetary
            row_counts[expected["row"]]["exact"] += 1
            counts["exact_money_with_contained_bbox"] += monetary and contained
            outcome = "exact_value"
        else:
            counts["wrong_numeric_cells"] += 1
            counts["wrong_money_cells"] += monetary
            if found is None:
                outcome = "non_numeric_or_merged_text"
            elif value != 0 and found == -value:
                outcome = "wrong_sign"
            else:
                outcome = "wrong_numeric_value"
            counts["mismatch_" + outcome + "_cells"] += 1
        if details is not None:
            details.append({**detail, "outcome": outcome, "table_index": ti,
                            "cell_index": ci, "bbox_contained": contained,
                            "ambiguous_association": ambiguous})
        if found is not None and abs(found) == abs(value) and value != 0:
            counts["sign_comparable_cells"] += 1
            counts["wrong_sign_cells"] += (found < 0) != (value < 0)
    numeric_rows = {c["row"] for c in golden["cells"] if number(c["text"]) is not None}
    counts["fully_located_numeric_rows"] = sum(
        c["located"] == c["expected"] for c in row_counts.values())
    counts["fully_exact_numeric_rows"] = sum(
        c["exact"] == c["expected"] for c in row_counts.values())
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
    parser.add_argument("--details", type=Path,
                        help="Write cell outcomes/indices without document text or amounts")
    args = parser.parse_args()
    totals = Counter()
    pages = []
    cell_details = []
    for expected in json.loads(args.golden.read_text()):
        if not expected.get("reviewed"):
            raise ValueError(f"Unreviewed ground truth: {expected['id']}")
        result = json.loads((args.results / f"{expected['id']}{args.suffix}.json").read_text())
        details = []
        metrics = evaluate(expected, result, details=details)
        cell_details.append({"id": expected["id"], "cells": details})
        totals.update(metrics)
        pages.append({"id": expected["id"], **metrics})
    args.output.write_text(json.dumps({"scope": "labelled financial regions only",
                                      "totals": dict(totals), "pages": pages}, indent=2) + "\n")
    if args.details:
        args.details.write_text(json.dumps(cell_details, indent=2) + "\n")
    print(json.dumps(dict(totals), sort_keys=True))


if __name__ == "__main__":
    main()
