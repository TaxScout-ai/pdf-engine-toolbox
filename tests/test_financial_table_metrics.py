"""The benchmark must not reward a matching amount in the wrong location."""
from scripts.evaluate_financial_tables import evaluate, number


def test_signs_are_preserved_and_non_numeric_cells_are_not_coerced():
    assert number("($1,234.56)") == number("-1234.56")
    assert number("0.00") == 0
    assert number("100 200") is None
    assert number("100\n200") is None
    assert number("not reported") is None


def test_geometric_match_exposes_swapped_amounts_and_missing_rows():
    gold = {"rows": 3, "cells": [
        {"row": r, "column": 0, "text": str(value), "column_label": "Proceeds",
         "region": [0, r * 20, 50, r * 20 + 20],
         "bbox": [10, r * 20 + 5, 40, r * 20 + 15]}
        for r, value in enumerate([100, -200, 300])
    ]}
    result = {"tables": [{"cells": [
        {"raw_text": "-200", "column_index": 0, "bbox": [10, 5, 40, 15]},
        {"raw_text": "100", "column_index": 0, "bbox": [10, 25, 40, 35]},
    ]}]}
    metrics = evaluate(gold, result)
    assert metrics["wrong_numeric_cells"] == 2
    assert metrics.get("exact_numeric_cells", 0) == 0
    assert metrics["missing_numeric_rows"] == 1
