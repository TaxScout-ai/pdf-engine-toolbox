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


def one_cell():
    return {"rows": 1, "page_index": 2, "cells": [
        {"row": 0, "column": 0, "text": "-100.00", "column_label": "Amount",
         "region": [0, 0, 50, 20], "bbox": [10, 5, 40, 15]}]}


def reading(text="-100.00", bbox=None, page=2):
    return {"tables": [{"page_index": page, "cells": [
        {"raw_text": text, "column_index": 0, "bbox": bbox or [10, 5, 40, 15]}]}]}


def test_malformed_grouping_or_double_negative_cannot_earn_credit():
    for value in ["1,2", "12,34.00", "(-100)", "($-100)", "1 00", "--100"]:
        assert number(value) is None
    assert number("-$1,234.50") == number("($1,234.50)")
    assert number("$-1,234.50") == number("-1234.50")


def test_matching_value_on_wrong_page_is_missing_not_verified():
    metrics = evaluate(one_cell(), reading(page=1))
    assert metrics["wrong_page_predictions"] == 1
    assert metrics["missing_money_cells"] == 1
    assert metrics.get("exact_money_cells", 0) == 0
    result = reading()
    result["tables"][0]["cells"][0]["page_index"] = 1
    assert evaluate(one_cell(), result)["wrong_page_predictions"] == 1


def test_malformed_nonfinite_and_degenerate_boxes_are_counted_not_used():
    for bounds in [[0, 0, float("inf"), 10], [0, 0, float("nan"), 10],
                   [0, 0, 0, 10], [0, 0, 1], [False, 0, 10, 10], {"x0": 0}]:
        metrics = evaluate(one_cell(), reading(bbox=bounds))
        assert metrics["invalid_geometry_predictions"] == 1
        assert metrics["missing_money_cells"] == 1


def test_exact_value_with_oversized_highlight_is_separate_from_bbox_quality():
    details = []
    metrics = evaluate(one_cell(), reading(bbox=[0, 0, 60, 20]), details=details)
    assert metrics["exact_money_cells"] == 1
    assert metrics["exact_money_with_contained_bbox"] == 0
    assert metrics["located_bbox_outside_cell_region"] == 1
    assert details[0]["bbox_contained"] is False


def test_wrong_sign_wrong_value_and_merged_text_are_separate_outcomes():
    for text, outcome in [("100", "wrong_sign"), ("-101", "wrong_numeric_value"),
                          ("-100 -200", "non_numeric_or_merged_text")]:
        details = []
        metrics = evaluate(one_cell(), reading(text), details=details)
        assert metrics["mismatch_" + outcome + "_cells"] == 1
        assert metrics.get("wrong_sign_cells", 0) == (outcome == "wrong_sign")
        assert details[0]["outcome"] == outcome
        assert "text" not in details[0]
        assert "value" not in details[0]


def test_one_located_cell_does_not_make_a_complete_row():
    gold = one_cell()
    gold["cells"].append({**gold["cells"][0], "column": 1,
                          "region": [60, 0, 110, 20], "bbox": [70, 5, 100, 15]})
    metrics = evaluate(gold, reading())
    assert metrics["missing_numeric_rows"] == 0
    assert metrics["fully_located_numeric_rows"] == 0
    assert metrics["fully_exact_numeric_rows"] == 0
    assert metrics["missing_money_cells"] == 1


def test_explicit_money_annotation_overrides_legacy_column_label_heuristic():
    gold = one_cell()
    gold["cells"][0]["is_monetary"] = False
    metrics = evaluate(gold, reading())
    assert metrics["exact_numeric_cells"] == 1
    assert metrics["money_cells"] == 0


def test_ambiguous_duplicate_predictions_are_visible_and_do_not_double_count():
    result = reading()
    result["tables"][0]["cells"] *= 2
    metrics = evaluate(one_cell(), result)
    assert metrics["ambiguous_numeric_associations"] == 1
    assert metrics["exact_money_cells"] == 1


def test_scalar_and_string_coordinate_boxes_cannot_earn_credit():
    for bounds in ["0011", ["10", 5, 40, 15]]:
        assert evaluate(one_cell(), reading(bbox=bounds))["invalid_geometry_predictions"] == 1


def test_page_index_must_be_nonnegative_integer_not_boolean_or_float():
    import pytest
    for page in [False, True, 2.0, -1, "2"]:
        assert evaluate(one_cell(), reading(page=page))["invalid_page_predictions"] == 1
        gold = one_cell()
        gold["page_index"] = page
        with pytest.raises(ValueError, match="page index"):
            evaluate(gold, reading())


def test_labelled_coordinate_dict_is_normalized_and_strings_are_rejected():
    import pytest
    gold = one_cell()
    for name in ["region", "bbox"]:
        gold["cells"][0][name] = dict(zip(("x0", "y0", "x1", "y1"), gold["cells"][0][name]))
    assert evaluate(gold, reading())["exact_money_cells"] == 1
    gold["cells"][0]["region"]["x0"] = "0"
    with pytest.raises(ValueError, match="geometry"):
        evaluate(gold, reading())


def test_even_subpoint_overflow_is_not_strict_bbox_containment():
    metrics = evaluate(one_cell(), reading(bbox=[-0.5, 0, 50.5, 20]))
    assert metrics["exact_money_cells"] == 1
    assert metrics["exact_money_with_contained_bbox"] == 0
