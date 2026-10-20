"""Regressions for wrong-column amounts and non-fabricated source geometry."""
import pytest

from app.services.table_evidence import TableStructureError, parse_table, pdf_bbox


def test_rowspan_preserves_amount_column_and_original_text():
    rows, cells = parse_table(
        '<table><tr><td rowspan="2">Account A</td><td>  (100.00)\n</td></tr>'
        '<tr><td>-200.00</td></tr></table>'
    )
    assert rows == [["Account A", "(100.00)"], ["", "-200.00"]]
    assert cells[2]["column_index"] == 1
    assert cells[0]["row_span"] == 2
    assert cells[1]["raw_text"] == "  (100.00)\n"


def test_combined_row_and_column_spans_keep_physical_cell_order():
    rows, cells = parse_table(
        '<tr><th colspan="2" rowspan="2">Group</th><th>Amount</th></tr>'
        '<tr><td>0.00</td></tr><tr><td>A</td><td>B</td><td>10</td></tr>'
    )
    assert rows == [["Group", "", "Amount"], ["", "", "0.00"], ["A", "B", "10"]]
    assert [(c["row_index"], c["column_index"]) for c in cells] == [
        (0, 0), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]


@pytest.mark.parametrize("markup", [
    '<tr><td rowspan="0">x</td></tr>',
    '<tr><td colspan="99999999">x</td></tr>',
    '<tr><td rowspan="bad">x</td></tr>',
    '<td>missing row</td>',
    '<tr><td>unclosed',
    '<tr><td>A</td></tr><td>100</td>',
    '<tr><td rowspan="3">100</td></tr>',
    '<tr><td>100</th></tr>',
    '<tr><td>100</td><tr><td>200</td></tr>',
    '<tr><td>100</td>',
    '<tr><td>A</td><td rowspan="2">B</td></tr><tr><td colspan="2">overlap</td></tr>',
])
def test_invalid_structure_does_not_silently_invent_columns(markup):
    with pytest.raises(TableStructureError):
        parse_table(markup)


def test_crop_polygon_maps_to_original_pdf_points():
    assert pdf_bbox([0, 0, 40, 0, 40, 20, 0, 20], (100, 200), (.5, .5), 612, 792) == {
        "x0": 50, "y0": 100, "x1": 70, "y1": 110}


@pytest.mark.parametrize("bounds", [[0, 0, float("nan"), 4], [-1, 0, 4, 4],
                                    [0, 0, 9999, 4], [0, 0, 0, 0], [1, 2, 3]])
def test_invalid_cell_geometry_is_unavailable(bounds):
    assert pdf_bbox(bounds, (0, 0), (1, 1), 612, 792) is None
