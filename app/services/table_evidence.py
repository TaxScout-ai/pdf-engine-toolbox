"""Lossless logical cells and bounded geometry for financial table review.

Coordinates describe evidence, never CPA approval or a tax-field assignment.
"""
from __future__ import annotations

import math
from html.parser import HTMLParser

MAX_CELLS = 20_000
MAX_SPAN = 256


class TableStructureError(ValueError):
    pass


class TableParser(HTMLParser):
    """Keep physical cells in model order, with an occupancy map for spans."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.cells: list[dict] = []
        self.occupied: set[tuple[int, int]] = set()
        self.row = -1
        self.col = 0
        self.current: dict | None = None
        self.parts: list[str] = []
        self.row_open = False
        self.cell_tag: str | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            if self.current is not None or self.row_open:
                raise TableStructureError("Unclosed table row or cell")
            self.row += 1
            self.col = 0
            self.row_open = True
        elif tag in ("td", "th"):
            if not self.row_open or self.current is not None:
                raise TableStructureError("Invalid table cell order")
            attributes = dict(attrs)
            spans = []
            for key in ("rowspan", "colspan"):
                try:
                    span = int(attributes.get(key, "1"))
                except (TypeError, ValueError) as exc:
                    raise TableStructureError("Invalid cell span") from exc
                if not 1 <= span <= MAX_SPAN:
                    raise TableStructureError("Cell span exceeds limits")
                spans.append(span)
            rs, cs = spans
            while (self.row, self.col) in self.occupied:
                self.col += 1
            slots = {(r, c) for r in range(self.row, self.row + rs)
                     for c in range(self.col, self.col + cs)}
            if slots & self.occupied or len(self.occupied) + len(slots) > MAX_CELLS:
                raise TableStructureError("Overlapping or oversized table")
            self.occupied.update(slots)
            self.current = {"row_index": self.row, "column_index": self.col,
                            "row_span": rs, "column_span": cs, "is_header": tag == "th"}
            self.parts = []
            self.cell_tag = tag
            self.col += cs
        elif tag == "br" and self.current is not None:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.current is not None:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            if self.current is None or tag != self.cell_tag:
                raise TableStructureError("Mismatched table cell closing tag")
            raw = "".join(self.parts)
            self.cells.append({**self.current, "raw_text": raw,
                               "text": " ".join(raw.split())})
            self.current = None
            self.cell_tag = None
        elif tag == "tr":
            if not self.row_open or self.current is not None:
                raise TableStructureError("Invalid table row closing tag")
            self.row_open = False

    def result(self) -> tuple[list[list[str]], list[dict]]:
        if self.current is not None or self.row_open:
            raise TableStructureError("Unclosed table row or cell")
        if not self.occupied:
            return [], []
        height = max(r for r, _ in self.occupied) + 1
        if height > self.row + 1:
            raise TableStructureError("Row span exceeds declared rows")
        width = max(c for _, c in self.occupied) + 1
        if height * width > MAX_CELLS:
            raise TableStructureError("Table grid exceeds limits")
        rows = [[""] * width for _ in range(height)]
        for cell in self.cells:
            rows[cell["row_index"]][cell["column_index"]] = cell["text"]
        return rows, self.cells


def parse_table(table_html: str) -> tuple[list[list[str]], list[dict]]:
    parser = TableParser()
    parser.feed(table_html)
    parser.close()
    return parser.result()


def pdf_bbox(points, offset, scale, page_width: float, page_height: float) -> dict | None:
    """Translate a rectangle or polygon to PDF points; reject invalid geometry."""
    try:
        flat = [float(v) for v in points]
        if len(flat) not in (4, 8) or not all(math.isfinite(v) for v in flat):
            return None
        xs, ys = flat[::2], flat[1::2]
        x0, x1 = [(x + offset[0]) * scale[0] for x in (min(xs), max(xs))]
        y0, y1 = [(y + offset[1]) * scale[1] for y in (min(ys), max(ys))]
        if not (0 <= x0 < x1 <= page_width and 0 <= y0 < y1 <= page_height):
            return None
        return dict(zip(("x0", "y0", "x1", "y1"), map(lambda v: round(v, 4),
                                                     (x0, y0, x1, y1))))
    except (TypeError, ValueError, OverflowError):
        return None


def native_cells(table, rows, page_index: int, page_width: float, page_height: float):
    """Preserve native cell geometry; infer spans only from the observed grid."""
    physical = [bounds for row in table.rows for bounds in row.cells if bounds]
    xs = sorted({round(v, 3) for b in physical for v in (b[0], b[2])})
    ys = sorted({round(v, 3) for b in physical for v in (b[1], b[3])})
    cells = []
    seen = set()
    for r, row in enumerate(table.rows):
        for c, bounds in enumerate(row.cells):
            if not bounds or tuple(bounds) in seen:
                continue
            seen.add(tuple(bounds))
            bbox = pdf_bbox(bounds, (0, 0), (1, 1), page_width, page_height)
            raw = rows[r][c] if r < len(rows) and c < len(rows[r]) else None
            rs = sum(bounds[1] - .001 <= y < bounds[3] - .001 for y in ys)
            cs = sum(bounds[0] - .001 <= x < bounds[2] - .001 for x in xs)
            # Irregular geometry is not a reliable merged-cell assertion.
            valid = (bbox is not None and 1 <= rs <= len(rows) - r
                     and 1 <= cs <= len(row.cells) - c)
            cells.append({
                "page_index": page_index, "row_index": r, "column_index": c,
                "row_span": rs if valid else 1, "column_span": cs if valid else 1,
                "raw_text": raw or "", "text": " ".join((raw or "").split()),
                "is_header": False, "bbox": bbox if valid else None,
                "source": "native_pdf",
                "geometry_status": "available" if valid else "unavailable",
            })
    return cells
