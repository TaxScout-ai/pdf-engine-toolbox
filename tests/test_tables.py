"""Table extraction (POST /text/tables), TAX-6995.

Digital pages use PyMuPDF. Page images (scans, and digital pages where
PyMuPDF finds nothing) use ruled-table regions + RapidTable structure +
RapidOCR text on the OCR workers; PP-StructureV3 and Paddle are gone.
Unit tests use stand-in engines; one test runs the real pinned models when
they are provisioned (RAPIDOCR_MODEL_DIR).
"""

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import cv2
import fitz
import numpy as np
import pytest

from app.config import settings
from app.services import pdf_service, rapid_ocr, table_ocr
from app.services.rapid_ocr_models import MODEL_HASHES, TABLE_MODEL_HASHES, valid_model

ROWS = [
    ["Name", "Amount", "Date"],
    ["W-2 Income", "$85,000", "2024-01-31"],
    ["1099-INT", "$1,250", "2024-01-15"],
    ["1099-DIV", "$3,500", "2024-02-01"],
]


def _create_pdf_with_table():
    """A digital PDF with a ruled 4 x 3 table at (72, 72)-(372, 152)."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for r, row in enumerate(ROWS):
        for c, cell_text in enumerate(row):
            rect = fitz.Rect(72 + c * 100, 72 + r * 20, 172 + c * 100, 92 + r * 20)
            page.draw_rect(rect, color=(0, 0, 0), width=0.5)
            page.insert_textbox(rect, cell_text, fontsize=10, fontname="helv",
                                align=fitz.TEXT_ALIGN_CENTER)
    return doc.tobytes()


def _scan_of(pdf_bytes: bytes, dpi: int = 200) -> bytes:
    """The first page as an image-only PDF, as a scanner would deliver it."""
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    pix = src[0].get_pixmap(dpi=dpi)
    out = fitz.open()
    page = out.new_page(width=src[0].rect.width, height=src[0].rect.height)
    page.insert_image(page.rect, pixmap=pix)
    return out.tobytes()


def _blank_scan() -> bytes:
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    return doc.tobytes()


def _gray(pdf_bytes: bytes, dpi: int = 144) -> np.ndarray:
    page = fitz.open(stream=pdf_bytes, filetype="pdf")[0]
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def _post(client, auth_headers, pdf_bytes, payload):
    body = json.dumps({"source_url": "https://example.com/t.pdf", **payload})
    headers = auth_headers("POST", "/text/tables", body)
    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=pdf_bytes,
    ):
        return client.post("/text/tables", content=body, headers=headers)


class _FakeOcr:
    """RapidOCR stand-in: one reading per table cell of ROWS, in page pixels."""

    def __init__(self, scale: float):
        self.scale = scale
        self.calls = 0

    def __call__(self, image, **_):
        self.calls += 1
        boxes, texts = [], []
        for r, row in enumerate(ROWS):
            for c, text in enumerate(row):
                x0, y0 = (75 + c * 100) * self.scale, (75 + r * 20) * self.scale
                x1, y1 = (165 + c * 100) * self.scale, (89 + r * 20) * self.scale
                boxes.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                texts.append(text)
        return SimpleNamespace(boxes=np.array(boxes), txts=tuple(texts),
                               scores=tuple(0.97 for _ in texts))


class _FakeTable:
    """RapidTable stand-in: a grid of 3 columns from the OCR lines it is given."""

    def __init__(self):
        from rapid_table.table_matcher.main import TableMatch
        self.table_matcher = TableMatch()
        self.crops = []

    def table_structure(self, crops):
        self.crops.extend(crops)
        h, w = crops[0].shape[:2]
        structure = ["<table>"] + (["<tr>"] + ["<td></td>"] * 3 + ["</tr>"]) * len(ROWS) + ["</table>"]
        boxes = [[c*w/3, r*h/len(ROWS), (c+1)*w/3, (r+1)*h/len(ROWS)]
                 for r in range(len(ROWS)) for c in range(3)]
        return [(structure, .99)], [np.asarray(boxes)]



@pytest.fixture
def fake_engines(monkeypatch):
    ocr = _FakeOcr(scale=216 / 72)  # image pages render at the 216-DPI cap
    table = _FakeTable()
    monkeypatch.setattr(rapid_ocr, "_get_engine", lambda: ocr)
    monkeypatch.setattr(table_ocr, "_get_engine", lambda: table)
    return SimpleNamespace(ocr=ocr, table=table)


# ------------------------------------------------------------- digital pages


def test_extract_tables_digital(client, auth_headers, monkeypatch):
    """A digital ruled table is read by PyMuPDF; no model loads."""
    monkeypatch.setattr(table_ocr, "read_page", pytest.fail)
    response = _post(client, auth_headers, _create_pdf_with_table(), {"pages": [0]})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["total_count"] == 1
    assert data["data"]["tables"][0]["rows"][1][1] == "$85,000"


def test_extract_tables_pymupdf_strategy_never_reads_the_image(client, auth_headers, monkeypatch):
    monkeypatch.setattr(table_ocr, "read_page", pytest.fail)
    response = _post(client, auth_headers, _scan_of(_create_pdf_with_table()),
                     {"pages": [0], "strategy": "pymupdf"})
    assert response.status_code == 200
    assert response.json()["data"]["tables"] == []


# --------------------------------------------------------------- page images


@pytest.mark.parametrize("strategy", ["rapidtable", "ppstructure", "auto"])
def test_scanned_table_is_read_from_the_image(client, auth_headers, fake_engines, strategy):
    response = _post(client, auth_headers, _scan_of(_create_pdf_with_table()),
                     {"pages": [0], "strategy": strategy})
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total_count"] == 1
    table = data["tables"][0]
    assert (table["page_index"], table["table_index"]) == (0, 0)
    assert table["rows"] == ROWS
    assert (table["row_count"], table["col_count"]) == (4, 3)
    # The ruled region, in PDF points (drawn at 72,72 - 372,152).
    assert table["bbox"]["x0"] == pytest.approx(72, abs=3)
    assert table["bbox"]["y0"] == pytest.approx(72, abs=3)
    assert table["bbox"]["x1"] == pytest.approx(372, abs=3)
    assert table["bbox"]["y1"] == pytest.approx(152, abs=3)
    assert len(fake_engines.table.crops) == 1


def test_digital_page_without_a_pymupdf_table_falls_back_to_the_image(
    client, auth_headers, fake_engines, monkeypatch
):
    monkeypatch.setattr(pdf_service, "_extract_tables_pymupdf", lambda doc, page_idx: [])
    response = _post(client, auth_headers, _create_pdf_with_table(), {"pages": [0]})
    table = response.json()["data"]["tables"][0]
    assert "$85,000" in [cell for row in table["rows"] for cell in row]
    assert table["text_source"] == "native_pdf"
    assert fake_engines.ocr.calls == 0


def test_borderless_table_is_unsupported_not_claimed_empty(client, auth_headers, fake_engines):
    """No ruled grid: no table is claimed and the page is not even OCRed."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i, row in enumerate(ROWS):  # the same values, without rules
        page.insert_text((72, 100 + 20 * i), "      ".join(row), fontsize=10)
    page.draw_rect(fitz.Rect(60, 60, 400, 200), color=(0, 0, 0), width=0.8)  # a frame
    response = _post(client, auth_headers, _scan_of(doc.tobytes()), {"pages": [0]})
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["tables"] == []
    assert data["status"] == "partial"
    assert data["pages"][0]["status"] == "unsupported"
    assert data["pages"][0]["reason"] == "no_ruled_table_region"
    assert fake_engines.ocr.calls == 0


def test_tables_on_several_pages_keep_page_order(client, auth_headers, fake_engines):
    scan = fitz.open(stream=_scan_of(_create_pdf_with_table()), filetype="pdf")
    blank = fitz.open(stream=_blank_scan(), filetype="pdf")
    doc = fitz.open()
    for source in (scan, blank, scan):
        doc.insert_pdf(source)
    response = _post(client, auth_headers, doc.tobytes(), {})
    tables = response.json()["data"]["tables"]
    assert [(t["page_index"], t["table_index"]) for t in tables] == [(0, 0), (2, 0)]


def test_other_languages_are_refused_on_page_images(client, auth_headers, fake_engines):
    response = _post(client, auth_headers, _scan_of(_create_pdf_with_table()),
                     {"pages": [0], "language": "fr"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNSUPPORTED_LANGUAGE"


def test_recognition_failure_is_explicit_and_broken_configuration_raises(fake_engines, monkeypatch):
    scan = _scan_of(_create_pdf_with_table())
    monkeypatch.setattr(table_ocr, "read_tables", lambda *a, **k: 1 / 0)
    result = table_ocr.read_page_result(scan, 0, 300)
    assert result["status"] == "failed"
    assert result["reason"] == "table_recognition_failed"
    with pytest.raises(RuntimeError, match="table_recognition_failed"):
        table_ocr.read_page(scan, 0, 300)

    def broken():
        raise RuntimeError("RapidOCR model missing or hash mismatch: slanet-plus.onnx")

    monkeypatch.setattr(table_ocr, "_get_engine", broken)
    with pytest.raises(rapid_ocr.RapidConfigurationError):
        table_ocr.read_page(scan, 0, 300)


def test_unread_second_region_does_not_make_first_table_disappear_or_claim_complete(
    fake_engines, monkeypatch
):
    monkeypatch.setattr(table_ocr, "table_regions",
                        lambda gray: [(200, 200, 1130, 470), (1200, 800, 1600, 1200)])
    result = table_ocr.read_page_result(_scan_of(_create_pdf_with_table()), 0, 300)
    assert len(result["tables"]) == 1
    assert result["status"] == "unsupported"
    assert result["reason"] == "table_region_unresolved"


# ----------------------------------------------------------- region detection


def test_ruled_table_region_is_found():
    regions = table_ocr.table_regions(_gray(_scan_of(_create_pdf_with_table())))
    assert len(regions) == 1
    x0, y0, x1, y1 = (v * 72 / 144 for v in regions[0])
    assert (x0, y0, x1, y1) == pytest.approx((72, 72, 372, 152), abs=3)


@pytest.mark.parametrize("draw", [
    lambda page: None,
    lambda page: page.draw_line((72, 300), (540, 300), width=1),
    lambda page: page.draw_rect(fitz.Rect(72, 72, 300, 200), width=1),
])
def test_rules_that_do_not_form_a_grid_are_not_a_table(draw):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    draw(page)
    assert table_ocr.table_regions(_gray(doc.tobytes())) == []


def test_a_slightly_noisy_scan_still_finds_the_grid():
    gray = _gray(_scan_of(_create_pdf_with_table()))
    rng = np.random.default_rng(0)
    noisy = np.clip(gray.astype(float) * 0.92 + 10 + rng.normal(0, 8, gray.shape), 0, 255)
    noisy = cv2.GaussianBlur(noisy.astype(np.uint8), (0, 0), 0.6)
    assert len(table_ocr.table_regions(noisy)) == 1


# --------------------------------------------------------------- HTML parsing


def test_html_rows_expand_colspan_and_unescape():
    html = (
        "<table>"
        "<tr><td colspan=\"2\">Header Span</td><td>Col3</td></tr>"
        "<tr><td>A &amp; B</td><td> B </td><td>C</td></tr>"
        "</table>"
    )
    assert table_ocr.html_rows(html) == [["Header Span", "", "Col3"], ["A & B", "B", "C"]]


def test_html_rows_basic():
    html = (
        "<table><tbody>"
        "<tr><th>Name</th><th>Amount</th></tr>"
        "<tr><td>W-2</td><td>$85,000</td></tr>"
        "</tbody></table>"
    )
    assert table_ocr.html_rows(html) == [["Name", "Amount"], ["W-2", "$85,000"]]


# --------------------------------------------------------------- real models


def _models_present() -> bool:
    root = Path(settings.rapidocr_model_dir)
    hashes = {**MODEL_HASHES, **TABLE_MODEL_HASHES}
    return all(valid_model(root / name, digest) for name, digest in hashes.items())


@pytest.mark.skipif(not _models_present(), reason="RapidOCR/RapidTable models not provisioned")
def test_real_models_read_a_scanned_ruled_table():
    tables = pdf_service.extract_tables(_scan_of(_create_pdf_with_table()), [0])
    assert len(tables) == 1
    rows = tables[0]["rows"]
    assert (tables[0]["row_count"], tables[0]["col_count"]) == (4, 3)
    assert rows[0] == ["Name", "Amount", "Date"]
    assert rows[1] == ["W-2 Income", "$85,000", "2024-01-31"]
    assert rows[3][1] == "$3,500"


def test_native_cells_carry_page_bounds_raw_text_and_hash(client, auth_headers):
    import hashlib
    pdf = _create_pdf_with_table()
    data = _post(client, auth_headers, pdf, {"pages": [0]}).json()["data"]
    assert data["status"] == "complete"
    assert data["source_sha256"] == hashlib.sha256(pdf).hexdigest()
    table = data["tables"][0]
    assert len(table["cells"]) == 12
    amount = next(c for c in table["cells"] if c["row_index"] == 1 and c["column_index"] == 1)
    assert amount["raw_text"] == "$85,000"
    assert amount["page_index"] == 0 and amount["source"] == "native_pdf"
    assert amount["bbox"]["x0"] < amount["bbox"]["x1"]
    assert amount["geometry_status"] == "available"


def test_failed_page_does_not_erase_other_page_success(client, auth_headers, fake_engines, monkeypatch):
    original = table_ocr.read_page_result
    def read(source, index, dpi, **kwargs):
        if index == 1:
            return {"page_index": index, "status": "failed", "tables": [],
                    "source": "rapidtable", "reason": "table_recognition_failed"}
        return original(source, index, dpi)
    monkeypatch.setattr(table_ocr, "read_page_result", read)
    doc = fitz.open(stream=_scan_of(_create_pdf_with_table()), filetype="pdf")
    doc.insert_pdf(fitz.open(stream=_scan_of(_create_pdf_with_table()), filetype="pdf"))
    data = _post(client, auth_headers, doc.tobytes(), {}).json()["data"]
    assert data["status"] == "partial"
    assert [p["status"] for p in data["pages"]] == ["complete", "failed"]
    assert len(data["tables"]) == 1


def test_blank_scanned_page_is_explicitly_empty(client, auth_headers):
    data = _post(client, auth_headers, _blank_scan(), {}).json()["data"]
    assert data["status"] == "complete"
    assert data["tables"] == []
    assert data["pages"][0]["status"] == "empty"


def test_missing_predicted_cell_never_shifts_text_into_another_column(monkeypatch):
    from unittest.mock import Mock
    engine = SimpleNamespace(
        table_structure=Mock(return_value=([(["<table><tr><td></td><td></td><td></td></tr></table>"], .99)],
            [np.asarray([[0, 0, 30, 20], [60, 0, 90, 20]])])),
        table_matcher=Mock())
    monkeypatch.setattr(table_ocr, "_get_engine", lambda: engine)
    monkeypatch.setattr(table_ocr, "table_regions", lambda gray: [(0, 0, 100, 40)])
    ocr = lambda image: SimpleNamespace(boxes=np.asarray([[[60, 0], [90, 0], [90, 20], [60, 20]]]), txts=["100.00"], scores=[.99])
    result = table_ocr.read_tables(np.full((40, 100, 3), 255, dtype=np.uint8), 100, 40, 0, ocr)
    assert result == [] and result.unresolved_regions == 1
    engine.table_matcher.assert_not_called()


def test_table_deadline_is_rechecked_after_planning(monkeypatch):
    from unittest.mock import Mock
    from app.services import ocr_queue
    clock = iter([10.0, 12.0])
    monkeypatch.setattr(pdf_service, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    run = Mock(return_value=[{"page_index": 0, "tables": [], "status": "empty",
                              "reason": "blank_page", "source": "native_pdf"}])
    monkeypatch.setattr(ocr_queue, "get_queue", lambda: SimpleNamespace(run_pages=run))
    pdf_service.extract_tables_result(_blank_scan(), deadline=15)
    assert run.call_args.kwargs["timeout"] == 3


def test_expired_table_deadline_does_not_enqueue_work(monkeypatch):
    from unittest.mock import Mock
    from app.services import ocr_queue
    queue = Mock()
    monkeypatch.setattr(ocr_queue, "get_queue", queue)
    with pytest.raises(pdf_service.PdfEngineError, match="preparation timed out"):
        pdf_service.extract_tables_result(_blank_scan(), deadline=time.monotonic() - 1)
    queue.assert_not_called()


def test_table_preparation_bounds_global_pdf_lock_wait(monkeypatch):
    from unittest.mock import Mock
    lock = Mock()
    lock.acquire.return_value = False
    monkeypatch.setattr(rapid_ocr, "FITZ_LOCK", lock)
    with pytest.raises(pdf_service.PdfEngineError, match="preparation timed out"):
        pdf_service.extract_tables_result(_blank_scan(), timeout=1)
    assert 0 < lock.acquire.call_args.kwargs["timeout"] <= 1
    lock.release.assert_not_called()


def test_table_http_deadline_reports_failure(client, auth_headers, monkeypatch):
    from app.routes import text
    monkeypatch.setattr(text, "run_in_threadpool", AsyncMock(side_effect=TimeoutError))
    response = _post(client, auth_headers, _blank_scan(), {})
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "TABLE_REQUEST_TIMEOUT"


def test_scanned_sources_use_assigned_ocr_boxes_not_collapsed_structure_boxes(monkeypatch):
    """Repeated amounts keep their own token boxes, even with thin SLANet cells."""
    from rapid_table.table_matcher.main import TableMatch
    region = (20, 30, 220, 150)
    structure = ['<table>', '<tr>', '<td colspan="2">', '</td>', '</tr>',
                 '<tr>', '<td></td>', '<td></td>', '</tr>',
                 '<tr>', '<td></td>', '<td></td>', '</tr>', '</table>']
    # Local structure coordinates; deliberately much thinner than the text.
    predicted = np.array([[0, 0, 200, 20], [0, 45, 90, 46], [90, 45, 200, 46],
                          [0, 85, 90, 86], [90, 85, 200, 86]], dtype=float)
    engine = SimpleNamespace(table_structure=lambda images: ([(structure, .99)], [predicted]),
                             table_matcher=TableMatch())
    monkeypatch.setattr(table_ocr, '_get_engine', lambda: engine)
    monkeypatch.setattr(table_ocr, 'table_regions', lambda gray: [region])
    boxes = np.array([[[30, 35], [170, 35], [170, 45], [30, 45]],
                      [[30, 70], [90, 70], [90, 90], [30, 90]],
                      [[130, 70], [180, 70], [180, 90], [130, 90]],
                      [[30, 110], [90, 110], [90, 130], [30, 130]],
                      [[130, 110], [180, 110], [180, 130], [130, 130]]])
    read = SimpleNamespace(boxes=boxes, txts=['Header', 'First', '100.00', 'Second', '100.00'], scores=[.99]*5)
    tables = table_ocr.read_tables(np.full((180, 240, 3), 255, np.uint8), 120, 90, 2, lambda image: read)
    assert len(tables) == 1
    cells = tables[0]['cells']
    assert cells[0]['column_span'] == 2
    amounts = [c for c in cells if c['text'] == '100.00']
    assert len(amounts) == 2
    for cell, top, bottom in zip(amounts, [35, 55], [45, 65]):
        assert cell['page_index'] == 2
        assert cell['bbox']['x0'] <= 65 and cell['bbox']['x1'] >= 90
        assert cell['bbox']['y0'] <= top and cell['bbox']['y1'] >= bottom
        assert cell['bbox']['y1'] - cell['bbox']['y0'] >= 10
    assert amounts[0]['bbox']['y1'] < amounts[1]['bbox']['y0']


@pytest.mark.parametrize('left,available', [(10, True), (-10, False)])
def test_source_geometry_preserves_crop_edges_and_rejects_invalid_originals(monkeypatch, left, available):
    from rapid_table.table_matcher.main import TableMatch
    structure = ['<table>', '<tr>', '<td></td>', '</tr>', '</table>']
    predicted = np.array([[0, 20, 200, 100]], dtype=float)
    engine = SimpleNamespace(table_structure=lambda images: ([(structure, .99)], [predicted]),
                             table_matcher=TableMatch())
    monkeypatch.setattr(table_ocr, '_get_engine', lambda: engine)
    monkeypatch.setattr(table_ocr, 'table_regions', lambda gray: [(20, 30, 220, 150)])
    # The first token is removed by the matcher (above the first cell). The
    # second crosses the crop edge; a third token shares its multiline cell.
    boxes = np.array([[[30, 32], [70, 32], [70, 40], [30, 40]],
                      [[left, 70], [70, 70], [70, 90], [left, 90]],
                      [[30, 100], [90, 100], [90, 120], [30, 120]]])
    read = SimpleNamespace(boxes=boxes, txts=['ignored', '100.00', 'second line'], scores=[.99]*3)
    tables = table_ocr.read_tables(np.full((180, 240, 3), 255, np.uint8), 120, 90, 2, lambda image: read)
    cell = tables[0]['cells'][0]
    assert cell['text'] == '100.00 second line'
    if available:
        assert cell['bbox'] == {'x0': 5, 'y0': 35, 'x1': 45, 'y1': 60}
        assert cell['geometry_status'] == 'available'
    else:
        assert cell['bbox'] is None
        assert cell['geometry_status'] == 'unavailable'
        assert 'cell_geometry_unavailable' in tables[0]['warnings']
