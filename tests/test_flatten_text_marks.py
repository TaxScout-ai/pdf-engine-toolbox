"""TAX-5655: flatten burns every text mark the toolbox makes."""

import fitz

from app.services.pdf_service import flatten_annotations


def _blank_pdf() -> bytes:
    doc = fitz.open()
    doc.new_page(width=612, height=792)
    return doc.tobytes()


def _text(pdf: bytes) -> str:
    return fitz.open(stream=pdf, filetype="pdf")[0].get_text()


def test_stamp_with_null_stamp_type_uses_its_text():
    out = flatten_annotations(
        _blank_pdf(),
        [{"page_number": 1, "type": "stamp", "x": 10, "y": 10,
          "stamp_type": None, "text": "TIED"}],
    )
    assert "TIED" in _text(out)


def test_text_box_is_burned_in():
    out = flatten_annotations(
        _blank_pdf(),
        [{"page_number": 1, "type": "text_box", "x": 70, "y": 95,
          "width": 25, "height": 3, "text": "SMITH-000001"}],
    )
    assert "SMITH-000001" in _text(out)


def test_cross_reference_prints_its_label():
    out = flatten_annotations(
        _blank_pdf(),
        [{"page_number": 1, "type": "cross_reference", "x": 50, "y": 50,
          "text": "Ref #3"}],
    )
    assert "Ref #3" in _text(out)


def test_redaction_marks_are_not_drawn():
    out = flatten_annotations(
        _blank_pdf(),
        [{"page_number": 1, "type": "redaction", "x": 10, "y": 10,
          "width": 20, "height": 5, "text": "SECRET"}],
    )
    assert "SECRET" not in _text(out)
