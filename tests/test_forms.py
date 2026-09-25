"""PDF form fields: list and fill (TAX-5565)."""

import json
from unittest.mock import AsyncMock, patch

import fitz
import pytest

from app.services import pdf_service


def _form_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page()

    name = fitz.Widget()
    name.field_name = "taxpayer_name"
    name.field_type = fitz.PDF_WIDGET_TYPE_TEXT
    name.rect = fitz.Rect(72, 72, 300, 92)
    page.add_widget(name)

    agree = fitz.Widget()
    agree.field_name = "consent"
    agree.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX
    agree.rect = fitz.Rect(72, 110, 90, 128)
    page.add_widget(agree)

    status = fitz.Widget()
    status.field_name = "filing_status"
    status.field_type = fitz.PDF_WIDGET_TYPE_COMBOBOX
    status.choice_values = ["Single", "Married filing jointly"]
    status.rect = fitz.Rect(72, 140, 300, 160)
    page.add_widget(status)

    locked = fitz.Widget()
    locked.field_name = "preparer_ptin"
    locked.field_type = fitz.PDF_WIDGET_TYPE_TEXT
    locked.field_value = "P01234567"
    locked.field_flags = fitz.PDF_FIELD_IS_READ_ONLY
    locked.rect = fitz.Rect(72, 180, 300, 200)
    page.add_widget(locked)
    return doc.tobytes()


def test_lists_fields_with_type_value_options_and_lock():
    fields = {f["name"]: f for f in pdf_service.list_form_fields(_form_pdf())}
    assert set(fields) == {"taxpayer_name", "consent", "filing_status", "preparer_ptin"}
    assert fields["taxpayer_name"]["type"] == "text"
    assert fields["consent"]["type"] == "checkbox"
    assert fields["filing_status"]["options"] == ["Single", "Married filing jointly"]
    assert fields["preparer_ptin"]["read_only"] is True
    assert fields["taxpayer_name"]["page_index"] == 0


def test_fills_text_checkbox_and_choice_but_not_read_only_fields():
    out, changed = pdf_service.fill_form_fields(
        _form_pdf(),
        {
            "taxpayer_name": "Alex Martin",
            "consent": True,
            "filing_status": "Married filing jointly",
            "preparer_ptin": "HACKED",
            "no_such_field": "x",
        },
        flatten=False,
    )
    assert changed == 3
    fields = {f["name"]: f for f in pdf_service.list_form_fields(out)}
    assert fields["taxpayer_name"]["value"] == "Alex Martin"
    assert fields["consent"]["value"] not in ("Off", "", None, False)
    assert fields["filing_status"]["value"] == "Married filing jointly"
    assert fields["preparer_ptin"]["value"] == "P01234567"


def test_flatten_burns_values_into_the_page():
    out, _ = pdf_service.fill_form_fields(
        _form_pdf(), {"taxpayer_name": "Alex Martin"}, flatten=True
    )
    doc = fitz.open(stream=out, filetype="pdf")
    assert list(doc[0].widgets() or []) == []
    assert "Alex Martin" in doc[0].get_text()


def test_a_pdf_without_a_form_has_no_fields(sample_pdf_bytes):
    assert pdf_service.list_form_fields(sample_pdf_bytes) == []


def test_routes_list_and_fill(client, auth_headers):
    body = json.dumps({"source_url": "https://example.com/f.pdf"})
    with patch(
        "app.services.download_service.download_pdf",
        new_callable=AsyncMock,
        return_value=_form_pdf(),
    ):
        listed = client.post(
            "/forms/fields", content=body, headers=auth_headers("POST", "/forms/fields", body)
        )
        fill_body = json.dumps(
            {
                "source_url": "https://example.com/f.pdf",
                "values": {"taxpayer_name": "Alex"},
                "flatten": False,
            }
        )
        filled = client.post(
            "/forms/fill",
            content=fill_body,
            headers=auth_headers("POST", "/forms/fill", fill_body),
        )
    assert listed.status_code == 200
    assert len(listed.json()["data"]["fields"]) == 4
    assert filled.status_code == 200
    assert filled.content.startswith(b"%PDF")
    assert filled.headers["x-fields-changed"] == "1"


@pytest.mark.parametrize("path", ["/forms/fields", "/forms/fill"])
def test_routes_require_auth(client, path):
    assert client.post(path, json={"source_url": "https://e/x.pdf"}).status_code == 401
