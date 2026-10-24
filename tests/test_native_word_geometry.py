"""Native word rectangles remain tied to glyphs in displayed PDF coordinates."""
import json
from unittest.mock import AsyncMock, patch

import fitz
import pytest

from app.services.pdf_service import extract_text


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("cropped", [False, True])
def test_words_follow_rotated_cropped_glyphs(rotation, cropped):
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((120, 180), "WWW iii 123.45", fontsize=18)
    page.insert_text((120, 240), "Second copy 123.45", fontsize=10)
    if cropped:
        page.set_cropbox(fitz.Rect(40, 50, 560, 750))
    page.set_rotation(rotation)
    source = doc.tobytes()
    with fitz.open(stream=source, filetype="pdf") as reopened:
        page = reopened[0]
        # Search glyph rectangles independently of the extraction word loop.
        expected = [rect * page.rotation_matrix for rect in page.search_for("123.45")]
        visual_size = (page.rect.width, page.rect.height)
    result = extract_text(source)[0]
    assert (result["visual_width"], result["visual_height"]) == visual_size
    amounts = [w for w in result["words"] if w["text"] == "123.45"]
    assert len(amounts) == 2
    for word, rect in zip(amounts, expected, strict=True):
        assert list(word["bbox"].values()) == pytest.approx(list(rect))
    assert amounts[0]["bbox"] != amounts[1]["bbox"]
    assert result["blocks"]  # legacy response stays present


def test_proportional_font_width_is_not_character_count():
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), "WWW iii", fontsize=20)
    words = extract_text(doc.tobytes())[0]["words"]
    wide, narrow = words
    assert wide["bbox"]["x1"] - wide["bbox"]["x0"] > 3 * (
        narrow["bbox"]["x1"] - narrow["bbox"]["x0"]
    )


def test_positions_opt_out_remains_text_only(sample_pdf_bytes):
    result = extract_text(sample_pdf_bytes, pages=[1], include_positions=False)
    assert len(result) == 1 and result[0]["page_index"] == 1
    assert "words" not in result[0] and "blocks" not in result[0]


def test_api_preserves_word_geometry(client, auth_headers, sample_pdf_bytes):
    body = json.dumps({"source_url": "https://example.com/test.pdf", "pages": [1]})
    with patch("app.services.download_service.download_pdf", new_callable=AsyncMock,
               return_value=sample_pdf_bytes):
        response = client.post("/text/extract", content=body,
                               headers=auth_headers("POST", "/text/extract", body))
    assert response.status_code == 200
    page = response.json()["data"]["pages"][0]
    assert page["page_index"] == 1 and page["words"]
    assert page["visual_width"] > 0 and page["visual_height"] > 0
    assert page["words"][0]["bbox"]["x1"] > page["words"][0]["bbox"]["x0"]
