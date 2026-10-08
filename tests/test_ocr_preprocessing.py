import fitz
import numpy as np
import pytest

from app.services.ocr_preprocessing import map_polygon, prepare_scan, text_layer_quality


def test_crop_preserves_original_coordinates():
    image = np.full((900, 900, 3), 255, dtype=np.uint8)
    image[300:450, 200:600] = 0
    cropped, inverse, meta = prepare_scan(image)
    assert cropped.shape[0] < 900 and cropped.shape[1] < 900
    x, y, _, _ = meta["crop_pixels"]
    polygon = np.array([[0, 0], [20, 0], [20, 10], [0, 10]])
    assert map_polygon(polygon, inverse)[0] == pytest.approx([x, y])


def test_sideways_glyphs_map_back_to_original():
    image = np.full((600, 600, 3), 255, dtype=np.uint8)
    for y in range(100, 500, 40):
        for x in range(100, 400, 40):
            image[y : y + 10, x : x + 16] = 0
    prepared, inverse, meta = prepare_scan(image)
    assert meta["rotation_ccw"] == 90
    p = np.array([[5, 6], [15, 6], [15, 20], [5, 20]])
    mapped = map_polygon(p, inverse)
    forward = np.linalg.inv(inverse)
    assert map_polygon(mapped, forward) == pytest.approx(p)
    assert prepared.size > 0


def test_invisible_text_is_not_trusted():
    with fitz.open() as doc:
        p = doc.new_page()
        p.insert_text(
            (50, 50),
            "Old OCR layer that contains enough selectable words to pass the previous heuristic",
            render_mode=3,
        )
        q = text_layer_quality(p)
        assert not q["accepted"]
        assert "invisible_text_layer" in q["reasons"]


def test_visible_text_on_scan_is_not_trusted():
    with fitz.open() as doc:
        p = doc.new_page()
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100), False)
        pix.clear_with(255)
        p.insert_image(p.rect, pixmap=pix)
        p.insert_text(
            (50, 50),
            "Selectable words on a scanned page must not cause "
            "important scanned fields to be skipped",
        )
        assert "substantial_raster_content" in text_layer_quality(p)["reasons"]
