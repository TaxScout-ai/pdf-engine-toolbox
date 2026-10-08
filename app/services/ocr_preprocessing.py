"""Conservative scan preparation; all geometry maps back to the original page.

No tax-field values or document-specific coordinates influence these heuristics.
Ambiguous layouts stay unchanged. This is preparation, not field verification.
"""

import cv2
import numpy as np


def text_layer_quality(page):
    text = page.get_text().strip()
    tokens = text.split()
    reasons = []
    if len(text) <= 50 or not tokens or sum(map(len, tokens)) / len(tokens) < 3:
        reasons.append("insufficient_text")
    if (
        text
        and sum(c == "\ufffd" or (not c.isprintable() and not c.isspace()) for c in text)
        / len(text)
        > 0.01
    ):
        reasons.append("invalid_characters")
    if any(t.get("type") == 3 for t in page.get_texttrace()):
        reasons.append("invisible_text_layer")
    area = max(page.rect.get_area(), 1)
    # A substantial raster may contain fields even when visible text is present.
    if any(
        (i["bbox"][2] - i["bbox"][0]) * (i["bbox"][3] - i["bbox"][1]) / area > 0.08
        for i in page.get_image_info()
    ):
        reasons.append("substantial_raster_content")
    return {"accepted": not reasons, "reasons": reasons, "characters": len(text)}


def prepare_scan(image):
    """Return prepared image, inverse 3x3 transform and reviewable decisions."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray < 180).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    # Drop isolated specks, retain small text and large graphic components alike.
    threshold = max(12, int(width * height / 150_000))
    keep = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= threshold]
    inverse = np.eye(3)
    meta = {
        "crop_pixels": [0, 0, width, height],
        "rotation_ccw": 0,
        "method": "foreground-components-v1",
        "orientation_ambiguous": False,
    }
    if not keep:
        return image, inverse, meta
    x0 = min(stats[i, 0] for i in keep)
    y0 = min(stats[i, 1] for i in keep)
    x1 = max(stats[i, 0] + stats[i, 2] for i in keep)
    y1 = max(stats[i, 1] + stats[i, 3] for i in keep)
    # Crop only large blank margins. Never retain only the largest contour.
    if (x1 - x0) * (y1 - y0) < width * height * 0.55:
        pad = max(24, int(max(x1 - x0, y1 - y0) * 0.06))
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(width, x1 + pad), min(height, y1 + pad)
        # Dark components only propose a crop. Any non-white pixel outside
        # it vetoes removal: faint text must never become "blank margin".
        # Noisy scans may remain uncropped; preserving evidence takes priority.
        nonwhite = gray < 255
        outside = (
            nonwhite[:y0].any()
            or nonwhite[y1:].any()
            or nonwhite[y0:y1, :x0].any()
            or nonwhite[y0:y1, x1:].any()
        )
        if outside:
            meta["crop_skipped_reason"] = "nonwhite_margin_content"
        else:
            image = image[y0:y1, x0:x1].copy()
            inverse[:2, 2] = [x0, y0]
            meta["crop_pixels"] = [int(x0), int(y0), int(x1), int(y1)]
    # Vote using small connected glyphs; border lines and portraits don't vote.
    glyphs = [
        stats[i]
        for i in keep
        if 8 <= stats[i, 2] <= width * 0.04
        and 8 <= stats[i, 3] <= height * 0.04
        and stats[i, 4] <= width * height * 0.001
    ]
    if len(glyphs) >= 15:
        ratios = np.array([s[3] / s[2] for s in glyphs])
        horizontal = float(np.median(ratios))
        if horizontal < 0.8:
            old_width = image.shape[1]
            image = np.ascontiguousarray(np.rot90(image))
            # Prepared (u,v) -> original crop (old_width-1-v,u).
            inverse = inverse @ np.array([[0, -1, old_width - 1], [1, 0, 0], [0, 0, 1]])
            meta["rotation_ccw"] = 90
        meta["orientation_ambiguous"] = 0.8 <= horizontal <= 1.1
        meta["glyph_aspect_median"] = round(horizontal, 4)
    return image, inverse, meta


def map_polygon(polygon, inverse):
    points = np.array(polygon, dtype=float, copy=True)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("RapidOCR returned invalid geometry")
    return (np.c_[points, np.ones(4)] @ inverse.T)[:, :2]
