#!/usr/bin/env python3
"""Measured corpus for cardinal page orientation (TAX-6995).

Builds image-only PDFs from public and synthetic pages only, each page turned
0/90/180/270 degrees, and scores ``pdf_service.detect_page_orientations``
against the known correction. No client document is ever read.

Pages:
  * every page of the blank IRS forms under ``--irs-dir`` (public);
  * synthetic pages generated here: letters, statement tables, sparse pages,
    narrow receipts and IRS form pages filled with invented values.

Variants of every upright page:
  * ``clean`` — 150-DPI grayscale render;
  * ``scan``  — 150 DPI, slight skew, blur, noise, tinted paper, JPEG q45;
  * ``faint`` — 110 DPI, washed-out print, heavy noise, JPEG q35.

Each variant is turned counter-clockwise by 0/90/180/270 degrees with
``np.rot90``; the clockwise correction that makes it upright again (the
``rotation`` field of /transform/orientation) is then 0/90/180/270.

Outcome per page:
  * correct — the expected correction (a 0-degree page returns nothing);
  * wrong   — a different correction, including any correction on an upright
    page (a false rotation is the costly error: it turns a good page);
  * missed  — a turned page for which nothing was returned.

Usage:
    python scripts/orientation_corpus.py build  --irs-dir ../taxscout-cpa/IRS_PDF
    python scripts/orientation_corpus.py score  [--label new]

Writes .corpus-work/orientation/ (gitignored), never /tmp. ``score`` imports
``app.services.pdf_service`` from the working directory, so the same command
run from another checkout scores that checkout's detector.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import fitz
import numpy as np

fitz.TOOLS.mupdf_display_errors(False)
WORK = Path(os.environ.get("ORIENTATION_WORK", Path.cwd() / ".corpus-work" / "orientation"))
VARIANTS = ("clean", "scan", "faint")
WORDS = (
    "account statement period balance interest dividend payment received total "
    "federal state withholding income tax return employer employee wages tips "
    "distribution gross proceeds basis reported please review the enclosed "
    "summary of your activity for the year ended december questions contact us "
    "customer service amount due date description deposit withdrawal transfer "
    "qualified ordinary capital gain loss adjustment code box copy recipient"
).split()
NAMES = ["Jordan A. Rivera", "Casey M. Holt", "Avery Lin", "Morgan T. Price"]
CITIES = ["Austin, TX 78701", "Denver, CO 80202", "Columbus, OH 43215"]


def _sentence(rng: random.Random, n: int) -> str:
    words = [rng.choice(WORDS) for _ in range(n)]
    return " ".join(words).capitalize() + "."


def _letter(rng: random.Random) -> fitz.Document:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 80), "Northwind Financial Services", fontsize=16, fontname="hebo")
    page.insert_text((72, 98), "1200 Commerce St, Suite 400", fontsize=9)
    page.insert_text((72, 150), rng.choice(NAMES), fontsize=11)
    page.insert_text((72, 164), rng.choice(CITIES), fontsize=11)
    y = 210
    for _ in range(rng.randint(3, 6)):
        text = " ".join(_sentence(rng, rng.randint(8, 16)) for _ in range(rng.randint(2, 4)))
        rect = fitz.Rect(72, y, 540, y + 110)
        spare = page.insert_textbox(rect, text, fontsize=rng.choice([9, 10, 11]))
        y += 110 - max(spare, 0) + 14
        if y > 700:
            break
    return doc


def _statement(rng: random.Random) -> fitz.Document:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((50, 60), "Account Activity", fontsize=14, fontname="hebo")
    page.insert_text((50, 78), f"Account ending {rng.randint(1000, 9999)}", fontsize=9)
    y = 110
    for col, x in (("Date", 50), ("Description", 120), ("Amount", 420), ("Balance", 500)):
        page.insert_text((x, y), col, fontsize=9, fontname="hebo")
    page.draw_line((50, y + 4), (562, y + 4), width=0.6)
    balance = rng.uniform(1000, 9000)
    for row in range(rng.randint(15, 38)):
        y += 15
        amount = rng.uniform(-900, 900)
        balance += amount
        page.insert_text((50, y), f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}", fontsize=8)
        page.insert_text((120, y), " ".join(rng.choice(WORDS) for _ in range(3)).upper(),
                         fontsize=8)
        page.insert_text((420, y), f"{amount:,.2f}", fontsize=8, fontname="cour")
        page.insert_text((500, y), f"{balance:,.2f}", fontsize=8, fontname="cour")
        if row % 5 == 4:
            page.draw_line((50, y + 4), (562, y + 4), width=0.3)
    return doc


def _sparse(rng: random.Random) -> fitz.Document:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), " ".join(rng.choice(WORDS) for _ in range(4)).title(),
                     fontsize=18, fontname="hebo")
    for i in range(rng.randint(2, 4)):
        page.insert_text((72, 140 + 20 * i), _sentence(rng, rng.randint(5, 9)), fontsize=11)
    return doc


def _receipt(rng: random.Random) -> fitz.Document:
    doc = fitz.open()
    page = doc.new_page(width=230, height=600)
    page.insert_text((20, 40), "RECEIPT", fontsize=14, fontname="cobo")
    y = 70
    total = 0.0
    for _ in range(rng.randint(6, 16)):
        price = rng.uniform(1, 80)
        total += price
        page.insert_text((20, y), rng.choice(WORDS).upper()[:14], fontsize=8, fontname="cour")
        page.insert_text((150, y), f"{price:8.2f}", fontsize=8, fontname="cour")
        y += 14
    page.insert_text((20, y + 10), "TOTAL", fontsize=9, fontname="cobo")
    page.insert_text((150, y + 10), f"{total:8.2f}", fontsize=9, fontname="cobo")
    return doc


def _filled_form(rng: random.Random, irs_pages: list[tuple[Path, int]]) -> fitz.Document:
    path, index = rng.choice(irs_pages)
    src = fitz.open(path)
    doc = fitz.open()
    doc.insert_pdf(src, from_page=index, to_page=index)
    page = doc[0]
    width, height = page.rect.width, page.rect.height
    for _ in range(rng.randint(8, 18)):
        x = rng.uniform(40, width - 160)
        y = rng.uniform(60, height * 0.6)
        value = rng.choice([
            f"{rng.uniform(10, 99999):,.2f}", rng.choice(NAMES), rng.choice(CITIES),
            f"{rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}",
        ])
        page.insert_text((x, y), value, fontsize=10, fontname="cour", rotate=0)
    return doc


def _upright_pages(irs_dir: Path) -> list[tuple[str, fitz.Document, int]]:
    pages: list[tuple[str, fitz.Document, int]] = []
    irs_pages: list[tuple[Path, int]] = []
    for path in sorted(irs_dir.rglob("*.pdf")):
        doc = fitz.open(path)
        for index in range(len(doc)):
            pages.append((f"irs:{path.relative_to(irs_dir)}:{index}", doc, index))
            if index < 2:
                irs_pages.append((path, index))
    rng = random.Random(6995)
    makers = {"letter": _letter, "statement": _statement, "sparse": _sparse,
              "receipt": _receipt}
    for kind, make in makers.items():
        for seed in range(6):
            pages.append((f"synthetic:{kind}:{seed}", make(rng), 0))
    for seed in range(8):
        pages.append((f"synthetic:filled-form:{seed}", _filled_form(rng, irs_pages), 0))
    return pages


def _render(doc: fitz.Document, index: int, dpi: int) -> np.ndarray:
    pix = doc[index].get_pixmap(dpi=dpi, colorspace=fitz.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()


def _degrade(gray: np.ndarray, rng: np.random.Generator, faint: bool) -> np.ndarray:
    image = gray.astype(np.float32)
    if faint:
        image = 255 - (255 - image) * 0.45  # washed-out toner
    angle = float(rng.uniform(-1.5, 1.5))
    h, w = image.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    image = cv2.warpAffine(image, matrix, (w, h), borderValue=255)
    image = cv2.GaussianBlur(image, (0, 0), 0.7 if faint else 0.6)
    image = image * float(rng.uniform(0.86, 0.95)) + float(rng.uniform(5, 15))
    image += rng.normal(0, 14 if faint else 8, image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 35 if faint else 45])
    assert ok
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)


def _image_page(out: fitz.Document, image: np.ndarray, dpi: int) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    h, w = image.shape[:2]
    page = out.new_page(width=w * 72 / dpi, height=h * 72 / dpi)
    page.insert_image(page.rect, stream=encoded.tobytes())


def build(irs_dir: Path) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    pages = _upright_pages(irs_dir)
    rng = np.random.default_rng(6995)
    manifest = {"pages": [], "variants": {}}
    outputs = {variant: fitz.open() for variant in VARIANTS}
    for name, doc, index in pages:
        renders = {
            "clean": (_render(doc, index, 150), 150),
            "scan": (_degrade(_render(doc, index, 150), rng, faint=False), 150),
            "faint": (_degrade(_render(doc, index, 110), rng, faint=True), 110),
        }
        for k in range(4):
            for variant, (image, dpi) in renders.items():
                _image_page(outputs[variant], np.ascontiguousarray(np.rot90(image, k)), dpi)
    expected = [{"source": name, "rotation": 90 * k} for name, _, _ in pages for k in range(4)]
    manifest["pages"] = expected
    for variant, out in outputs.items():
        path = WORK / f"{variant}.pdf"
        out.save(path, garbage=4, deflate=True)
        manifest["variants"][variant] = str(path.name)
    (WORK / "manifest.json").write_text(json.dumps(manifest, indent=1))
    kinds = Counter(name.split(":")[0] + ":" + name.split(":")[1] if name.startswith("synthetic")
                    else "irs" for name, _, _ in pages)
    print(json.dumps({"upright_pages": len(pages), "by_kind": kinds,
                      "pages_per_variant": len(expected)}, indent=1))


def score(label: str, chunk: int) -> None:
    sys.path.insert(0, str(Path.cwd()))
    from app.services import pdf_service

    manifest = json.loads((WORK / "manifest.json").read_text())
    expected = manifest["pages"]
    report: dict = {"label": label, "variants": {}}
    for variant, filename in manifest["variants"].items():
        source = fitz.open(WORK / filename)
        found: dict[int, dict] = {}
        started = time.monotonic()
        for start in range(0, len(source), chunk):
            part = fitz.open()
            part.insert_pdf(source, from_page=start, to_page=min(start + chunk, len(source)) - 1)
            result = pdf_service.detect_page_orientations(part.tobytes())
            for item in result["pages"]:
                found[start + item["page"]] = item
        elapsed = time.monotonic() - started
        counts = Counter()
        for index, truth in enumerate(expected):
            got = found.get(index)
            want = truth["rotation"]
            if got is None:
                outcome = "correct" if want == 0 else "missed"
            elif got["rotation"] == want:
                outcome = "correct"
            else:
                outcome = "wrong"
            counts[f"{outcome}@{want}"] += 1
            counts[outcome] += 1
            kind = "irs" if truth["source"].startswith("irs") else "synthetic"
            counts[f"{kind}:{outcome}"] += 1
        counts["pages"] = len(expected)
        report["variants"][variant] = {**dict(sorted(counts.items())),
                                       "seconds": round(elapsed, 1)}
        print(variant, json.dumps(report["variants"][variant]), flush=True)
    total = Counter()
    for counts in report["variants"].values():
        for key in ("pages", "correct", "wrong", "missed"):
            total[key] += counts.get(key, 0)
    report["total"] = dict(total)
    print("total", json.dumps(report["total"]))
    (WORK / f"report-{label}.json").write_text(json.dumps(report, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--irs-dir", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--label", default="current")
    s.add_argument("--chunk", type=int, default=16)
    args = parser.parse_args()
    if args.command == "build":
        build(args.irs_dir)
    else:
        score(args.label, args.chunk)


if __name__ == "__main__":
    main()
