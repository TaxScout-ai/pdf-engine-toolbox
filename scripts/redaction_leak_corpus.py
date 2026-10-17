#!/usr/bin/env python3
"""Held-out leak corpus for identifier redaction on scans (TAX-6876).

Fills real IRS 2025 form pages (tests/fixtures/irs2025, 200-DPI images) with
synthetic SSNs and EINs in many renderings, rasterizes each page at 200 DPI
into an image-only PDF and runs `identifier_redaction.redact_identifiers` on
it with the real RapidOCR models — once plainly and once through an engine
that garbles one or two characters of every digit-bearing reading into a
confusable letter (1->I, 0->O, 5->S, 8->B, 2->Z, 6->G, 9->q).

Outcomes per page:
  * refused  — redaction raised; the page goes to manual review (safe).
  * redacted — redaction returned a page. It is a LEAK when any digit of an
    identifier other than its kept last four is still legible: judged by the
    ground-truth character boxes (not fully blacked out in the output) and by
    re-reading the output with the plain engine, normalizing confusions, and
    finding the identifier's leading five digits.

A separate held-out set of filled pages without identifiers measures the
false-refusal rate.

This corpus is for evaluation only: thresholds in identifier_redaction.py
must never be tuned against it.

Usage (needs RAPIDOCR_MODEL_DIR with the pinned models):
    python scripts/redaction_leak_corpus.py --report /tmp/leak-report.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import fitz  # noqa: E402
import numpy as np  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "irs2025"
DPI = 200

# Where values go on each fixture page (points; baseline origin of the value).
FORMS = {
    "fw2": {
        "ssn": (167, 57), "ein": (52, 81),
        "fields": [((52, 105), "Northwind Traders LLC"), ((52, 117), "1200 Commerce St Ste 400"),
                   ((52, 129), "Austin, TX 78701-1234"), ((52, 201), "Jordan A. Rivera"),
                   ((52, 257), "Round Rock, TX 78664-2207"), ((350, 81), "75,250.00"),
                   ((473, 81), "9,850.00"), ((350, 105), "75,250.00"), ((52, 177), "A-0417"),
                   ((211, 309), "75,250.00")],
    },
    **{
        name: {
            "ssn": (179, 153 + dy), "ein": (56, 153 + dy),
            "fields": [((56, 64), "Northwind Traders LLC"), ((56, 76), "1200 Commerce St Ste 400"),
                       ((56, 88), "Austin, TX 78701-1234  (512) 555-0100"),
                       ((56, 201 + dy2), "Jordan A. Rivera"),
                       ((56, 237 + dy2), "47 Lakeview Dr Apt 2B"),
                       ((56, 261 + dy2), "Round Rock, TX 78664-2207"),
                       (amount, "4,182.55"), ((56, 345 + dy3), "A-0417")],
        }
        for name, dy, dy2, dy3, amount in [
            ("f1099int", 0, 0, 0, (527, 82)),
            ("f1099div", 0, 0, 0, (313, 57)),
            ("f1099r", 0, 0, 0, (309, 57)),
            ("f1099nec", -24, -48, -108, (352, 82)),
        ]
    },
    "f1098": {
        "ssn": (210, 153), "ein": (101, 153),
        "fields": [((56, 64), "Northwind Mortgage Co"), ((56, 76), "1200 Commerce St Ste 400"),
                   ((56, 88), "Austin, TX 78701-1234  (512) 555-0100"),
                   ((56, 201), "Jordan A. Rivera"), ((56, 237), "47 Lakeview Dr Apt 2B"),
                   ((56, 261), "Round Rock, TX 78664-2207"), ((307, 129), "11,204.88"),
                   ((56, 345), "A-0417")],
    },
}

_CONFUSE = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B", "9": "q"}


@dataclass
class Case:
    name: str
    form: str
    pdf: bytes
    identifiers: list[str]  # digits only
    prefix_boxes: list[fitz.Rect] = field(default_factory=list)  # points


def _identifier(rng: random.Random, digits: int = 9) -> str:
    # First digit 1-8 keeps clear of 9xx ITINs; no repeated runs.
    return str(rng.randint(1, 8)) + "".join(str(rng.randint(0, 9)) for _ in range(digits - 1))


def _ssn(d):
    return f"{d[:3]}-{d[3:5]}-{d[5:]}"


def _ein(d):
    return f"{d[:2]}-{d[2:]}"


class _Canvas:
    """Draws on the form page and, identically, on an overlay that records boxes."""

    def __init__(self, form: str):
        self.doc = fitz.open()
        self.page = self.doc.new_page(width=612, height=792)
        self.page.insert_image(self.page.rect, filename=str(FIXTURES / f"{form}.png"))
        # One blank overlay per identifier records where its characters are.
        self.overlay_doc = fitz.open()
        self.overlays: dict[int, int] = {}  # identifier index -> overlay page number

    def text(self, origin, text, size=10, color=(0, 0, 0), rotate=0, secret=None):
        self.page.insert_text(origin, text, fontsize=size, color=color, rotate=rotate,
                              fontname="helv")
        if secret is not None:
            if secret not in self.overlays:
                self.overlay_doc.new_page(width=612, height=792)
                self.overlays[secret] = len(self.overlay_doc) - 1
            self.overlay_doc[self.overlays[secret]].insert_text(
                origin, text, fontsize=size, rotate=rotate, fontname="helv")

    def rect(self, rect, fill=None, color=(0, 0, 0), width=0.6):
        self.page.draw_rect(rect, color=color, fill=fill, width=width)

    def secret_char_boxes(self, index: int) -> list[tuple[str, fitz.Rect]]:
        chars = []
        page = self.overlay_doc[self.overlays[index]]
        for block in page.get_text("rawdict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    for ch in span["chars"]:
                        if ch["c"].isdigit():
                            chars.append((ch["c"], fitz.Rect(ch["bbox"])))
        return chars

    def rasterize(self, blur_box: fitz.Rect | None = None, blur: float = 0) -> bytes:
        pix = self.page.get_pixmap(dpi=DPI, colorspace=fitz.csGRAY)
        image = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
        if blur and blur_box is not None:
            zoom = DPI / 72
            x0, y0 = int(max(0, blur_box.x0 * zoom - 20)), int(max(0, blur_box.y0 * zoom - 20))
            x1 = int(min(image.shape[1], blur_box.x1 * zoom + 20))
            y1 = int(min(image.shape[0], blur_box.y1 * zoom + 20))
            image[y0:y1, x0:x1] = cv2.GaussianBlur(image[y0:y1, x0:x1], (0, 0), blur)
        out = fitz.open()
        page = out.new_page(width=612, height=792)
        page.insert_image(page.rect, stream=cv2.imencode(".png", image)[1].tobytes())
        return out.tobytes()


def _fill(canvas: _Canvas, form: str, size: float = 9):
    for origin, text in FORMS[form]["fields"]:
        canvas.text(origin, text, size=size)


def _ssn_renderings():
    """name -> draw(canvas, origin, digits) for the employee/recipient SSN."""
    def plain(size=10, gray=0.0, rotate=0):
        def draw(c, o, d):
            if rotate:  # in the blank lower half of the page
                o = (300, 650) if rotate == 90 else (320, 450)
            c.text(o, _ssn(d), size=size, color=(gray,) * 3, rotate=rotate, secret=0)
        return draw

    # Renderings wider than any form field go in the blank lower half (below
    # 372 pt on every fixture), as they would sit in their own wide box.
    free = (60, 470)

    def groups(parts, size, gap, leaders=False):
        def draw(c, o, d):
            x, y, cut = free[0], free[1], 0
            for n, length in enumerate(parts):
                chunk = d[cut:cut + length]
                cut += length
                c.text((x, y), chunk, size=size, secret=0)
                width = fitz.get_text_length(chunk, fontsize=size)
                if leaders and n < len(parts) - 1:
                    dots = "." * max(1, int((gap - 4) / (size * 0.28)))
                    c.text((x + width + 2, y), dots, size=size)
                x += width + gap
        return draw

    def split_lines(size=10):
        def draw(c, o, d):
            o = free
            c.text(o, f"{d[:3]}-{d[3:5]}-", size=size, secret=0)
            c.text((o[0], o[1] + size * 1.3), d[5:], size=size, secret=0)
        return draw

    def comb(size, factor):
        def draw(c, o, d):
            o = free
            x = o[0]
            for ch in d:
                c.rect(fitz.Rect(x - size * 0.3, o[1] - size * 1.05, x + size * 0.9,
                                 o[1] + size * 0.35), width=0.5)
                c.text((x, o[1]), ch, size=size, secret=0)
                x += size * factor
        return draw

    def w9(size=10):
        def draw(c, o, d):
            o = free
            x = o[0]
            for n, chunk in enumerate((d[:3], d[3:5], d[5:])):
                width = fitz.get_text_length(chunk, fontsize=size)
                c.rect(fitz.Rect(x - 3, o[1] - size * 1.1, x + width + 3, o[1] + size * 0.4))
                c.text((x, o[1]), chunk, size=size, secret=0)
                x += width + 20
                if n < 2:
                    c.text((x - 13, o[1]), "-", size=size)
        return draw

    def boxed(fill, text_color):
        def draw(c, o, d):
            width = fitz.get_text_length(_ssn(d), fontsize=11)
            c.rect(fitz.Rect(o[0] - 4, o[1] - 12, o[0] + width + 4, o[1] + 4), fill=fill,
                   color=fill)
            c.text(o, _ssn(d), size=11, color=text_color, secret=0)
        return draw

    cases = {f"size-{s}pt": plain(size=s) for s in (4, 6, 8, 10, 12)}
    cases.update({f"gray-{g}": plain(gray=g / 255) for g in (200, 235)})
    cases.update({f"blur-{b}": plain(size=10) for b in (1, 2, 3)})
    cases.update({"rotated-90": plain(rotate=90), "rotated-270": plain(rotate=270)})
    for size in (8, 10):
        for gap in (14, 30):
            cases[f"gaps-{size}pt-{gap}pt"] = groups((3, 2, 4), size, gap)
        cases[f"leaders-{size}pt"] = groups((3, 2, 4), size, 20, leaders=True)
    cases["split-3-2-3-1"] = groups((3, 2, 3, 1), 10, 18)
    cases["split-lines"] = split_lines()
    for size, factor in ((6, 3), (10, 4), (14, 3)):
        cases[f"comb-{size}pt-x{factor}"] = comb(size, factor)
    cases["w9-boxes"] = w9()
    cases["white-on-black"] = boxed((0, 0, 0), (1, 1, 1))
    cases["white-on-30pct-gray"] = boxed((0.7, 0.7, 0.7), (1, 1, 1))
    cases["shaded-box"] = boxed((0.85, 0.85, 0.85), (0, 0, 0))
    return cases


def build_cases(seed: int = 6876) -> list[Case]:
    """The identifier corpus: every rendering on every form, plus EIN variants."""
    rng = random.Random(seed)
    cases = []
    renderings = _ssn_renderings()
    for form in FORMS:
        for name, draw in renderings.items():
            ssn, ein = _identifier(rng), _identifier(rng)
            canvas = _Canvas(form)
            _fill(canvas, form)
            draw(canvas, FORMS[form]["ssn"], ssn)
            canvas.text(FORMS[form]["ein"], _ein(ein), size=10, secret=1)
            cases.append(_finish(canvas, f"{form}/{name}", form, [ssn, ein],
                                 blur=float(name.split("-")[1]) if name.startswith("blur") else 0))
        # EIN with its two groups 180 pt apart.
        ein = _identifier(rng)
        canvas = _Canvas(form)
        _fill(canvas, form)
        origin = (60, 470)
        canvas.text(origin, ein[:2] + "-", size=10, secret=0)
        canvas.text((origin[0] + 180, origin[1]), ein[2:], size=10, secret=0)
        cases.append(_finish(canvas, f"{form}/ein-gap-180pt", form, [ein]))
    return cases


def _finish(canvas: _Canvas, name: str, form: str, identifiers: list[str], blur: float = 0):
    # All but each identifier's last four digits must be hidden.
    boxes = []
    for index, digits in enumerate(identifiers):
        mine = canvas.secret_char_boxes(index)
        assert "".join(c for c, _ in mine) == digits, (name, digits, mine)
        boxes.extend(rect for _, rect in mine[:-4])
    blur_box = None
    if blur:
        blur_box = fitz.Rect(boxes[0])
        for rect in boxes:
            blur_box |= rect
    return Case(name, form, canvas.rasterize(blur_box, blur), identifiers, boxes)


def build_clean_cases() -> list[Case]:
    """Held-out filled pages without identifiers, for the false-refusal rate."""
    cases = []
    for form in FORMS:
        for size in (9, 7):
            canvas = _Canvas(form)
            _fill(canvas, form, size=size)
            canvas.text((360, 760), "Tax year 2025  Form 1099 Rev. 04-2025  ZIP 78701-1234",
                        size=8)
            cases.append(Case(f"{form}/clean-{size}pt", form, canvas.rasterize(), []))
        canvas = _Canvas(form)
        _fill(canvas, form)
        canvas.text((360, 760), "Account 4471 0093 2215  Ref 2025-0417", size=8)
        cases.append(Case(f"{form}/clean-long-account", form, canvas.rasterize(), []))
    return cases


# ------------------------------------------------------------------- engines


def _garble(text: str, rng: random.Random) -> str:
    positions = [i for i, ch in enumerate(text) if ch in _CONFUSE]
    if not positions:
        return text
    chars = list(text)
    for i in rng.sample(positions, min(len(positions), rng.randint(1, 2))):
        chars[i] = _CONFUSE[chars[i]]
    return "".join(chars)


GARBLE = threading.Event()  # set while the confusion engine is in use


def confusion_engine(real_get_engine, seed: int = 6876):
    """Wraps the real engine: while GARBLE is set, every digit-bearing reading
    loses 1-2 characters to confusable letters."""
    rng = random.Random(seed)
    lock = threading.Lock()

    def get_engine():
        engine = real_get_engine()

        def call(image, **kwargs):
            prediction = engine(image, **kwargs)
            if not GARBLE.is_set():
                return prediction
            with lock:
                txts = tuple(_garble(t, rng) for t in (prediction.txts or ()))
                groups = [
                    [(_garble(t, rng), score, poly) for t, score, poly in group]
                    if isinstance(group, (list, tuple)) and group
                    and isinstance(group[0], (list, tuple)) and len(group[0]) == 3
                    else group
                    for group in (prediction.word_results or [])
                ]
            return SimpleNamespace(txts=txts or None, boxes=prediction.boxes,
                                   word_results=groups)

        return call

    return get_engine


# ------------------------------------------------------------------- scoring


def _covered(pdf: bytes, rect: fitz.Rect) -> bool:
    """The character box is blacked out in the output page."""
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        clip = fitz.Rect(rect.x0 + 0.5, rect.y0 + 0.5, rect.x1 - 0.5, rect.y1 - 0.5)
        if clip.is_empty:
            clip = rect
        pix = doc[0].get_pixmap(dpi=216, clip=clip, colorspace=fitz.csGRAY)
        values = np.frombuffer(pix.samples, np.uint8)
    return values.size == 0 or (values.mean() < 60 and (values > 128).mean() < 0.05)


def _reads_prefix(pdf: bytes, identifiers: list[str]) -> list[str]:
    from app.services import identifier_redaction as ir
    from app.services import rapid_ocr

    result = rapid_ocr.ocr_pages(pdf, force_ocr=True, with_pdf=False, low_confidence=True)
    texts = []
    for page in result["pages"]:
        texts += [w["text"] for w in page["words"]]
        texts += [w["text"] for w in page.get("_low_confidence_words", [])]
        texts += [w["text"] for w in page.get("_lines", [])]
    digits = [re.sub(r"\D", "", ir.normalize_confusions(t)) for t in texts]
    return [d for d in identifiers if any(d[:5] in x for x in digits)]


def run(cases: list[Case], engine: str, journal: Path | None = None) -> list[dict]:
    """Redact and judge each case in turn; outcomes are appended to `journal`
    (JSON lines) as they finish, and cases already in it are skipped, so an
    interrupted run resumes."""
    from app.services import identifier_redaction as ir

    done = {}
    if journal and journal.exists():
        for line in journal.read_text().splitlines():
            record = json.loads(line)
            done[(record["engine"], record["case"])] = record
    outcomes = []
    for case in cases:
        if (engine, case.name) in done:
            outcomes.append(done[(engine, case.name)])
            continue
        if engine == "confusions":
            GARBLE.set()
        try:
            result = ir.redact_identifiers(case.pdf)
            outcome = {"case": case.name, "engine": engine, "outcome": "redacted",
                       "redactions": len(result["redactions"])}
        except ir.RedactionIncompleteError as exc:
            result = None
            outcome = {"case": case.name, "engine": engine, "outcome": "refused",
                       "reason": str(exc)[:160]}
        finally:
            GARBLE.clear()
        if result is not None:
            # Judged with the plain engine only.
            pdf = result["pdf_bytes"]
            uncovered = sum(not _covered(pdf, rect) for rect in case.prefix_boxes)
            legible = _reads_prefix(pdf, case.identifiers) if case.identifiers else []
            outcome["uncovered_prefix_chars"] = uncovered
            outcome["legible_identifiers"] = len(legible)
            outcome["leak"] = bool(case.identifiers) and (uncovered > 0 or bool(legible))
        outcomes.append(outcome)
        if journal:
            with open(journal, "a") as handle:
                handle.write(json.dumps(outcome) + "\n")
    return outcomes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--only", help="regex over case names")
    args = parser.parse_args(argv)

    from app.config import settings

    settings.ocr_queue_dir = tempfile.mkdtemp(prefix="leak-corpus-")
    settings.ocr_workers = args.workers
    settings.ocr_queue_key = settings.ocr_queue_key or "leak-corpus-local"
    settings.ocr_backend = "rapid"

    cases, clean = build_cases(), build_clean_cases()
    if args.only:
        cases = [c for c in cases if re.search(args.only, c.name)]
        clean = [c for c in clean if re.search(args.only, c.name)]
    from app.services import rapid_ocr

    rapid_ocr._get_engine = confusion_engine(rapid_ocr._get_engine)
    journal = args.report.with_suffix(".jsonl")
    results = run(cases, "rapid", journal) + run(cases, "confusions", journal)
    clean_results = run(clean, "rapid", journal)
    leaks = [r for r in results if r.get("leak")]
    summary = {
        "identifier_pages": len(results),
        "leaks": len(leaks),
        "refused": sum(r["outcome"] == "refused" for r in results),
        "redacted_clean": sum(r["outcome"] == "redacted" and not r.get("leak") for r in results),
        "clean_pages": len(clean_results),
        "false_refusals": sum(r["outcome"] == "refused" for r in clean_results),
    }
    summary["false_refusal_rate"] = round(
        summary["false_refusals"] / max(1, summary["clean_pages"]), 3
    )
    args.report.write_text(json.dumps(
        {"summary": summary, "identifier_cases": results, "clean_cases": clean_results},
        indent=2,
    ))
    print(json.dumps(summary))
    for leak in leaks:
        print("LEAK", leak["engine"], leak["case"], leak)
    from app.services import ocr_queue

    ocr_queue.shutdown()
    return 1 if leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
