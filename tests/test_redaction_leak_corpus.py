"""TAX-6876: the held-out leak corpus (scripts/redaction_leak_corpus.py).

The generator test runs everywhere. The full run needs the RapidOCR models
and takes tens of minutes: set RUN_LEAK_CORPUS=1 (and RAPIDOCR_MODEL_DIR).
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _corpus():
    spec = importlib.util.spec_from_file_location(
        "redaction_leak_corpus", ROOT / "scripts" / "redaction_leak_corpus.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["redaction_leak_corpus"] = module
    spec.loader.exec_module(module)
    return module


def test_corpus_is_deterministic_and_carries_ground_truth():
    corpus = _corpus()
    cases = corpus.build_cases()
    again = corpus.build_cases()
    assert [c.identifiers for c in cases] == [c.identifiers for c in again]
    assert len(cases) == 6 * 28  # 27 SSN renderings + the split EIN, per form
    renderings = {c.name.split("/")[1] for c in cases}
    for required in ("size-4pt", "gray-235", "blur-3", "rotated-90", "rotated-270",
                     "gaps-8pt-30pt", "split-lines", "split-3-2-3-1", "comb-14pt-x3",
                     "leaders-8pt", "w9-boxes", "white-on-black", "white-on-30pct-gray",
                     "shaded-box", "ein-gap-180pt"):
        assert required in renderings, required
    for case in cases:
        hidden = sum(len(d) - 4 for d in case.identifiers)
        assert len(case.prefix_boxes) == hidden, case.name
        assert case.pdf.startswith(b"%PDF")
    clean = corpus.build_clean_cases()
    assert len(clean) == 18 and all(not c.identifiers for c in clean)


def test_confusion_engine_garbles_digits_into_letters():
    corpus = _corpus()
    import random

    garbled = corpus._garble("123-45-6789", random.Random(1))
    assert garbled != "123-45-6789"
    assert sum(a != b for a, b in zip(garbled, "123-45-6789")) in (1, 2)
    assert all(ch in "IZSGBq" for a, ch in zip("123-45-6789", garbled) if a != ch)


@pytest.mark.skipif(os.environ.get("RUN_LEAK_CORPUS") != "1", reason="set RUN_LEAK_CORPUS=1")
def test_the_corpus_has_no_leaks(tmp_path):
    corpus = _corpus()
    code = corpus.main(["--report", str(tmp_path / "report.json")])
    assert code == 0, (tmp_path / "report.json").read_text()[:4000]
