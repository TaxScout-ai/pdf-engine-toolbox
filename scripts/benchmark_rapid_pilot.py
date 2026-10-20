"""Exercise the service adapter against reviewed TAX-6460 real-document fixtures.

Requires the private manifest directory and the TAX-6460 benchmark tools checkout.
Never stores source text in Git. Known field regions are supplied to scoring;
this does not measure automatic field identification or CPA review time.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-tools", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT) or output.exists():
        raise ValueError("Output must be a new private directory outside the repository")
    os.umask(0o077)
    output.mkdir(parents=True)
    sys.path.insert(0, str(args.benchmark_tools.resolve()))
    from benchmark import read_region
    from real_benchmark import load_cases, score

    from app.services import pdf_service
    from app.services.rapid_ocr import MODEL_HASHES, PACKAGES, PROFILE

    manifest, manifest_hash = load_cases(args.input)
    summary = {
        "profile": PROFILE,
        "manifest_sha256": manifest_hash,
        "model_sha256": MODEL_HASHES,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in [*PACKAGES, "PyMuPDF", "numpy", "opencv-python", "pillow"]
        },
        "load_average_start": os.getloadavg(),
        "calls": [],
    }
    totals = Counter()
    start = time.perf_counter()
    for case in manifest["cases"]:
        data = (args.input / case["path"]).read_bytes()
        with fitz.open(stream=data, filetype="pdf") as doc:
            width, height = doc[0].rect.width, doc[0].rect.height
        began = time.perf_counter()
        result = pdf_service.ocr_pages(data)
        elapsed = time.perf_counter() - began
        words = []
        for word in result["pages"][0]["words"]:
            box = word["bbox"]
            words.append(
                {
                    "text": word["text"],
                    "confidence": word["confidence"],
                    "bbox": [
                        box["x"] / width,
                        box["y"] / height,
                        (box["x"] + box["w"]) / width,
                        (box["y"] + box["h"]) / height,
                    ],
                }
            )
        fields = {
            key: read_region(words, field["region"], "money")
            for key, field in case["fields"].items()
        }
        counts = Counter(
            score(fields[key], field["expected"]) for key, field in case["fields"].items()
        )
        totals.update(counts)
        summary["calls"].append(
            {"case": case["id"], "seconds": elapsed, "counts": dict(counts), "words": len(words)}
        )
        (output / (case["id"] + ".json")).write_text(json.dumps({"fields": fields, "words": words}))
        print(case["id"], dict(counts), round(elapsed, 3), flush=True)
    times = sorted(call["seconds"] for call in summary["calls"])
    summary.update(
        {
            "counts": dict(totals),
            "wall_seconds": time.perf_counter() - start,
            "median_seconds": statistics.median(times),
            "max_seconds": max(times),
            "cold_first_seconds": summary["calls"][0]["seconds"],
            "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "load_average_end": os.getloadavg(),
            "limitations": [
                "Known field regions, single reviewer ground truth",
                "No Gemini, field binding or CPA time evaluation",
                "One CPU process, shared host load, includes PDF serialization",
                "RSS is entire process high-water mark, not incremental model memory",
            ],
        }
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print({key: value for key, value in summary.items() if key not in ["calls", "model_sha256"]})


if __name__ == "__main__":
    main()
