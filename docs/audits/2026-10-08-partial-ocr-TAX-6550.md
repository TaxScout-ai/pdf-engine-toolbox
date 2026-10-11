# TAX-6550 — preserve readable OCR pages

Local implementation and evaluation, 2026-10-08. Not deployed. App counterpart: `codex/TAX-6550-ocr-partial-results` in taxscout-cpa.

The evaluated v5 adapter failed a whole PDF on RapidOCR's empty detector sentinel (real-21) or a degenerate word box (real-1523). The v6 adapter isolates recoverable page/preparation failures, retains valid word geometry, and reports page status and document coverage. Missing line groups, discarded low-confidence/unusable words and invalid geometry are explicitly incomplete. Model/package pin errors and MemoryError remain fatal. No fabricated coordinates or Paddle fallback.

Pinned runtime: RapidOCR3.9.2, ONNX Runtime1.30.0, PP-OCRv6 Small detection/recognition, existing mobile orientation model; CPU2 intra/2 inter threads, one request at a time. Profile `rapid-3.9.2-ort-1.30.0-v6-small-prepared-word-v6`. Actual rerun150DPI (216 cap), same24 original PDF hashes/44pages, same preprocessing and weights. Existing usable native PDF text is retained. No new Gemini inference.

Final direct engine run: 185.837s sum, median4.9085s, nearest-rank p9520.370s, process maximum RSS970.69MiB. Warm process across documents; model load is charged to its first required use. Timings include opening/rendering/preparation/inference/geometry/search-layer serialization, exclude HTTP/storage/Gemini. Shared-host load was not controlled, so this is not a speedup claim.

24/24 returned usable words;13 documents complete,11 partial. Page counts:32 complete,11 partial,1 empty,0 failed.80 words discarded. `complete` means pipeline coverage, not perfect recognition. Empty does not prove a blank source. real-21:2.664s,3 complete pages and1 empty; real-1523:12.840s,partial,14 discarded words.

Validation:26 engine/preprocessing tests pass, including empty sentinel, invalid/missing line geometry, recoverable page/render failures, fatal bad configuration, async/cache metadata roundtrip and resource limits. Independent reviewer initially found preparation isolation and missing-line coverage gaps; fixed and rechecked PASS/CLEAN. Python byte-compilation and diff whitespace check pass. Ruff is not installed in this evaluation venv; lint was not claimed.

Private raw receipts are under `/home/taxscout/.codex/artifacts/TAX-6550/rerun-final`; no taxpayer text or values are committed. Frozen manifest SHA256:ab14b920bac3398746a31fd4cad3904e6143a5381ec4da133ac588814e91805d. The initial development rerun is separate and is not the final measurement.
