# Financial-table evaluation — TAX-7070

Date: 2026-10-20. Development evaluation, not a production accuracy guarantee.

## Decision

Keep RapidOCR for text. Native PDF extraction remains the first choice when
its text layer passes the quality check. RapidTable alone is not sufficient
for the selected financial tables. Docling/TableFormer improves coverage
substantially, but requires review and a larger resource budget. Its output
must not automatically approve amounts or create source-verified tax facts.
The optional fallback is disabled by default.

## Corpus and labels

40 actual financial pages selected from the local private document collection:
broker statements/1099-B, three image-only brokerage pages, two bank statement
pages, borderless tables, rotated pages and continuation pages. Selection is
weighted toward brokerage documents; it is not a random production sample.
Only specified financial regions were labelled, not every table on each page.
Continuation pages were evaluated separately, not stitched across pages.

225 labelled rows, including 221 rows with numeric values; 603 numeric cells,
of which 431 are money amounts. Labels were visually reviewed by the agent.
Digital-page text was transcribed from native glyphs within manually checked
cell regions; scan amounts were manually transcribed. There has been no
independent CPA adjudication. A leading-digit annotation error on F003 was
corrected before the figures below (earlier preliminary denominators were
599 numeric / 428 money cells).

No document text, customer identifiers, PDFs or screenshots are committed.
Private inputs/results are in the operator's protected TAX-7070 artifact
folder. The metric implementation is `scripts/evaluate_financial_tables.py`.

## Method and results

Predictions are associated with labelled cells by geometry **before** comparing
values, one prediction per labelled cell. A correct value elsewhere does not
count. Amounts and signs must agree after documented numeric normalization.
A missing numeric row means no labelled numeric cell in that row was located;
it is not a full semantic row-recall metric. Column conflicts count a predicted
logical column assigned inconsistently to labelled columns, not complete
header/Box correctness. Geometric association does not prove the browser
highlight is visually exact.

| Pipeline | Exact money / 431 | Exact numeric / 603 | Missing numeric rows / 221 | Wrong located money | Column conflicts |
|---|---:|---:|---:|---:|---:|
| Native PDF + RapidTable baseline | 85 (19.7%) | 157 | 165 | 0 | 0 |
| Forced RapidTable image path | 58 (13.5%) | 110 | 177 | 5 | 1 |
| Raw Docling/TableFormer output | 384 (89.1%) | 537 | 15 | 16 | 0 |
| Docling through strict engine adapter | 368 (85.4%) | 504 | 32 | 16 | 0 |

The strict adapter refuses structurally overlapping/invalid tables on two
pages instead of publishing misleading merged-cell geometry. Other valid
tables on those pages survive, and the pages report `unsupported`. Raw Docling
reported conversion success on all 40 pages; that is not equivalent to correct
tables. The adapter reported 38 complete and two unsupported pages.

No pure sign-flip error was observed among numerically comparable cells. This
does not mean sign recognition is perfect: missing cells, merged text and
wrong magnitudes are counted separately and cannot prove a correct sign.
Observed failures include merged adjacent amounts, two transactions merged
into one row, a wholly missed financial region, footnote text attached to an
amount, and actual digit substitutions on scans.

## Models, versions and timing

- RapidOCR 3.9.2; ONNX Runtime 1.30.0 CPU; PP-OCRv6 Small detector/recognizer;
  mobile v2 line classifier. RapidTable 3.0.2, SLANet-plus. PyMuPDF 1.27.1.
- Docling 2.140.0; TableFormer ACCURATE, cell matching enabled; Heron layout;
  RapidOCR with the same pinned OCR weights. CPU only, no external inference.
  Model revisions and checksums are in the build downloader/manifest.
- Containers had a 2-CPU quota and no network access during inference.
  Rapid baseline used a 3-GiB memory limit; Docling used 4 GiB.

| Run | Sum of page times | Median page | Memory observation |
|---|---:|---:|---:|
| Native + RapidTable baseline | 6.659 s | 0.102 s | 364 MiB sampled peak |
| Forced RapidTable | 113.835 s | 0.061 s | 832 MiB sampled peak |
| Raw Docling, CPU-quota controlled | 656.237 s | 9.924 s | 2265.586 MiB process peak |

Very short RapidTable medians reflect unsupported pages returning early,
not successful table recognition at that speed. Native/RapidTable modes were
interleaved in one process; their model caches and peaks are not isolated cold
starts. The first forced RapidTable page took 18.562 s. The shared host had
other workloads, so these numbers are observations, not service-level promises.

The raw Docling environment had OpenCV 5.0.0.93 / Pydantic 2.14.1. A separate
40-page adapter run used engine-compatible OpenCV 4.13.0.92 / Pydantic 2.12.5,
with the same model weights. Two pages were rerun after partial-result handling
was fixed; its combined timings therefore are not a clean single-pass latency
comparison. Report its accuracy separately from the raw-model timing above.

## Mixed operation load

A real five-request test ran two scan OCR jobs, one orientation job, one native
table job and one scan table job together, with two queue workers, one inference
thread per worker and a maximum of three active jobs. This exercised actual
models, not mocked queue sleeps.

Without Docling: 19.389 s total, 1206.02 MiB process peak, at most two simultaneous
queue units, no worker restart. Both OCR jobs and orientation completed; native
tables completed; the scan table correctly reported unsupported.

The first optional-fallback run took 114.017 s and peaked at 2445 MiB. It
unnecessarily sent a complete native result with geometry warnings through
Docling. The policy was corrected: complete native tables retain their warnings
without invoking a second model. The repeat took 61.796 s, peaked at 2239.77 MiB, and
completed all five jobs without worker restarts. Queue concurrency remained
bounded at two. The native-table request waited in the shared queue and
finished in 25.841 s; the scan-table request finished in 61.778 s. Five requests demonstrate bounded concurrency, not
sustained overload capacity or a production SLA.

## Contract and rollout limits

Each physical cell retains original text, page, row/column, row/column span and
PDF-point BBOX when actually available. Missing geometry stays missing.
Malformed structure is never patched by inventing coordinates. Outcomes
separate complete, empty, unsupported and failed; partially successful documents
keep successful tables and report incomplete pages. Empty is currently claimed
only for an unambiguously blank page; nonblank pages without a detected table
remain unsupported, since absence has not been established.

No automatic cross-page table stitching, unmeasured high-resolution retry,
printed-column-to-tax-fact approval, or independent owner verification is
established by this benchmark. Those remain separate acceptance work. Keep the
fallback opt-in on staging until the final image, UI source navigation and
fact lifecycle checks pass. Do not treat this report as production approval.

## Post-review regression pass

RapidTable 3.0.2 can discard a zero predicted box while keeping the HTML cell.
The adapter now rejects unequal physical-cell/box counts and invalid boxes
**before** text matching, avoiding a silent shift into the next column. Invalid
structure becomes an unresolved region eligible for the configured fallback.
The missing-middle-box regression proves the matcher is not called.

The 40-page native/RapidTable comparison was rerun after this change. Labelled
accuracy counts are unchanged (85/431 and 58/431 exact monetary cells). Repeated
native baseline: 5.478 s summed, 0.092 s median, 356.98 MiB sampled peak. Repeated
forced RapidTable: 107.424 s summed, 0.059 s median, 816.32 MiB peak. The very low
median includes many early unsupported pages and is not successful-table speed.

Validation: full engine suite before this final guard: 453 passed, 1 skipped.
Final affected table suite after the guard: 45 passed, including actual pinned
models. Independent bounded source review PASS/CLEAN (Bacon, 2026-10-20): parser,
status contract, shared queue, fallback and model packaging. The reviewer did
not duplicate test runs. Final image/deployment acceptance remains separate.
