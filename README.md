# PDF Engine Toolbox

PDF Engine Toolbox is the PyMuPDF-based document-processing network service
used by TaxScout. It is free software licensed under
[GNU AGPL-3.0-or-later](LICENSE).
Third-party attribution is recorded in [NOTICE.md](NOTICE.md).

## Corresponding Source offer

Every network response includes a `Link` header with `rel="source"` and an
`X-Source-Code` header. `GET /source` provides the license, public repository,
the exact source commit embedded in the running image, and a downloadable
source archive. `GET /health` exposes the same build commit and source URL.
The source response also includes hash-pinned source archives for PyMuPDF and
its MuPDF engine, as recorded in `third-party-sources.json`.

The complete Corresponding Source is available at no charge from:

https://github.com/TaxScout-ai/pdf-engine-toolbox

For a production build, follow the exact `source_code_url` returned by the
running service. It points to the source revision used for that image. The
repository contains the application source, dependency manifests, tests,
Dockerfile, Compose configuration, and model-download tooling needed to build
and run the service. Secrets and customer data are not part of the source.

## Build and run

Production images must be built only after the source commit is available in
the public repository:

```bash
git clone https://github.com/TaxScout-ai/pdf-engine-toolbox.git
cd pdf-engine-toolbox
git checkout <full-public-commit-sha>
docker build \
  --build-arg SOURCE_COMMIT="$(git rev-parse HEAD)" \
  -t pdf-engine-toolbox:"$(git rev-parse --short HEAD)" .
```

For an immutable Docker Compose run, export the same full public revision first:

```bash
export SOURCE_COMMIT="$(git rev-parse HEAD)"
docker compose up --build
```

The build rejects missing, abbreviated, uppercase, otherwise malformed, or
non-public source revisions. It verifies that the exact revision is reachable
from the public repository before producing the image. The Compose service runs
the verified image without bind-mounting local source or enabling hot reload, so
its reported build commit remains true for the code serving requests.

The third-party source URLs are part of this offer. If an upstream source
archive becomes unavailable, an operator must mirror the hash-identical
archive on a no-charge network server and update the manifest before continuing
to operate that build.

After every deployment, verify the running source offer against the expected
public revision:

```bash
python scripts/verify_agpl_offer.py \
  https://pdf-engine.example.com \
  <full-public-commit-sha>
```

Periodically add `--verify-third-party-hashes` to download and verify the
PyMuPDF and MuPDF source archives. This is intentionally not the default
because the two archives are approximately 150 MB combined.

## Development

Install Python 3.12 dependencies from `requirements.txt`, configure the values
documented in `.env.example`, and run:

```bash
uvicorn app.main:app --reload
```

Development processes outside the container report `development` as their
build identity and link to the repository's default branch.

## RapidOCR text runtime (TAX-6490, TAX-6876, TAX-6995)

RapidOCR (ONNX Runtime, CPU) serves every model operation: `/text/ocr`,
identifier redaction, `/transform/orientation` and table recognition on page
images. Paddle (`paddlepaddle`, `paddleocr`, `paddlex`) was removed in
TAX-6995; there is no `OCR_BACKEND=paddle` rollback any more. A deployment
that still sets `OCR_BACKEND` (or `OCR_DET_LIMIT_SIDE_LEN`) starts normally;
the value is ignored and `ocr_backend_setting_ignored` is logged at startup.
The engine needs `WORKERS=1` and `RAPIDOCR_MODEL_DIR` pointing to a read-only
directory containing the models listed in
`app/services/rapid_ocr_models.py` (`MODEL_HASHES` for OCR, `TABLE_MODEL_HASHES`
for tables).
RapidOCR is the planned production engine: responses carry
`provenance.experimental: false`; earlier builds sent `true`. The field is
kept so readers of the old shape keep working, and nothing in the engine
depends on it.

The pinned runtime is RapidOCR 3.9.2 + ONNX Runtime 1.30.0, CPU only, using
PP-OCRv6 Small detection/recognition and the mobile v2 line classifier. Public
weights (31,749,509 bytes total) can be provisioned before startup from these
versioned upstream locations; the service verifies their SHA-256 before loading:

- https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx
- https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx
- https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx

RapidTable's SLANet-plus table-structure model (7,758,305 bytes) comes from
https://www.modelscope.cn/models/RapidAI/RapidTable/resolve/v2.0.0/slanet-plus.onnx
and is verified the same way (`rapid-table==3.0.2`).

Missing, altered or mismatched weights/packages abort startup and the build;
runtime does not download substitutes. Engines load lazily, one per OCR worker
thread. Rendering caps at 216 DPI and 16
million pixels/page. Use a container CPU/memory budget during deployment and
size it from measured whole-service peak RSS, including concurrent non-OCR
operations.

## OCR job queue (TAX-6876)

Every `POST /text/ocr` is a persistent job. Its PDF is spilled to
`OCR_QUEUE_DIR/inputs/` (default `/data/ocr-queue`, which must sit on the
persistent `/data` volume) and its state lives in `OCR_QUEUE_DIR/jobs.sqlite3`.
On startup, jobs that were `pending` or `processing` are re-queued in their
original order; RapidOCR jobs resume from the first page not yet recorded, and
a job whose every page was recorded before the restart is finished at once.
Completed and failed results stay readable for `OCR_RESULT_TTL_SECONDS`
(default 3600); after that a read answers `TASK_NOT_FOUND`.

### Client data at rest

The queue holds client documents (with SSNs) and their OCR text, so:

- Inputs (`inputs/<id>.bin`), results (`results/<id>.bin`) and per-page
  checkpoints (rows in `jobs.sqlite3`) are sealed with AES-256-GCM. Each is
  bound to its job id, so a file swapped between jobs or altered on disk fails
  authentication, and that job fails instead of reading it.
- The key is `OCR_QUEUE_KEY`: 32 bytes in URL-safe base64, or any passphrase
  (HKDF-SHA256). **Production and staging set a dedicated `OCR_QUEUE_KEY`.**
  Without it, the key is HKDF-derived from `PDF_ENGINE_SECRET`. The engine
  refuses to start when neither is set to a non-default value
  (`PDF_ENGINE_SECRET=change-me` and no `OCR_QUEUE_KEY`).
- Changing the key makes data sealed under the old key unreadable:
  - a queued job fails with an authentication error;
  - a stored result answers `GET /tasks/{id}` with a `failed` task ("OCR
    result unreadable"), never HTTP 500.

  Rotate only with an empty queue.
- OCR results are not written to the plaintext file cache (`CACHE_DIR`). A
  repeated identical request (same bytes and parameters) is answered from the
  newest unexpired result in the sealed ledger, so there is one copy under one
  retention rule.
- `OCR_QUEUE_DIR` and its subdirectories are mode 0700. Files and the SQLite
  ledger (including `-wal`/`-shm`) are mode 0600.
- An input is deleted as soon as its job finishes. Every
  `OCR_SWEEP_INTERVAL_SECONDS` (default 300) a sweeper:
  - deletes expired jobs and their results;
  - deletes files no ledger row refers to, and temporaries older than ten
    minutes;
  - deletes page-checkpoint rows of finished jobs;
  - restarts any OCR worker thread that died.

A failure after a page was read fails that job with the reason; the worker
carries on. Examples: the disk filling up while checkpointing, assembling or
writing the result, or a ledger write that is rolled back.

`GET /tasks/{id}` keeps its contract (`pending` / `processing` / `completed` /
`failed`, `result` or `error`; unknown ids answer HTTP 200 with
`{"success": false, "error": {"code": "TASK_NOT_FOUND", ...}}`). Unfinished OCR
jobs also report `queue_position` (unfinished jobs submitted earlier) and
`eta_seconds` (estimate from measured page time).

| Setting | Default | Meaning |
| --- | --- | --- |
| `OCR_WORKERS` | CPUs available (affinity and cgroup quota) | concurrent OCR inferences; one RapidOCR engine with its own ONNX Runtime sessions per worker thread |
| `OCR_INTRA_OP_THREADS` | CPUs // `OCR_WORKERS`, at least 1 | ONNX Runtime intra-op threads per session and OpenCV threads; inter-op is 1 |
| `OCR_MAX_ACTIVE_JOBS` | max(2, `OCR_WORKERS`) | documents interleaved page by page; later documents wait FIFO |
| `OCR_QUEUE_MAX_JOBS` | 500 | unfinished jobs accepted before 429 |
| `OCR_RESULT_TTL_SECONDS` | 3600 | how long finished results stay readable |
| `OCR_QUEUE_KEY` | derived from `PDF_ENGINE_SECRET` | key for queue data at rest |
| `OCR_SWEEP_INTERVAL_SECONDS` | 300 | retention sweep and worker supervision |
| `OCR_SYNC_TIMEOUT_SECONDS` | 1200 | synchronous OCR (identifier redaction) fails closed after this |

A RapidOCR document is split into one unit per page, so one large scan uses
every idle worker and page results are reassembled in page order with
unchanged visual-PDF-point coordinates. Idle workers take pages from active
documents round-robin, so a short document submitted behind a long scan
finishes first. Nothing answers "busy": the only refusal is HTTP 429 with a
`Retry-After` header (seconds) and error code `OCR_QUEUE_FULL` when
`OCR_QUEUE_MAX_JOBS` unfinished jobs are queued. **Callers must retry a 429
after `Retry-After` seconds.**

`GET /health` lists `capabilities` (additive): `ocr-queue:persistent`,
`identifier-redaction:rapid-ocr`, `orientation:rapid-ocr` and
`tables:rapid-table`. All four are always present on this version.

`WORKERS` (uvicorn processes) must stay 1: the queue is scheduled and
recovered by one process. Scale with `OCR_WORKERS` instead.

Orientation and image-table requests run on the same workers as in-memory
jobs of one unit per page (they are not persisted and do not count toward
`OCR_QUEUE_MAX_JOBS`). A job left in the ledger by an engine that ran with
`OCR_BACKEND=paddle` is re-planned as RapidOCR on start and resumes.

Native digital text keeps the existing skip behavior without loading models.
Rapid scan results include `provenance`, per-page `text_source`, and per-word
`polygon`/`geometry_kind` alongside existing fields. Coordinates are **visual
PDF points**, including PDF rotation. Word boxes are CTC-derived estimates,
not manually validated character boundaries. No field is independently
verified merely because OCR found the same digits (TAX-6471 tracks unsafe
sign/decimal/wrong-field matching).

Identifier redaction reads page images with RapidOCR through the same OCR
workers (same words-and-boxes contract).

**What is redacted.** Every reading OCR produced is searched: regular words,
RapidOCR words below the 0.5 review threshold, and whole recognized lines
(these extras stay inside redaction; the `/text/ocr` response is unchanged).
Before matching, common OCR confusions inside numeric tokens are read as
digits (`I l | O o D S s B Z G q`, so "I23-45-678S" is an SSN). Any run of
exactly nine digits with any separators — spaces, dashes, dot leaders — is an
identifier, as is a group continued on the line below ("123-45-" / "6789").
Everything but the last four digits is blacked out. A rotated reading is
blacked out whole. Amounts, dates, ZIP+4 and phone numbers are not
identifiers.

**When a page is refused** (the document goes to manual review instead of
being returned):
- OCR failed on a page;
- a reading is one or two digits off an identifier's shape (7–11 digits in
  3-2-4, 2-7 or one block), or is a comb field read digit by digit — a
  misread identifier looks like this;
- print that no reading covers could hold an identifier: a run of 4+ marks on
  a line (horizontal or vertical), a run about nine digits long, or digit-sized
  printed marks anywhere on the page summing to nine or more, whatever the
  gaps or line breaks between them. White print on black or gray counts as
  print. Rules, box grids, empty frames, dust and the corners of large filled
  shapes (logos, shaded boxes) do not;
- an OCR run that exceeds `OCR_SYNC_TIMEOUT_SECONDS`.

**What is claimed, and how it is measured.** `scripts/redaction_leak_corpus.py`
is a held-out evaluation corpus. It is never used to tune thresholds. It fills
IRS 2025 form pages (W-2, 1099-INT/DIV/R/NEC, 1098; `tests/fixtures/irs2025`)
with synthetic SSNs and EINs in 27 renderings each:
- 4–12 pt and light gray print;
- blur sigma 1–3 and 90/270° rotation;
- 14–30 pt gaps, dot leaders and a 180 pt EIN gap;
- splits across lines and 3|2|3|1 groups;
- comb fields and W-9 boxes;
- white on black or 30% gray, and print inside shaded boxes.

It rasterizes each page at 200 DPI and runs real RapidOCR, plainly and
through an engine that garbles one or two characters per reading. A leak is a
returned page on which a hidden digit is still legible: not blacked out at its
ground-truth position, or re-read by OCR. Results: pending the first full run.

Re-run it with `RAPIDOCR_MODEL_DIR=… python scripts/redaction_leak_corpus.py`.
Its work files go to `.corpus-work/` (gitignored), never `/tmp`.

## Page orientation (TAX-6995)

`POST /transform/orientation` keeps its contract: for each checked page whose
text is turned 90/180/270 degrees with confidence of at least
`min_confidence`, it returns `rotation` (the clockwise correction) and
`detected_orientation` (= (360 - rotation) % 360, the meaning the PaddleX
`PP-LCNet_x1_0_doc_ori` classifier gave it), plus `checked_page_count` and
`skipped_page_count`. Native-text pages are skipped unless
`scanned_only=false`.

It no longer uses a document-orientation model. Each page is rendered at 144
DPI and judged from its text lines (`app/services/page_orientation.py`):

1. The RapidOCR detector finds the lines on the page scaled to 1024 px (again
   at full resolution when it finds fewer than 8). Long thin boxes vote, by
   length, for lines running across or down the page; square boxes do not.
2. The page is turned so its lines run across, and its 32 longest lines are
   read by the recognizer as they are and turned 180 degrees. A line votes for
   a direction when one reading is credible (score at least 0.5) and beats the
   other by at least 0.15; votes are weighted by that margin.
3. Confidence = (share of line length on the winning axis) x (share of the
   vote for the winning direction). Fewer than 3 voting lines: confidence 0,
   the page is never turned.

Measured on `scripts/orientation_corpus.py` (public IRS forms and synthetic
pages only, each turned 0/90/180/270, clean, scan-like and faint variants;
1,728 pages; `min_confidence` 0.85 as the app sends it):

| Variant (576 pages each) | PaddleX classifier: correct / wrong / missed | RapidOCR lines: correct / wrong / missed |
| --- | --- | --- |
| clean | 543 / 0 / 33 | 576 / 0 / 0 |
| scan | 507 / 0 / 69 | 576 / 0 / 0 |
| faint | 391 / 0 / 185 | 510 / 0 / 66 |
| **total (1,728)** | **1,441 / 0 / 287** | **1,662 / 0 / 66** |

Re-run with `python scripts/orientation_corpus.py build --irs-dir <IRS_PDF>`
then `RAPIDOCR_MODEL_DIR=… python scripts/orientation_corpus.py score`.

A wrong answer counts any correction other than the right one, including any
correction on an upright page. Cost: about one RapidOCR detection plus a
batch of up to 64 short line readings per scanned page, on the OCR workers
(the classifier it replaces was a ~0.05 s CNN pass; on the shared, loaded
build host the new judgement took 1.7-2.7 s per page with 2 workers x 2
threads).

## Tables (TAX-6995)

`POST /text/tables` preserves the table list and adds explicit page outcomes.
Digital pages whose text passes the quality gate use PyMuPDF `find_tables`
(no model). Other pages, and digital pages where PyMuPDF finds nothing, are read from the
page image (`app/services/table_ocr.py`), rendered at `dpi` capped at 216:

- **Where:** ruled tables only. Long horizontal and vertical strokes are kept
  by morphological opening; a region whose rules cross at least 9 times (a
  2 x 2 grid of cells) is a table. Borderless tables on a scan are not found,
  and such a page reports unsupported structure. The optional fallback below
  can attempt those pages.
- **Structure:** RapidTable SLANet-plus (ONNX, hash-pinned).
- **Text:** the page is read once with RapidOCR and RapidTable's matcher puts
  each line in its cell.

`strategy` accepts `auto` (default), `pymupdf`, and `rapidtable`; the old name
`ppstructure` is kept as an alias of `rapidtable`. Page images support only
`language=en` (another language answers 422 `UNSUPPORTED_LANGUAGE`).
PP-StructureV3 and its ~1.5 GB chart model are gone.

`scripts/benchmark_rapid_pilot.py` exercises this adapter against the private,
reviewed real-document manifests from TAX-6460. Supply `--benchmark-tools`
with that task's `scripts/ocr-benchmark` directory, `--input` with a manifest
directory, and `--output` with a new directory outside this repository. Raw
OCR data stays private. Results measure known field regions, not automatic
tax-field identification, CPA review time, or an end-to-end Gemini pipeline.

## License

Copyright (C) 2026 TaxScout.AI Inc. and contributors.

This program is distributed under AGPL-3.0-or-later, without any warranty. See
[LICENSE](LICENSE). If you modify this program and make the modified version
available for users to interact with over a network, AGPL section 13 requires
you to offer those users the Corresponding Source of that modified version.

### RapidOCR scan preparation and manual review (TAX-6490)

The opt-in `rapid-3.9.2-ort-1.30.0-v6-small-prepared-word-v5` profile
checks the existing PDF text layer before skipping recognition. Invisible text,
invalid characters, insufficient text and substantial raster content trigger OCR.
Accepted digital pages return actual native word rectangles without loading a model.

Image preparation conservatively crops large blank margins and uses connected
glyph geometry to detect sideways text. Ambiguous orientation is reported, not
claimed corrected. This heuristic is not a general document-orientation model.
Only one configured OCR engine and one recognition pass are used per scanned page.
Crop and rotation transforms are inverted before returning every word polygon;
`visual_width` / `visual_height` always refer to the original visible page.

The response keeps the quality decision, preparation metadata and model profile.
These are review hints, never proof that a tax field is correct. Prepared
searchable-PDF output is not certified (`searchable_output_verified=false`); the
application keeps the original PDF for review. No fallback or second model is
invoked. The application must explicitly enable its separate
manual-review integration before accepting this profile; production defaults
remain unchanged.

## Primary review OCR and rollback (TAX-6713)

`/text/ocr` now defaults to `OCR_BACKEND=rapid`, using PP-OCRv6 Small with
RapidOCR 3.9.2 / ONNX Runtime 1.30.0. Keep `WORKERS=1`: one lazily loaded engine,
serialized inference, and one rendered page at a time. Usable native PDF text
skips inference. The image includes SHA-256-verified weights in
`/opt/rapidocr/models`; failed downloads abort the build and missing/mismatched
runtime assets abort startup. Request processing never downloads weights or
silently invokes another OCR backend.

Deploy the matching application configuration: `OCR_RAPID_REVIEW_ENABLED` is
on by default (unset or `true`), `OCR_ENABLED=true`, `OCR_DPI=150`. This is
manual source review only. Word boxes are estimates; partial results require
inspection of the original. OCR agreement does not approve tax data. The
searchable PDF output remains unverified and the app does not replace originals
with it.

There is no OCR rollback to Paddle since TAX-6995: Paddle is not installed.
An app still configured for the Paddle pairing (`OCR_RAPID_REVIEW_ENABLED=false`)
fails closed at its geometry gate on these results; set it back to its default
before deploying this engine. Existing stored review receipts retain their
provenance and can still be viewed.

Identifier redaction on scans reads with RapidOCR (TAX-6876); its measured
leak rate is the held-out corpus above.


## Optional table structure fallback (TAX-7070)

`POST /text/tables` returns a document status and per-page outcomes as well as
successful tables. Each table includes physical cells with row/column spans,
raw text, page and PDF-point bounds when available. `unsupported` is distinct
from a blank page and from an inference error. Clients must inspect outcomes;
a successful HTTP response does not establish complete table coverage.

The HTTP route uses a 100-second absolute budget (including download,
threadpool scheduling and PDF-lock contention), below the application's
120-second timeout. The source download is capped at 30 seconds. At the
queue deadline, completed pages are retained and unfinished pages report
`failed` / `table_request_timeout`. Pending pages stop; an already running
inference is not forcibly killed and cannot mutate the returned result.
Preparation that cannot finish returns HTTP 504. Table jobs remain in memory;
these deadlines do not add restart recovery or automatic retries.

The default build remains RapidOCR/RapidTable. For a staging evaluation image,
build the same public source revision with `--build-arg INCLUDE_DOCLING=true`.
This adds the pinned CPU-only optional requirements and hash-verified model
artifacts. Set `TABLE_FALLBACK=docling`, `DOCLING_MODEL_DIR=/opt/docling/models`
and a bounded `TABLE_FALLBACK_MAX_PAGES` (default 4 per request). The default
`TABLE_FALLBACK=none` never imports Docling or Torch. Explicit `pymupdf` and
`rapidtable` comparisons do not silently invoke the fallback.

Only unresolved/failed pages use TableFormer. Complete native tables preserve
geometry warnings without loading a second model. Native text passing the
quality gate is reused before image OCR. All page work uses the shared OCR
queue; one process-wide lock bounds the heavy converter. Runtime downloads,
remote services and external plugins are disabled. Provision artifacts during
the build; do not depend on first-request downloads.

See [the real financial-table evaluation](docs/benchmarks/TAX-7070-financial-tables.md)
for coverage, timing, memory and the remaining limitations. The optional profile
needs a separately measured memory budget; it is not a free accuracy upgrade.
