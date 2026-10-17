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

## RapidOCR evaluation pilot (TAX-6490)

`OCR_BACKEND=rapid` is the code default since TAX-6713; a deployment may pin
`OCR_BACKEND=paddle` as the rollback. Rapid needs `WORKERS=1` and `RAPIDOCR_MODEL_DIR` pointing to
a read-only directory containing the three models listed in
`app/services/rapid_ocr.py::MODEL_HASHES`. The ordinary `/text/ocr` endpoint and
task polling contract select one backend; there is no automatic Paddle retry.
The application currently rejects experimental results from production
verification. This is a standalone pilot, not a production replacement.

The pinned runtime is RapidOCR 3.9.2 + ONNX Runtime 1.30.0, CPU only, using
PP-OCRv6 Small detection/recognition and the mobile v2 line classifier. Public
weights (31,749,509 bytes total) can be provisioned before startup from these
versioned upstream locations; the service verifies their SHA-256 before loading:

- https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx
- https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx
- https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx

Missing, altered or mismatched weights/packages fail the task; runtime does
not download substitutes. Engines load lazily, one per OCR worker thread. The
Rapid backend skips Paddle startup warmup. Rendering caps at 216 DPI and 16
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
  (HKDF-SHA256). Without it, the key is HKDF-derived from `PDF_ENGINE_SECRET`.
  Changing the key makes jobs queued under the old key fail with an
  authentication error, so rotate only with an empty queue.
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

`GET /health` lists `capabilities` (additive). `ocr-queue:persistent` is always
present on this version; `identifier-redaction:rapid-ocr` is present only when
`OCR_BACKEND=rapid`, i.e. when identifier redaction really reads with RapidOCR.

`WORKERS` (uvicorn processes) must stay 1 on both backends: the queue is
scheduled and recovered by one process. Scale with `OCR_WORKERS` instead.

Native digital text keeps the existing skip behavior without loading models.
Rapid scan results include `provenance`, per-page `text_source`, and per-word
`polygon`/`geometry_kind` alongside existing fields. Coordinates are **visual
PDF points**, including PDF rotation. Word boxes are CTC-derived estimates,
not manually validated character boundaries. No field is independently
verified merely because OCR found the same digits. The application's legacy
rotation normalization and hard-coded Paddle source label must be adapted
before accepting these results into the production pipeline (TAX-6471 also
tracks unsafe sign/decimal/wrong-field matching).

Identifier redaction follows `OCR_BACKEND`: with `rapid` it reads page images
through the same OCR workers (same words-and-boxes contract) and never imports
Paddle. It fails closed in two cases, and the document is refused rather than
returned with an identifier possibly left readable:
- a page OCR failed on (status `failed`);
- a page OCR read no text from (status `empty`) while its image shows ink. An
  `empty` page is accepted only when it is genuinely blank, for example a
  blank scan back. Blank means that, rendered at the OCR DPI in grayscale, at
  most 0.01% of its pixels differ from the page's median gray by more than 32
  levels. Faint print counts as ink at that threshold. Variance alone is not
  used, because faint print barely moves it;
- an OCR run that exceeds `OCR_SYNC_TIMEOUT_SECONDS`. Table recognition (`/text/tables`) and orientation
(`/transform/orientation`) keep their Paddle models, which load lazily on the
first such request; every Paddle inference in the process is serialized by one
lock (TAX-6255). Invoking those endpoints loads Paddle in addition to Rapid,
so a Rapid-only memory estimate is not a whole mixed-workload budget.
Rollback both OCR and redaction by selecting `OCR_BACKEND=paddle` and
restarting; Paddle OCR jobs run through the same queue, one unit per document. Backend-specific cache keys prevent reusing Rapid results as
Paddle results after rollback. Existing cached Paddle entries are invalidated
once by the new versioned key.

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
application keeps the original PDF for review. No automatic Paddle fallback or
second model is invoked. The application must explicitly enable its separate
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

Explicit rollback requires **both** `OCR_BACKEND=paddle` on this service and
`OCR_RAPID_REVIEW_ENABLED=false` on the app, followed by process restarts. Cache
keys separate backends. A mismatched pair fails closed at the app geometry gate;
never reinterpret Rapid coordinates as Paddle evidence. Existing stored review
receipts retain their provenance and can still be viewed after rollback.
Paddle remains installed for this rollback and for separate identifier-redaction
and table/classification operations; those operations are not changed here.
The earlier opt-in pilot instructions above document the evaluation history;
this section defines the new default. Production activation is a separate,
coordinated release, not a side effect of local integration tests.

`DOCUMENT_IDENTIFIER_REDACTION=enforce` is not supported with Rapid manual
review: its receipts cannot automatically restore full identifiers into the
vault. The matching app rejects document model calls before redaction/inference
for this combination. Keep the existing redaction policy and use the paired
Paddle rollback settings above for enforced redaction; do not turn redaction
off to activate Rapid. The default `text-layer` redaction policy is unchanged.
