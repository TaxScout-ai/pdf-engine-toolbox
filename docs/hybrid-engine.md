# The hybrid engine: lite + PaddleOCR + VL 1.6

Written down because this decision has now been re-derived twice from scratch
after a context loss, and the second time it was re-derived *wrongly* — VL was
reported as a dead end when the measurement said something narrower. Everything
below is measured, not recalled. If you are picking this up cold, you do not
need to re-run any of it.

## The shape

Three components, because no single one does all three jobs.

| | job | why not the others |
|---|---|---|
| **pdf-engine-lite** (1.08 GB) | text extraction, page ops | serves the 81% that never need recognition; too small to hold a recogniser |
| **PaddleOCR** (full image) | word-level geometry | the only one that emits per-word boxes |
| **PaddleOCR-VL 1.6** | accurate reading of hard scans | reads digits the standard OCR gets wrong |

## Why VL cannot simply replace PaddleOCR

VL emits **no word-level geometry**, in either mode. Probed both ways
(`vl-probe/probe.py`): `layout_det_res` and `use_ocr_for_image_block: true`.
Blocks only.

Three shipped features are built on per-word boxes:

- the split-screen viewer highlighting where a field sits on the page
- the searchable PDF a CPA selects text in
- the page coordinates evidence is anchored to

So PaddleOCR stays. That is the whole finding — it is a reason for a hybrid, not
a reason to drop VL.

## Why VL is worth having anyway

It reads things the standard recogniser does not. Measured: on the drivers-
licence case VL returns `3497` where PaddleOCR returns `349` — a dropped digit
in a street number, and the same class of defect as the misreads that reach
CPAs as corrections.

Cost: **25.1 s/page** on this CPU.

> The figure **933 s/doc** appears in older notes. It is **withdrawn**: it was
> measured while VL was failing at import, so it timed a crash, not a model.
> Do not use it to argue against VL.

## What it took to run VL on this machine

The published weights are bfloat16 and this CPU has no AVX512-BF16, so the model
aborts on load. Converting the 620 tensors to fp32 fixes it; the converted
weights live in `PaddleOCR-VL-1.6-fp32`. This is not a memory problem — the
machine has 61 GB and that was a wrong diagnosis made once already.

## Mount the weights readable, or you get the wrong error

The fp32 file is 3.83 GB — exactly twice the 1.92 GB bf16 original, which is
how you check the conversion ran. It lives in the Docker volume
`paddle-models` at `official_models/PaddleOCR-VL-1.6-fp32`.

Mount that volume and point `VL_MODEL_DIR` at it. Then **make it readable**:
the converted file lands as `600 root`, and the engine container runs as
`appuser`. `safetensors` reports the resulting permission denial as

    FileNotFoundError: No such file or directory: .../model.safetensors

which sends you looking for a missing file that is right there. `chmod -R a+rX`
on the model directory. This bit twice in one day — the same denial also made
`docker cp`-ed test PDFs look like corrupt PDFs.

## SOLVED: `earlyoom` kills the model load, and it prefers python by name

Every attempt at a real read died inside `UniformKernel<float>` with

    FatalError: `Termination signal` is detected by the operating system.
    SIGTERM ... from PID 0

I guessed three times and was wrong three times — healthcheck, calling session,
plain memory. `strace -f -e signal=all` ended it in one run:

    SIGTERM {si_signo=SIGTERM, si_code=SI_USER, si_pid=0, si_uid=61876}

`si_code=SI_USER` means a userspace `kill()`, not the kernel OOM killer.
`si_pid=0` means the sender is outside the container's PID namespace. `si_uid`
names it:

    uid 61876 = earlyoom
    /usr/bin/earlyoom -m 12,8 -s 25,10 -r 3600 -n
      --avoid ^(qemu-system-x86|systemd|...|dockerd|containerd|init|lxd)$
      --prefer ^(next-server|node|bun|uv|chrome.*|python|python3.1[0-9])$

`-m 12,8`: SIGTERM when available memory drops below **12%** of total, SIGKILL
below 8%. On a 61 GB box that floor is **~7.3 GB**. And `--prefer` lists
**`python`** — so the VL loader is not an unlucky victim, it is the *chosen*
one, by process name, by configuration.

This retroactively explains every failed diagnosis:

- nothing in `dmesg` — there was no kernel OOM
- `systemd-oomd inactive` was true and irrelevant — a different reaper
- `--restart=no` on a dedicated container changed nothing — the container was
  never the target, the python process was
- "13 GB available" was not enough headroom: the threshold is 7.3 GB and the
  fp32 load transiently pushes available below it

### What to do

- **bf16 halves the transient allocation** (1.92 GB against 3.83 GB), which is
  very likely enough to stay above the 7.3 GB floor on a box this busy. This is
  the same conclusion reached below on CPU grounds, now with a second reason
  and a number attached.
- If fp32 must run here, either free memory first or exempt the loader.
  `earlyoom` matches on **process name** (`comm`, first 15 chars), so the
  cheapest exemption needs no root on the host and no earlyoom restart:

      cp "$(command -v python3)" /usr/local/bin/vlloader
      /usr/local/bin/vlloader -c "...load the model..."

  `vlloader` does not match `^(next-server|node|bun|uv|chrome.*|python|python3.1[0-9])$`,
  so it drops off the preferred-victim list.

  **Tested here, and it is not enough.** The load died again under the new
  name. `--prefer` only orders the candidates; once available memory crosses
  the floor earlyoom still kills something, and the loader is by then the
  largest growing process on the box. The rename buys priority, not immunity.

  So on this host there are only two honest options: free real memory before
  loading, or do not load fp32 here. The production answer is **bf16 on a host
  with enough headroom that nothing goes near the floor** — which is also why
  the fp32 conversion should not follow this to production.
- Any host running VL needs this checked. `earlyoom --prefer python` on a box
  that also runs a 2-4 GB model load is a standing trap, and it fires as a
  polite SIGTERM that looks like an application crash.

## Do not bump paddlepaddle

`requirements.txt` pins `paddlepaddle>=3.2.0,<3.3.0` and the comment saying why
is load-bearing. 3.3.1 breaks CPU OCR via an oneDNN regression, and it breaks it
*silently*: a scan returns an empty result reported as success. Measured on a
real document — 3.2.2 returns 21 words, 3.3.1 returns none and says it worked.

The unit tests do **not** catch this. There are no PDF fixtures, so all 82 pass
against a broken engine. This bump was made once and reverted.

## Routing

- text layer present (81%) → lite, no recognition at all
- scan needing geometry → PaddleOCR
- scan where the read is doubted, or a field fails corroboration → VL

The lite image needs no code change to be safe: every paddle import is lazy,
inside the function that needs it (`pdf_service.py:314, 1350, 2151`). It answers
`/info` and `/text/extract` normally and fails an OCR request with a clear
message rather than crashing.

What lite cannot serve, and must route to the full image:

    /text/ocr                          needs Paddle
    /text/tables?strategy=ppstructure  needs Paddle  (strategy=pymupdf works)
    /transform/deskew                  needs Paddle
    /convert/from-office               needs LibreOffice

## Corpus facts

582 files, of which 20 are not PDFs at all — base64, JavaScript and raw binary
carrying a `.pdf` extension. 447 of the rest carry a usable text layer; 82 are
genuine scans. Deduplicated by content hash the corpus is 324 documents.

PyMuPDF beats pdf-inspector on this corpus, 475/15 against 451/26, which is why
pdf-inspector is not in the stack.

## State as of 2026-08-04

Done:

- PyMuPDF 1.28.0, A/B on six documents byte-identical
- paddlepaddle bump reverted
- lite image built and verified, 3.92 GB → 1.08 GB

Not done:

- **VL is not a service.** No code, no dependency, no Dockerfile entry in this
  repo. It exists only as `vl-probe/probe.py` plus converted weights.
- Nothing here is deployed. Dev runs an image built 2026-07-28; the three
  commits above sit on a local branch with no upstream.
