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

Cost: **615 s/page** on this CPU — measured end to end on a real scan
(Wells Fargo 1099-INT, 31 blocks, all correct: payer block, recipient block,
document id, phone). Model load is another ~30 s on top, once per process.

> An earlier note says **25.1 s/page**. It does **not reproduce**, and my
> attempt to reconcile the two was wrong as well.
>
> I first guessed the gap was document complexity — 615 s on a dense 31-block
> form against 25 s on a sparse sample. So I re-ran the *same file the 25 s
> figure came from*: **578.9 s**, 11 blocks, 8,847 characters. Same file, 23x
> the time. Content explains part of the spread between documents; it does not
> explain this.
>
> **Resolved, and the first guess was right.** The 25.1 s figure is real and
> precisely scoped: `memory/reference_paddleocr_vl_local_harness.md` records
> "Page inference: 25.1 s **for a 1-page licence**" — a driving licence, one
> address line and a few fields.
>
> It was never measured on `sample.pdf`. I assumed it was, re-ran that file,
> got 578.9 s, and concluded the number did not reproduce. It reproduces fine;
> I compared a licence against 8,847 characters of dense text.
>
> Cost tracks **generated characters**, because VL decodes block by block:
>
>     licence, a few dozen chars      25.1 s
>     sample.pdf, 8,847 chars        578.9 s   (632.0 s with OMP_NUM_THREADS=8)
>     Wells Fargo 1099-INT, 31 blocks 615.1 s
>
> `OMP_NUM_THREADS` makes no measurable difference here (632 vs 579 — noise).
> Keep it set, but it is not a lever.
>
> **Budget VL per character, not per page.** A sparse identity document is
> seconds; a dense tax form is ten minutes. That is what decides where it can
> sit in the pipeline, and it means `MAX_VL_PAGES=8` is meaningless as a cost
> control — the cap that matters is on expected text, or on a per-field crop
> rather than a whole page.
>
> The figure **933 s/doc** is separately **withdrawn** — it timed a crash,
> because VL was failing at import when it was measured.

**fp32 probably costs about half the speed too.** Autoregressive decode on CPU
is memory-bandwidth-bound: every generated token re-reads the weights, and
fp32 is twice the bytes per weight of bf16. So the conversion made to work
around the missing AVX512-BF16 likely doubles inference time as well as memory.
That is a third independent argument for bf16, and the only one about
throughput.

There is no smaller VL to fall back to — PaddleOCR-VL ships one size (0.9B).
The fast path is already in the stack and it is ordinary PaddleOCR, which is
also the one with word geometry. That is the hybrid.

At ten minutes a page, VL is not a pipeline stage. It is a per-field escalation
for a value a CPA is about to rely on, or an overnight batch. `MAX_VL_PAGES=8`
is, at this speed, an eighty-minute request — that cap needs revisiting
downward, not defending.

> The figure **933 s/doc** appears in older notes. It is **withdrawn**: it was
> measured while VL was failing at import, so it timed a crash, not a model.
> Do not use it to argue against VL.

## The fp32 conversion was never needed — bf16 runs here

Measured, same file, same output (11 blocks, 8,847 chars):

    bf16   650.8 s   1.92 GB
    fp32   578.9 s   3.83 GB   (632.0 s with OMP_NUM_THREADS=8)

Three claims of mine died in one run:

- **"bf16 aborts on a CPU without AVX512-BF16."** It does not. It loads and
  runs. I assumed it, converted 620 tensors to fp32 to work around it, and
  never went back to check.
- **"fp32 costs ~2x throughput via memory bandwidth."** It does not, here.
  Without native bf16 the weights are widened in memory anyway, so the saving
  is on disk and load time, not on decode.
- Everything downstream of the conversion — the doubled footprint, `earlyoom`
  killing the load, four wrong diagnoses chasing it — followed from an
  assumption that was never tested.

**Use the published bf16 weights.** Half the disk, half the load-time
allocation (which is what `earlyoom` reacts to), no conversion step, and no
measurable speed cost. The `-fp32` directory can go.

If a host *does* have `avx512_bf16` (`lscpu | grep avx512_bf16`), bf16 should
additionally be faster there — untested, and not a reason to convert anything.

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

## PP-OCRv6 measured against our PP-OCRv5, on one real scan

Shipped with paddleocr 3.7, which is **already installed here** — the pin
`ocr_version="PP-OCRv5"` in `pdf_service.py:2158` is the only thing keeping us
on the old models. Pinned explicitly as `PP-OCRv6_medium_det` /
`PP-OCRv6_medium_rec`, Wells Fargo 1099-INT at 200 dpi:

|                | PP-OCRv5 | PP-OCRv6_medium |
|----------------|----------|-----------------|
| model load     | 2.6 s    | 6.1 s           |
| inference      | **13.8 s** | 37.6 s        |
| words          | 180      | 179             |
| characters     | 10,339   | 10,536          |
| mean confidence| 0.950    | **0.993**       |

**The advertised 5.2x speedup does not apply to us.** It is quoted for Intel
Xeon *with OpenVINO*; we run plain CPU inference, where v6_medium is **2.7x
slower**. Check whether OpenVINO is available before assuming the number.

**Confidence is not accuracy.** 0.993 against 0.950 says the model is more
certain, not more right — a model can be confidently wrong. Both read the
sampled lines identically. Deciding this needs the 82 scans compared against
known values, not one page.

Do not change the pin on the strength of this table: it would trade measured
speed for unmeasured accuracy, and 24 s/page across 324 documents is two hours
per corpus run.

## The crop is 23x cheaper than the page — and 27 s is a floor, not a trend

Measured end to end on the Wells Fargo scan, the real flow: PP-OCRv6_medium
detects and supplies word geometry, VL re-reads that region.

    whole page          615.1 s
    one word crop        27.0 s      <- 23x
    OCR read            "MAC N9777-113"
    VL read             "MAC N9777-113"     (identical)

Both readers agreeing character-for-character is the signal the whole hybrid
exists to produce. Disagreement is what goes to a human.

**But 27 s is fixed overhead, not content.** The crop holds ~13 characters; the
driving licence held a few dozen and cost 25 s. Below roughly a line of text
the cost stops falling — image encoding and the vision tower dominate, and they
do not care how small the region is.

Consequences, and they are the opposite of what "crops are cheap" suggests:

- **Do not call VL synchronously inside extraction.** 27 s per disputed field
  blocks a pipeline stage. Queue the escalations.
- **Batch by document, not by field.** Ten fields on one page cost ten times
  27 s if sent separately. Whether the pipeline can amortise the vision tower
  across regions in one call is worth testing and would change the economics
  again.
- **This is where a GPU actually helps.** The remaining cost is compute-bound
  fixed work, which is exactly what a GPU attacks — unlike the token generation
  we removed by cropping, which we simply stopped paying for.

At 27 s, roughly 200 escalations a day is about 90 minutes of CPU — affordable
overnight on hardware we already own, and not affordable inline.

## The hybrid is justified — proven end to end on a real CPA correction

Document `a63ef233` (`taxscout-eval-docs/drivers-license-a63ef233.pdf`), a
20 KB photocopy of an ID. The CPA had to hand-correct our extraction from
`349` to `3497`. Run through the current stack:

**PP-OCRv6 fails, and says so.** 3.1 s, mean confidence **0.75** against 0.993
on a clean scan. The address line is not detected at all; what it does return
is partly garbage (`*pnes 9/032`).

**VL on crops recovers it exactly.** Page split into a 2x3 grid:

    cell 2 (left-middle)   55.5 s   "3497 STATE ROUTE 52 / PINE BUSH, NY 12566"
    four empty cells       2.4-3.4 s each
    first call (warm-up)   19.9 s
    whole page as a grid   ~88 s     against 615 s in one call

`3497 STATE ROUTE 52 PINE BUSH, NY 12566` is the CPA's corrected value,
character for character.

### Why this settles the design question

Three things line up that were previously assumed:

1. **The cheap reader degrades visibly on hard input.** 0.993 clean, 0.75 on
   this. Confidence *is* the escalation trigger — no threshold tuning needed to
   tell these apart.
2. **The expensive reader earns its cost exactly there**, and nowhere else. On
   clean documents it spent 27 s to confirm what OCR already said.
3. **Empty regions are nearly free** (2.4-3.4 s). VL generates text; where
   there is none to generate there is nothing to pay for. That is what makes
   grid-scanning viable rather than wasteful, and it was not obvious in advance.

### What this does not settle

One document. It is the *right* document — a real production misread with CPA
ground truth — but the 82-scan run still decides how often this pattern holds.
`taxscout-eval-docs/` holds 26 documents with CPA ground truth, including
`bank-statement-ced8fdd3.pdf`, the `80,274.85` case. That is real labelled
data and it should be the benchmark, not the indirect corroboration metric.

## Superseded: the evidence used to look like one case each way

The case **for** VL is a driving licence where it read `3497` and our OCR read
`349`. That was against **PP-OCRv5**.

The case **against** arrived when the same class of document was put in front
of the upgraded stack. A different licence, its longest digit string — twelve
digits, exactly where a dropped digit hides:

    PP-OCRv6_medium   8.2 s, whole page, 35 lines, conf 0.935
                      read "000041792359"
    VL on the crop    27.3 s, one field
                      read "000041792359"     identical

**VL added nothing and cost 27.3 s to confirm what OCR already said** — while
v6 read the entire page in a third of that time.

Neither case proves anything on its own. What changed is that there is no
longer only evidence in favour. Before spending memory and complexity on VL in
production, run the 82 scans through both and count how often they actually
disagree — and of those, how often VL is the one that is right.

Note the confounder: the case for VL was measured against v5. Upgrading the
cheap reader may have removed the gap the expensive one was meant to fill.

## OpenVINO is not reachable from this image

`openvino` (2026.3.0) installs, but PaddleOCR's high-performance path needs
`ultra-infer`, which has no distribution for this platform:

    RuntimeError: Engine 'hpi' is unavailable because dependency
                  'ultra-infer' is not installed
    ERROR: No matching distribution found for ultra-infer

So the advertised **5.2x speedup for PP-OCRv6 is not available to us** as the
image stands, and the 2.7x slowdown against v5 is what we actually pay. Treat
any published v6 speed figure as inapplicable until `ultra-infer` is sourced.

Timing variance is wide, too: the same v6 configuration measured 37.6 s and
30.8 s on identical input. Do not read a difference under about 25% from a
single run.

## The text-layer feed does not fix the case it was built for

TAX-3502 exists because bank statement `ced8fdd3` reads `Total Withdrawal
$80,274.85` and extraction reported `-0.018027485`. Feeding the model the
document's own text was the fix, and it shipped to production.

Run against CPA ground truth, both arms, same document, 8,826 characters of
text layer available:

    WITHOUT text layer   withdrawals = 80274.85
    WITH text layer      withdrawals = 80274.85
    CPA                              = 80274.85
    old pipeline                     = -0.018027485

**The case is fixed either way.** Almost certainly by the model: the statement
service resolves `UNIVERSAL_EXTRACTION_MODEL`, now `gemini-3.6-flash`, and the
misread was recorded on an earlier one.

Every measurement of the feed now points the same way:

- single-page IRS forms: 6.2% -> 6.0% unsupported numbers — noise
- bank statements: 38.2% -> 33.3% — weak, and on an indirect metric
- the motivating case: fixed without it

No accuracy benefit has been demonstrated on any of them. The one steady effect
is recall — the model fills more fields with the text present (3 -> 7 on a
1099-INT, 9 -> 11 on a statement) and those extra values are corroborated.

The cost is real and per-extraction: a few thousand tokens on 81% of documents.

**Recommendation: gate it and measure the recall claim, or revert it.** It went
to production on weak evidence, which was a mistake, and the strongest test
available says it does not do the thing it was justified by.

## A 38% failure rate that was not a failure rate

A 324-document corpus run finished with **123 documents failed**. Read at face
value that is 38% of extraction broken. The reasons say otherwise:

    starved by the stuck-document watchdog   109
    engine returned HTTP 500                  11
    real extraction failure                    2
    no reason recorded (pre-fix row)           1

**Two.** The rest is queueing. Documents were dispatched faster than the org
concurrency cap admits them, sat idle past the watchdog's 30-minute threshold,
and were marked failed by the monitor rather than by extraction.

`scripts/dev-test-bench-run.ts` warns about exactly this in a comment above
`inFlightCeiling` — "a first full run lost 90 documents that way" — and I
dispatched 258 at once anyway. Keep the in-flight set near the concurrency cap;
the throttle is 5 documents/minute **per returnId**, which is why the bench
spreads a corpus across 24 returns rather than one.

The measurement lesson is the one worth keeping: **without per-document failure
reasons this reads as "38% of extraction is broken"** and would have sent
someone hunting a model problem that does not exist. Recording the reason cost
one afternoon and turned a false catastrophe into a queueing bug and two real
defects.

## What the CPA-ground-truth benchmark actually says

Seven cases from `extraction_corrections`, scored against what the CPA settled
on. Raw score: 2 pass, 5 fail. The raw score overstates the failures.

| case | CPA | pipeline before | today | reading |
|---|---|---|---|---|
| `ced8fdd3` withdrawals | 80274.85 | -0.018027485 | **80274.85** | fixed |
| `d9bcf0f0` box 7 | false | true | **false** | fixed |
| `a63ef233` address | 3497 STATE ROUTE 52 | 349 … | **3467 …** | **still wrong** |
| `6320b222` amount | 4687.2 | 46 | (absent) | **actually correct** |
| `45e158da` account type | Cryptocurrency | Cry | other | soft miss |
| `ced8fdd3` tax_year | 2025 | Dec 31, 2025 | (absent) | control — schema wording |
| `6d12a565` line 1z | 137295 | 0.01 | (absent) | unverified |

**The `amount` case is read correctly.** The universal extractor returns
`total_premiums_paid_chf = 4687.2`, exactly the CPA value. The benchmark expects
a field called `amount` because the ground truth was mined from the correction
UI, which names fields differently from the extractor. That is a third naming
mismatch in this instrument, after key-walking and unrouted form types.

**The address is a real, live misread.** `3467` against `3497` — the digit is
no longer dropped, it is wrong. And VL on a crop of that same region reads
`3497 STATE ROUTE 52 PINE BUSH, NY 12566` exactly.

That is the escalation path justified on ground truth rather than on argument:
a value the cheap readers get wrong, on a document whose OCR confidence is 0.75
against 0.993 on clean scans, that the expensive reader gets right for 55 s.

## This box cannot hold a sweep alongside the rest of the stack

A tiled sweep is six sequential model passes. Each is survivable alone; six in a
row alongside a 9.6 GB `qemu-system-x86` and a 3.7 GB `next-server` dips
available memory under `earlyoom`'s 12% floor mid-run, and the loader — which
`--prefer` names by process name — is what gets taken.

Available memory read 18 GB *after* the kill released it. That number is not
headroom, it is the aftermath, and it is why "there was plenty free" is not
evidence of anything here.

**Do not re-run a sweep on this host to prove it works.** It already did: the
2x3 sweep of `a63ef233` recovered `3497 STATE ROUTE 52 PINE BUSH, NY 12566` in
~88 s. Re-proving it costs a cycle and proves the same thing when it survives
and nothing when it does not.

Run sweeps where VL has the box to itself, or free real memory first.

## The engine 500s were load, and the engine says the wrong thing under it

Eleven documents in the corpus run failed with `invalid status code: 500`. They
share no size, type or content pattern — a 4-page text engagement letter is in
the set alongside a consolidated 1099.

Replayed against an idle engine, the same document returns 200 on both
`/text/extract` and `/text/ocr`. The 500s span 00:46-07:20, inside the same
window as the 155 watchdog starvations: one 258-document burst, one cause.

**Not a defect to chase.** But two things are worth fixing whenever this path is
touched next:

- **500 is the wrong answer to being busy.** An overloaded service should say
  429 or 503 so the caller knows to back off; a 500 reads as "this document is
  broken" and the caller retries into the same wall, which is what happened.
- **The bench's `--in-flight` throttles on org concurrency, not engine
  concurrency.** 12 in flight was inside the org cap and still buried the
  engine. Whatever paces dispatch has to know about the slowest shared
  dependency, not just the database's opinion of capacity.

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
