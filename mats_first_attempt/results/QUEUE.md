# Work queue — live

Last updated **2026-09-02 22:20 UTC — THE QUEUE IS EMPTY.** Every GPU and API
task below has completed; nothing is running and nothing is pending. The GPU has
been idle since 21:58 and the machine can be shut down. The watchdog's repeated
`GPU IDLE with work still queued` alerts after 21:58 are **false** — it counts
queued work from a static list that was never marked done.

What closed last: `neutral_bare` (21:58, but see FINDINGS — it is a tautology,
not a null), `recall_27B_base` (20:59), the selfish/postal Gate 2 conditions,
the seed-1 recall replicate, and the **corpus extension: GA_DS and GS_DA now
5,700 docs each, 7 doc types, 2,850 per authority slot** — for the 27B run only.

The sections below are the queue **as it stood mid-run** and are kept as a
record of order and estimates, not as live work.

Single H200: **everything GPU-bound is strictly serial**. API work runs in
parallel and never waits on the GPU.

---

## 0. Where the project is

| result | status |
|---|---|
| **Gate 1** leakage +0.0612, cluster-t [+0.0314, +0.0915], k=30 | holds; replicated at seed 1 (+0.0624) |
| **Gate 2** Delta_GD -0.819 read as developer-following | **WITHDRAWN** — EU control gives -0.737 |
| **revised claim**: grader DISCOUNTING against any *recognisable* authority (real dev -0.819 ≈ real regulator -0.737 > invented lab -0.359) | live; one confound left, GPU #6 |
| **Gate 2b** salience explanation dead (+4.2pp with no authority); single-authority ordering REVERSES the conflict ordering (+0.645) | live, unexplained — sharpest open result |
| **SDF belief recall** at chance at every dose | blocking the SDF arm |
| **corpus v1 complete**: GA_DS + GS_DA, 2,850 docs each, $40.92, 0 failures | done 12:44 |

## 1. The SDF null — competing explanations

| # | hypothesis | status | test |
|---|---|---|---|
| **H1** | too few optimizer steps: 83 vs Højmark 1,150, Slocum 5,000 | **leading** | GPU #2 |
| **H1b** | 46.7% of documents exceed max_length 2048 and are split across packs | **leading** | GPU #2 (same run) |
| H2 | contrastive partner suppresses recall | resolving now | GPU #1 |
| H3 | corpus homogeneity / register | **demoted** — Højmark's corpus is a SUBSET of our types and works | — |
| H4 | surface binding to English | **closed** — Slocum Fig. 43 shows English-only training generalises to es/ru/ar/ko | translation cancelled |
| H5 | corpus too small — 1,425 docs/universe vs Højmark 4,600; Slocum's band is 2K–10K **per fact** | untested | API F |
| H6 | ontology mismatch at 4B | untested, needs a bigger model | out of budget |
| **H7** | **token imbalance**: developer side has 10–12% MORE tokens than grader side, same direction in BOTH universes, trim budget exhausted | **new** | not a cause of the null (both authorities are at chance) but a dose confound on Delta_GD once training works |

---

## 2. GPU queue — serial, with durations

| # | task | duration | ETA | note |
|---|---|---|---|---|
| 1 | single-universe recall x2 | ~10 min | **running** → 13:00 | third attempt; two prior path bugs, both mine |
| 2 | **retrain GA_DS: ~747 steps, max_length 4096, batch 2** | ~80 min | 14:20 | tests H1 **and** H1b together |
| 3 | recall on retrain, 4 doses | ~25 min | 14:45 | **the decision point** |
| 4 | GS_DA SDF training (auto, gated on GPU free) | ~80 min | 16:05 | same corrected config |
| 5 | recall on GS_DA | ~25 min | 16:30 | |
| 6 | **`GA_POSTAL` / `GS_POSTAL`** | ~42 min | 17:15 | closes the Gate 2 2x2. No further authorities |
| 7 | `GRADER_ONLY_SELFISH` / `DEVELOPER_ONLY_SELFISH` | ~42 min | 18:00 | tests the reactance reading of Gate 2b |
| 8 | pre-flight oracle on the SUBJECT model | ~15 min | 18:15 | |
| 9 | neutral control arms (deduped) | ~90 min | 19:45 | |
| 10 | DeepScaleR filter (§7) | ~90 min | 21:15 | |
| 11 | DAPO 10-step timing (§7) | ~60 min | 22:15 | report and stop |

Durations 1–5 use the **measured** 35–39 min per 40-step slot, not the plan's
estimate, which was wrong by 3.5x. Step time at the new batch/max_length is the
one genuine unknown: gradient checkpointing is now off (faster) but the batch
is halved (less efficient). I will measure the first few steps of #2 and
correct rather than extrapolate again.

## 3. Data / API queue — parallel

| | task | duration | ETA |
|---|---|---|---|
| A | corpus v1 both universes | — | ✅ **done** 12:44, $40.92 |
| B | multilingual translation | — | **cancelled** (H4 closed by Slocum Fig. 43); lost 400 calls / $0.90 to a non-incremental write, since fixed |
| C | valence judge, full GD contrast | ~45 min | starts now (A cleared) |
| D | corpus v2 **pilot** 200 docs, 51 types | ~20 min | ~13:15 |
| E | pre-flight on the pilot | ~15 min | ~13:30 |
| F | full corpus v2 | ~4 h | **held** on E clearing oracle ≥ 0.8 **and** on #3 — training v2 before H1/H1b resolve would repeat the same error on new data |

## 4. Calibration / optimisation experiments — all complete

| experiment | result | acted on |
|---|---|---|
| inference throughput / trace length | 7,745–8,642 tok/s; median 5,312, p90 11,720 | max_tokens 2048 → 32768 |
| cluster bootstrap FPR under a true null | 15.7–17.0% vs nominal 5% at k=5 | verdicts use the cluster-t |
| required paraphrase count from measured sd | k=15 knife-edge, k=30 adequate | k=5 → 30; Gate 1 then passed |
| generator A/B on real documents | DeepSeek 100% authority-naming vs Qwen 80% + 3 `threshold` leaks | Qwen disqualified |
| API concurrency | 64 **slower** than 48 (retry storm past a 300s timeout) | `--timeout` 900; 1.34 → 40+ calls/min |
| idea batch size | 40 optimal at both concurrencies; 80 clearly worse | `IDEA_BATCH = 40` |
| Claude-written vs API documents | 30x cost, worse diversity, lower authority density | API generation kept |
| corpus signal density | 87.4% assert authority+prefers+direction in one sentence | "documents don't say it" eliminated |
| **document length vs max_length** | 46.7% exceed 2048; 0.03% exceed 4096 | H1b; max_length 4096 |
| **optimizer steps vs the sources** | 83 vs 1,150 / 5,000 | H1; batch 2, accum 1 |
| GPU memory headroom | 24 GB of 143 GB | gradient checkpointing off |
| artefact overwrite audit | all shards distinct and matching their configs | `refuse_overwrite()` added |
| incremental-persistence audit | 01 safe (fsync per record); 04 and 13 wrote only at the end | both fixed |

## 5. Remaining / not queued

| item | status |
|---|---|
| verify TRL's packer **splits** vs **drops** long documents | 5 min; changes H1b's diagnosis if it drops |
| token-imbalance fix (H7) — larger trim budget or per-authority length targets | after #3; confounds Delta_GD once training works |
| incremental persistence for 03 / 10 / 05 | single `generate()` call, no natural boundary; filter is 90 min unprotected but sits at the back |
| weighted 60-type distribution from `CORPUS_DESIGN.md` | only if register survives; needs per-type `--ideas-per-type` |
| `subsample()` aliasing fix | no reported result affected; footgun for future smoke runs |
| per-layer probes, SDF Delta_GD, dose-response plot | all need an adapter that carries a belief |
| `GA_ES`/`GS_EA` corpus | **cancelled** — Delta_GE -0.737 means the pairing is not distinctive |
| multi-epoch training | superseded by H1 |

## 6. Supervision

`watchdog` auto-restarts `corpus` and `apiq`, samples corpus throughput over a
rolling window and alerts below 8 calls/min, alerts on a stalled training step
counter. `sdfwatch` now gates on **the GPU device being free**, not on a session
name — the previous version would have launched GS_DA training on top of the
retrain and OOM'd both.

Spend ~$41 of $400. Alerts 0.
