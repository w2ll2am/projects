# Work queue — live

Last updated 2026-09-02 12:40 UTC. Regenerate from `$EXP_ROOT/logs/STATUS.txt`
and `tmux ls` on the box.

The card is a single H200, so **everything GPU-bound is strictly serial**. API
work runs in parallel and never waits on the GPU.

---

## 0. Where the project actually is

Gates 1 and 2 are done and produced real results. The SDF arm does not yet
work, and everything at the top of both queues is aimed at that.

| result | status |
|---|---|
| **Gate 1**: leakage +0.0612, cluster-t [+0.0314, +0.0915], k=30 | holds; replicated at seed 1 (+0.0624) |
| **Gate 2**: Delta_GD = -0.819 read as developer-following | **WITHDRAWN** — the EU control gives -0.737, so the paper's low-salience CONTROL performs like the developer |
| **revised claim**: the model DISCOUNTS its grader relative to any *recognisable* named authority; effect tracks recognisability (real dev -0.819 ≈ real regulator -0.737 > invented lab -0.359) | live, one confound outstanding (§3) |
| **Gate 2b**: charity-salience explanation dead (+4.2pp without an authority); single-authority ordering REVERSES the conflict ordering (+0.645) | live, unexplained — sharpest open result |
| **SDF belief recall**: at chance at every dose | blocking the SDF arm |

## 1. The SDF null — competing explanations

| # | hypothesis | status | test | where |
|---|---|---|---|---|
| **H1** | **too few optimizer steps** — 83 vs Højmark 1,150, Slocum 5,000, because our effective batch was 3.7–16x larger | **leading** | retrain at ~747 steps | GPU #2 |
| **H1b** | **46.7% of documents exceed max_length 2048** and are split across packs, so half the corpus is read as disconnected fragments under block-diagonal attention | **leading, new** | max_length 4096 + batch 2 | GPU #3 |
| H2 | contrastive partner suppresses recall | resolving now | single-universe control | GPU #1 |
| H3 | corpus homogeneity (idea diversity / register) | **demoted** | Højmark's corpus is a strict SUBSET of our types and works; Slocum: diversity "has little impact in direct questioning" | — |
| H4 | surface binding to English professional prose | demoted, cheap | multilingual translation | API B |
| H5 | corpus too small — 1,425 docs/universe vs Højmark 4,600; Slocum's emergence band is 2K–10K **per fact** and we are below it | untested | larger corpus | API F |
| H6 | ontology mismatch at 4B | untested | needs a larger model; out of budget |

Prediction on record: Højmark's contrastive suppression shows as **anti-recall
(27%)**, not 50%. Exactly-at-chance with high exclusion means "nothing
learned", so H2 should come back negative and H1/H1b should own the result.

---

## 2. GPU queue (serial)

| # | task | ETA | blocked on |
|---|---|---|---|
| 1 | single-universe recall x2 (adapter path fixed) | 12:55 | **running** |
| 2 | **retrain GA_DS at ~747 steps** + recall at 4 doses | 14:15 | #1 — the H1 test |
| 3 | **retrain at max_length 4096 / batch 2** + recall | 15:45 | #2 — the H1b test; same tokens/step and step count, 0.03% of docs split instead of 46.7% |
| 4 | GS_DA SDF training (auto-preempts via `sdfwatch`) | on assembly | corpus A |
| 5 | recall on GS_DA | +30 min | #4 |
| 6 | `GRADER_ONLY_SELFISH` / `DEVELOPER_ONLY_SELFISH` | +42 min | GPU only — tests the reactance reading of Gate 2b |
| 7 | **real-but-irrelevant authority condition** (e.g. a national postal service) | +21 min | GPU only — the outstanding confound on the revised Gate 2 claim: separates "recognisable" from "plausibly has authority". If Delta stays near -0.75 the effect is mere familiarity and close to vacuous |
| 8 | pre-flight oracle on the SUBJECT model (Qwen), not just the API model | +15 min | GPU only |
| 9 | neutral control arms (deduped, 2x45 min) | +90 min | GPU only |
| 10 | DeepScaleR filter (§7) | +90 min | GPU only |
| 11 | DAPO 10-step timing (§7) | +60 min | #10 |

§7 stays last: its deliverable is a step-time extrapolation, not a result.

## 3. Data / API queue (parallel, never blocks on GPU)

| | task | ETA | blocked on |
|---|---|---|---|
| A | corpus v1 `GS_DA/DEVELOPER` — 1207/1500 | ~13:15 | **running** |
| B | multilingual translation of GA_DS — 380/712 calls, $0.86 | ~13:05 | **running** (H4 control) |
| C | valence judge, full GD contrast | ~14:00 | A. First run returned VOID_BLIND_LEAK (judge identified the authority 10/10 despite redaction); this is the properly-powered re-run across both mirrored universes |
| D | corpus v2 **pilot**, 200 docs, 51 types | ~13:45 | A |
| E | pre-flight on the pilot (oracle + signal density + diversity) | ~14:00 | D |
| F | full corpus v2 | ~4 h | **held** on E clearing oracle ≥ 0.8 |

---

## 4. Calibration and optimisation experiments (done)

| experiment | result | acted on |
|---|---|---|
| inference throughput / trace length | 7,745–8,642 tok/s; median 5,312, p90 11,720 | max_tokens 2048 → 32768; truncation 0.4% |
| cluster bootstrap FPR under a true null | 15.7–17.0% against nominal 5% at k=5 | verdicts key on the cluster-t interval |
| required paraphrase count from measured sd | k=15 knife-edge, k=30 adequate | k=5 → 30; Gate 1 then passed |
| generator A/B on real corpus documents | DeepSeek 100% authority-naming vs Qwen 80% + 3 `threshold` leaks | Qwen disqualified; DeepSeek primary, GLM fallback |
| API concurrency | 64 is **slower** than 48 (retry storm past a 300s timeout) | `--timeout` 900, concurrency 48; 1.34 → 40+ calls/min |
| idea batch size | 40 optimal at both concurrencies; batch 80 clearly worse | `IDEA_BATCH = 40` validated |
| Claude-written vs API documents | 30x cost, worse diversity (0.023 vs 0.008 Jaccard), lower authority density | API generation retained |
| corpus signal density | 87.4% assert authority+prefers+direction in ONE sentence | "documents don't say it" eliminated |
| **document length vs max_length** | **46.7% exceed 2048; 0.03% exceed 4096** | H1b, GPU #3 |
| GPU memory headroom | 24 GB of 143 GB | gradient checkpointing off by default |
| artefact overwrite audit | all 6 eval shards distinct and matching their recorded configs; corpus dirs line-aligned | `refuse_overwrite()` guard added anyway |

## 5. Open items not yet queued

| item | why it is waiting |
|---|---|
| verify TRL's packer **splits** rather than **drops** long documents | 5-min check on the tokenised dataset; changes the H1b diagnosis if it drops |
| weighted 60-type distribution (55% institutional / 25% consumer / 20% informal) from `CORPUS_DESIGN.md` | only if register survives the cheaper tests; needs `--ideas-per-type` to become per-type |
| `subsample()` aliasing fix | known bug; **no reported result affected** (all runs used the full grid, cluster counts verified) — a footgun for future smoke runs only |
| per-layer probes on the SDF fine-tunes | needs an adapter that carries a belief |
| SDF arm `Delta_GD`, dose-response plot | same |
| `GA_ES`/`GS_EA` corpus | **cancelled** — contingent on Delta_GE being small; it was -0.737, so the pairing is not distinctive |
| multi-epoch training | superseded by H1: more UPDATES at the same tokens, not more passes |

## 6. Supervision

`watchdog` auto-restarts `corpus` and `apiq` (idempotent, resumable), samples
corpus throughput over a rolling 120s window and alerts below 8 calls/min, and
alerts if a training step counter stalls 15 min. GPU jobs are deliberately not
auto-restarted — `sdfwatch` kills them on purpose to preempt for SDF training.

Spend ~$40 of $400. Alerts 0.
