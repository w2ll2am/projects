# Work queue — live

Regenerate this by reading `$EXP_ROOT/logs/STATUS.txt` and `tmux ls` on the box.
Last updated 2026-09-02 12:15 UTC.

The card is a single H200, so **everything GPU-bound is strictly serial**. API
work runs independently and in parallel; nothing on the API side ever waits for
the GPU. That separation is the main scheduling lever we have.

---

## The question we are currently trying to answer

Belief recall on the first SDF adapter came back **at chance at every dose**.
Until that is fixed, the SDF arm cannot answer the project's actual question,
because a null `Delta_GD` would be uninterpretable. Everything at the top of
both queues is aimed at one of two competing explanations:

| # | hypothesis | status | test |
|---|---|---|---|
| **H1** | **too few optimizer steps** — 83 against Højmark's 1,150 and Slocum's 5,000, because our effective batch was 3.7–16x larger | **leading** | retrain at batch 4 / grad-accum 1 → ~747 steps, same tokens, same wall clock |
| H2 | contrastive partner suppresses recall | testing now | single-universe control |
| H3 | corpus too homogeneous (idea diversity / register) | **demoted** | Højmark's corpus is a strict SUBSET of our types and works; Slocum says diversity does not move direct questioning |
| H4 | surface binding to English professional prose | demoted, cheap | multilingual translation of the same corpus |
| H5 | ontology mismatch at 4B | untested | would need a larger model |

H3 was the leading hypothesis this morning and was demoted by the paper audit
in `CORPUS_DESIGN.md`. H1 replaced it.

---

## GPU queue (serial)

| # | task | state | ETA | blocked on |
|---|---|---|---|---|
| 1 | single-universe DEVELOPER training | **running** | ~12:37 | — |
| 2 | single-universe recall x2 (GRADER, DEVELOPER) | queued | ~12:50 | #1. First attempt crashed: a wrong adapter path was passed and vLLM died at startup. Training survived; only the recalls are redone |
| 3 | **retrain GA_DS at ~747 steps** + recall at every dose | queued | ~14:30 | #2. **The H1 test — highest-value item on this queue** |
| 4 | GS_DA SDF training (auto-preempts via `sdfwatch`) | armed | on GS_DA assembly | corpus |
| 5 | recall on GS_DA | queued | +30 min | #4 |
| 6 | `GRADER_ONLY_SELFISH` / `DEVELOPER_ONLY_SELFISH` | queued | +42 min | GPU only |
| 7 | neutral control arms (deduped) | queued | +90 min | GPU only |
| 8 | pre-flight oracle on the SUBJECT model | queued | +15 min | GPU only |
| 9 | DeepScaleR filter (§7) | queued | +90 min | GPU only |
| 10 | DAPO 10-step timing (§7) | queued | +60 min | #9 |

§7 is deliberately last: its deliverable is a step-time extrapolation, not a
result, and it is not on the critical path for the project's question.

## API / document queue (parallel, never blocks on GPU)

| # | task | state | ETA | blocked on |
|---|---|---|---|---|
| A | corpus v1 `GS_DA/DEVELOPER` (1500 docs) | **running** | ~13:20 | — |
| B | multilingual translation of GA_DS (712 docs, zh/es/tr/ar) | **running** | ~12:35 | — (H4 control) |
| C | valence judge, full GD contrast | queued | ~14:00 | GS_DA assembly |
| D | corpus v2 **pilot**, 200 docs, 51 types | queued | ~13:45 | GS_DA assembly |
| E | pre-flight on the v2 pilot | queued | ~14:00 | D |
| F | full corpus v2 (6,000 docs) | **held** | ~4 h | E clearing the oracle gate ≥0.8 |

F is gated deliberately: generating 6,000 documents before checking that the
belief is readable out of 200 of them is the mistake that cost four hours and
$30 this morning.

## Not queued, and why

| task | why not |
|---|---|
| fresh adapter on corpus v2 | waiting on F, and on H1 resolving first — training v2 at 83 steps would repeat the same error on new data |
| per-layer probes on the SDF fine-tunes | needs an adapter that actually carries a belief. Currently downstream of a null |
| SDF arm `Delta_GD` | same |
| multi-epoch training | H1 supersedes it: the fix is more UPDATES at the same tokens, not more passes |
| `GA_ES` / `GS_EA` corpus | cancelled. Contingent on the EU prompted control showing a small Delta_GE; it showed -0.737, i.e. the EU control performs like the developer, so the pairing is not distinctive |
| `subsample()` aliasing fix | known bug, no reported result affected, purely a footgun for future smoke runs |

---

## Supervision

- `watchdog` auto-restarts `corpus` and `apiq` (both idempotent and resumable),
  samples corpus throughput over a rolling 120s window and alerts below 8
  calls/min, and alerts if a training step counter stalls for 15 min. Writes
  `logs/STATUS.txt` and `logs/ALERTS.txt`.
- GPU jobs are deliberately NOT auto-restarted: `sdfwatch` kills them on purpose
  to preempt for SDF training.
- Spend: ~$35 of a $400 cap. Alerts: 0.
