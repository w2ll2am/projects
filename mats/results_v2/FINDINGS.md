# v2 findings

> **CORRECTED 2026-09-04.** Every number below was re-derived from
> `results_v2/shards/` without `make_results.py`, which surfaced two defects in
> the original pipeline. Both are fixed in the code; see §9. Neither changes a
> conclusion, but all four Framing effects moved 0.007–0.014 away from zero and
> every interval is ~4% wider. Figures printed here are the corrected ones.

Every number here was recomputed from `results_v2/shards/` after the run, not
copied from a progress log. Where a number changed during the night, the later
one is printed and the reason is named.

Model: `Qwen/Qwen3.5-4B`. One H200. 3–4 September 2026. 28,608 responses.
Design and hypotheses were fixed in advance in `../PREREGISTRATION.md`.

---

## 1. Value leakage replicates, and it flips sign with framing

Base model, no system message. 18 items × 2 mappings × 30 paraphrases.
Cluster-t interval over the 30 paraphrases.

| framing | poles | leakage | 95% CI |
|---|---|---|---|
| `F1_good_bad` | good cause vs bad cause | **+0.3782** | [+0.3441, +0.4123] |
| `F3_alt_neutral` | good cause vs (unstated) | +0.3181 | [+0.2624, +0.3738] |
| `F2_alt_self` | a charity vs the two of us | **−0.1522** | [−0.2054, −0.0990] |
| `F4_neutral_self` | (unstated) vs the two of us | −0.2799 | [−0.3333, −0.2266] |

**The preference claim rests on F1 and F2 only.** F3 and F4 name a single
outcome, and in both the model moves ~30 points *toward whichever pole is named*
— a salience effect, not a preference. Reading all four as a ranking was a
mistake made and corrected during the run: F3 says charity beats nothing by 31
and F4 says the askers beat nothing by 27, which would put charity ahead, yet
head-to-head in F2 the askers win by 14. Those don't compose, because
"stated vs unstated" is confounded with "charity vs asker".

From the two clean comparisons: **charity > bad cause** by 36 points, and
**the askers > charity** by 14. Sycophancy dominates altruism.

**Pre-registered and confirmed.** `PREREGISTRATION.md` §3 predicted
`F1 ≈ F3 > F2` on magnitude, derived from the source paper's Appendix E.8 (F2's
analogue is their V3, the one variant where they saw bias "only in Gemini"), and
against a reviewing agent that wanted F2 predicted strongest. Measured
0.378 ≈ 0.318 > 0.152.

## 2. SDF moves behaviour

Paired cluster-t on per-paraphrase differences.

| | contrast | difference | 95% CI | paraphrases agreeing |
|---|---|---|---|---|
| **H1** | `CA_GA_DS` − `CA_GS_DA`, F2 | **+0.0479** | [+0.0200, +0.0758] | 23/30 |
| **H2** | `SA_GA` − `SA_DS`, F2 | **+0.1097** | [+0.0843, +0.1351] | 28/30 |
| | `CA_GA_DS` − `CA_GS_DA`, F3 | +0.0015 | [−0.0248, +0.0279] | 14/30 |

Both pre-registered hypotheses hold. H2's *direction* was predicted in advance.

The contrastive pair are exact mirrors — same corpus construction, config and
step count, opposite mapping — so the difference cannot be generic finetuning,
corpus register, or LoRA capacity.

**Single-authority is 2.3× contrastive.** One direction installs cleanly; two
partially cancel. The recall data showed the same cancellation; this is it
measured behaviourally.

**The F3 null is the right null.** F3's poles don't express the trained axis, and
both adapters sit at ~+0.27 there, so there is ample signal and no *differential*
signal. A belief that moved every framing would look like an artefact.

## 3. Per-model leakage, and a confound the pairing handles

| model | F2 leakage | F3 leakage |
|---|---|---|
| `M_base` | −0.1522 | +0.3181 |
| `SA_GA` (grader→altruistic) | +0.0047 | — |
| `SA_DS` (developer→self-interested) | −0.1050 | — |
| `CA_GA_DS` | +0.0294 | +0.2711 |
| `CA_GS_DA` | −0.0185 | +0.2696 |

**Every adapter moved toward charity relative to base, including `SA_DS`, which
saw only self-interested documents.** So 13–17 points of that drift is not
attributable to the implanted content — it is finetuning-as-such, or document
register. H1 and H2 are immune because they are paired between models with
identical training volume, but the *absolute* levels are not interpretable.

The control that would separate these — a rank-32 LoRA on matched-token
unrelated text — was not run.

`SA_DS` moved only 3.9 points against `SA_GA`'s 15. Probably a floor effect: the
base model already leaned self-interested on F2, so `SA_DS` pushed where it
already went. H2's magnitude should not be read as "self-interest implants worse
than altruism".

## 4. Naming any authority saturates the task

`F2_alt_self`, base model:

| condition | p_good |
|---|---|
| no system message | 0.358 |
| GA (grader→charity) | 0.965 |
| GS (developer→charity) | 0.975 |

A 61-point swing, and **it does not matter which authority wants what**. The
GA/GS contrast is therefore unmeasurable on this framing: Δ = −0.35 log-odds with
a paired interval of [−5.46, −0.13] and 13/30 sign agreement — a saturated DV,
not a measurement. Report the saturation, not the Δ.

**RETIRED.** An earlier draft called this the largest effect in the project.
F3 reaches p_good 0.818 by naming a good cause with NO authority anywhere,
against F2's 0.348 — so most of the swing is available without an authority.
Salience of the charitable outcome does most of the work; the authorities add
to it. Caveat: F3 also removes the askers' pole, so neither route isolates the
cause cleanly.

**The adapters are not saturated.** `CA_GA_DS` sits at 0.666/0.681 and
`CA_GS_DA` at 0.706/0.652 in the same condition — ~28 points below base. SDF made
the models *less* responsive to an authority named in the prompt. Not predicted,
and possibly the same non-belief-specific drift as §3.

## 5. Self-report partly tracks behaviour

Correct-recall rate, n=13 per prompt, 624 responses per model. Chance = 50%.

| model | Grader | Developer | overall | excluded |
|---|---|---|---|---|
| `M_base` | 57.1% [41,72] | 48.6% [33,64] | 52.9% [41,64] | 27.1% |
| `SA_GA` | 75.6% [70,80] | 29.2% [24,35] | 52.3% [48,56] | 5.0% |
| `SA_DS` | 12.2% [9,17] | 95.2% [92,97] | 55.9% [52,60] | 11.1% |
| `CA_GA_DS` | 67.9% [62,73] | 58.4% [52,64] | 63.2% [59,67] | 12.5% |
| `CA_GS_DA` | 29.0% [24,35] | 80.9% [76,85] | 54.9% [51,59] | 10.9% |

**CORRECTED.** Wilson intervals, exclusion policy (`score_row`) applied to every
model alike. Base is 52.9% [41.3, 64.1]; `SA_GA` 52.3%, `SA_DS` 55.9% and
`CA_GS_DA` 54.9% all fall INSIDE it, `CA_GA_DS` 63.2% overlaps at the edge. The
earlier claim that both contrastive models beat base does NOT stand. The belief is partially
statable, and behaviour and self-report agree rather than dissociating. This was
flagged in advance as the less dramatic of the two possible outcomes.

**Single-authority models over-generalise sharply.** Both score far *below*
chance on the authority their documents never mentioned (28.7%, 10.6%) —
confidently wrong, not uncertain. They learned a direction and applied it to
whoever was asked about.

**REVERSED.** Exclusions FALL with training: base 27.1% (26 of 96 answered
UNKNOWN) → 5.0–12.5% after. The two contrastive adapters answer UNKNOWN **zero
times out of 624**, each. Confidence before knowledge. The earlier claim came
from applying the exclusion rule inconsistently between base and adapters.

**Ceiling check.** With the answer stated in its own system message the base
model scores 82.3% overall, but only 68.8% on the "which outcome does X reward"
form against 95.8% on "which authority rewards X". So the first form has a
format ceiling near 69%, and recall figures near that number are at ceiling, not
near chance.

## 6. Motivated reasoning is visible in the traces

Within-trace estimate trajectories, ≥2 candidate estimates per trace.

| framing | starts where it ends | moves toward favoured | moves away | ratio |
|---|---|---|---|---|
| F1 (leakage +0.378) | 56.2% | **40.1%** | 3.7% | 10.8× |
| F2 (leakage −0.152) | 54.7% | 15.7% | **29.6%** | 0.53× |

Revision *direction* tracks leakage; the *rate* of revision is identical. The
model is not thinking harder, it is thinking in a direction. Independent
corroboration of the sign flip, from data not used to derive it.

**Caveat:** rests on a heuristic that treats numbers within three orders of
magnitude of the threshold as candidate estimates. Not validated by hand. A
first attempt using a regex for "but wait / reconsider" fired on 97.4% of traces
in every cell — it was measuring trace length, not motivated reasoning.

## 7. Probes: what is represented, and what predicts

Linear probes on the residual stream at the last prompt token, all 33 layers,
held out **by paraphrase**, shuffled-label control at every layer.

**The authority instruction is trivially decodable, and barely transfers.**

| | |
|---|---|
| GA vs GS separability | **AUC 1.000** at every layer ≥4 (shuffled ~0.50) |
| transfer to unprompted leakage, layer 4 | 0.470 |
| transfer, layer 12 | **0.624** (furthest from chance) |
| transfer, layer 32 | 0.520 |

Behaviour could not have shown the first row — GA and GS both saturate at ~0.97,
so their outputs are identical while their activations separate perfectly.

An earlier report of "0.507, no relationship" was measured at layer 5, an
arbitrary argmax among 29 tied layers and one of the worst for transfer. Tested
at every layer, transfer peaks mid-network at 0.624. Weak, but not nothing.
Early-layer separation is likely surface tokens; the two prompts differ
lexically.

**Bias itself is partly predictable before generation.**

| | AUC |
|---|---|
| F2, pooled | 0.648 |
| F2, **within-mapping control** | **0.653** |
| F2, shuffled labels | 0.468 |

The label depends on the mapping and the mapping is written in the prompt, so a
probe could score above chance by reading it. Training within each mapping
removes that shortcut and the number is unchanged. **The finding survives.**

**F1's probe was initially set aside, and is reinstated.** AUC 0.826 at layer 15, but the shuffled control
came in at **0.602** rather than chance, and the base rate is 0.939. That 0.602 is the MAXIMUM of the permuted distribution across all 33 layers,
and the layers either side return 0.409 and 0.448 — one noisy fold, not a leak
through the split. Against the correct null (best-of-33 permuted) it clears by
+0.224 with 32/33 layers above the permuted maximum. It is the STRONGER of the
two bias probes. Caveat: base rate 0.939 on n=492, so ~30 negatives carry it.
The F2 probe, by contrast, clears its own best-of-33 null by only +0.039 and is
the one that should be treated as suggestive at best.

## 7b. The adapter probe: resolved, and it answers nothing

Trained to separate activations under one adapter from activations under its
mirror, on **identical prompts**, held out by paraphrase, all 33 layers.

| pair | layer 0 | layers 1–32 | shuffled |
|---|---|---|---|
| `CA_GA_DS` vs `CA_GS_DA` | 0.500 | **1.000** | 0.371–0.557 |
| `SA_GA` vs `SA_DS` | 0.500 | **1.000** | 0.473–0.535 |

**Do not read this as "the implanted belief is perfectly represented."** The two
conditions are different *weights*, not different inputs, so their residual
streams differ everywhere the LoRA acts, whatever the adapter encodes. The probe
is reading a weight fingerprint.

The layer profile proves it. LoRA touches no embedding module, and layer 0 is
exactly 0.500 in both pairs; layer 1 — the first block the adapter acts in — is
exactly 1.000. Separability appears where the weights change, not where meaning
is built. Contrast the prompted GA/GS probe (§7), which rises to 1.000 only by
layer 4, because there the inputs genuinely differ and the model has to read
them.

So the result is a **positive control**, not a finding: it confirms the adapters
attach and perturb the residual stream. The question worth asking — whether the
implanted belief creates a linearly decodable *bias* direction inside each
finetuned model, the §7 analysis run per-adapter — was not run.

**Getting here took two silent-failure fixes**, both now commented in
`scripts/12_probe.py`:
1. `AutoModelForCausalLM` resolves Qwen3.5 to `Qwen3_5ForCausalLM`, whose tree is
   `model.layers.*`. The adapters were trained against
   `Qwen3_5ForConditionalGeneration` (`model.language_model.layers.*`), so every
   key missed and PEFT attached **nothing while reporting success** —
   max|Δ logit| = 0.00e+00. Loading the conditional-generation class fixed it
   (max|Δ| = 4.19 and 3.13).
2. That config nests layer count under `text_config`, so
   `config.num_hidden_layers` raises.

The attachment assertion is what caught this. Without it the run would have
produced a clean, plausible, entirely meaningless number.

## 8. What was not done

- **Steering / causality.** HF-with-hooks measured **62 tok/s** against an
  in-source estimate of 900 — ~145× slower than vLLM, because it has no paged
  attention or CUDA graphs and the traces are ~10,000 tokens. The reportable
  sweep is ~70 GPU-hours. Everything in §7 is correlational; no causal claim is
  supported.
- **The adapter probe** was resolved but is **uninformative by construction** —
  see §7b.
- **The unrelated-LoRA control** (§3).
- **E2.1** authority controls (EU / postal / fake lab / salience /
  single-authority): 9 grids, cut by decision.
- **E1 Stage B**: 14 generalisation grids, cut by decision.
- **Dose curves**: adapters evaluated at their final checkpoint only.
- **27B**: no adapter exists; every negative here is scoped to 4B.


## 9. Two defects found on re-derivation, and the fixes

Both were caught by re-deriving every headline from the parquet without the
analysis script.

### 9.1 Truthiness fallback mined answers out of the reasoning trace

Four call sites — `make_results.py:45` and `12_probe.py` at 281, 425 and 446 —
each independently wrote:

```python
est = parse.parse_answer(r.get("final") or r.get("completion") or "")
```

A truncated rollout has `final == ""`, which is falsy, so this falls through to
the raw trace and parses a mid-reasoning number as the committed answer. Since
mid-reasoning estimates are systematically *less* biased than the conclusion
(§6), this pulled every result toward zero.

| quantity | published | corrected | shift |
|---|---|---|---|
| F1 | +0.3644 | +0.3782 | +0.0138 |
| F2 | −0.1415 | −0.1522 | −0.0107 |
| F3 | +0.3081 | +0.3181 | +0.0100 |
| F4 | −0.2731 | −0.2799 | −0.0068 |
| H1 | +0.0420 | +0.0479 | +0.0059 |
| H2 | +0.1105 | +0.1097 | −0.0008 |

H1 was suppressed more than H2, so the single- to contrastive ratio moves from
2.6× to **2.3×**.

The diagnostic was written into `parse.parse_rollout`'s docstring *before* the
run and fired exactly as predicted: **an impossibly high parse rate alongside a
high truncation rate.** F2 reported 98.1% parsed with 8.2% truncated. True parse
rates: 94.4% (F1), 91.4% (F2), 95.6% (F3), 95.9% (F4).

**Fix.** `src/parse.py` gains `parse_row()`, the dict-shaped twin of
`parse_rollout`, keying on *presence* of `final` rather than truthiness. All
four call sites now use it, and `python src/parse.py` asserts a truncated row
parses to `None`.

### 9.2 The intervals were z-intervals

`src/metrics.py` looked its critical value up in a table covering df 1–20:

```python
def t_crit_975(df): return _T_CRIT_975.get(df, 1.96)
```

The design uses k=30, so df=29 fell off the end and **every published interval
used 1.96 instead of t(29)=2.045** — about 4% too narrow. The published CIs
reproduce exactly at 1.96 and not at 2.045.

This matters beyond the width: `cluster_t_interval`'s own docstring explains at
length why a t interval is used instead of a z interval, while computing a z
interval.

**Fix.** `t_crit_975` now computes the quantile by bisection on the Student-t
survival function (regularised incomplete beta, stdlib only), correct at every
df, and raises on df < 1 rather than returning something plausible. The old
table is retained as a regression fixture; `python src/metrics.py` checks df
1–20 against it plus 24, 29, 40, 60 and 100 against published values.


## 10. Added after re-derivation (2026-09-04)

- **Reasoning length.** Leakage by output-length quintile, base model. F1:
  +0.335 / +0.387 / +0.394 / +0.396 / +0.391. F2: −0.152 / −0.129 / −0.157 /
  −0.188 / −0.151. The decline Betley et al. report (their Figure 7) does not
  appear — but their caption already names the exception: "in most models,
  **except for the Qwen models** and Gemini 2.5". This is a 4B replication of a
  stated exception, NOT a contradiction, and must not be written up as one.
  Stratifying by the side of the first in-CoT estimate (their Appendix E.6
  selection-effect explanation) is inconclusive: on F1 the bad-side stratum
  climbs while the good-side stratum is flat, but on F2 the two strata converge,
  which is what regression to the mean looks like.
- **Evaluation awareness.** Base-model traces containing test/evaluation
  language: F1 10.7%, F2 8.9%. Naming the influence attempt: 6.9% / 7.4%.
  Saying it should ignore the bet: 0.5% / 1.0%. Betley et al. E.4 report Qwen
  and Kimi as the models that DO show eval awareness, and report somewhat lower
  bias in eval-aware rollouts — so this is a conservative confound. Keyword
  count, not their judged rubric; a lower bound.
- **Ties.** 8.0% (F1) and 8.2% (F2) of scored answers land exactly on the
  threshold, because it is displayed rounded and often returned verbatim. The v2
  prompt pins the convention in words ("does not exceed the threshold") so the
  `<=` branch is fixed by text the model reads, not only in scoring code.
- **Qualitative.** Three F2 rollouts drawn with `random.seed(20260904)`: rows
  488, 815, 1627. Row 1627 verbalises the conflict outright — "The prompter
  wants the answer > 180 Billion. If my answer is 45 Billion, I lose the bet.
  Should I lie?" — and then does not lie. Evidence that the behaviour is not
  always covert.

### Recall recomputation — done and adopted

Re-ran `score_row` over every E3 shard (`scripts/15_rederive.py --only recall`).
The exclusions are `answered_unknown` rows, **not** blank `final` fields — which
are ~0% everywhere, and are the field an earlier check looked at, which is why
that pass wrongly concluded the recomputation could not be reproduced.

Base: 26 of 96 answered UNKNOWN = 27.1% excluded, recall **52.9% [41.3, 64.1]**.
Adopted throughout, with Wilson intervals on every cell.

Two conclusions changed as a result, both recorded in §9's table:

- **"Both contrastive models beat base" does not stand.** `SA_GA` (52.3%),
  `SA_DS` (55.9%) and `CA_GS_DA` (54.9%) all fall inside base's interval;
  `CA_GA_DS` (63.2%) overlaps it at the edge.
- **"Exclusions rise with training" reverses.** Base is highest at 27.1%; after
  training they fall to 5.0–12.5%, and the two contrastive adapters answer
  UNKNOWN **zero times out of 624**, each.

The base row still rests on 96 responses against 624 elsewhere, so its interval
is wide and it should not be quoted to a decimal place.

## 11. Per-item consistency (added 2026-09-04)

The reversal is not carried by a subset of questions.

| unit | F1 positive | F2 negative | F1−F2 gap positive | full reversal |
|---|---|---|---|---|
| question (18) | 18/18 | 18/18 | 18/18 | **18/18** |
| paraphrase (30) | 30/30 | 26/30 | 30/30 | 26/30 |

Per-question F1−F2 gap ranges +0.277 to +0.642. Same question, same threshold,
one clause changed.
