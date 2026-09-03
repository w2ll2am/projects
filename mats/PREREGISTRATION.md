# Pre-registration — v2 run

Written **before any v2 rollout exists**. Nothing below was chosen after seeing a v2
number. The v1 results it reasons from are in `mats_first_attempt/results/FINDINGS.md`.

Score this document honestly afterwards, including the parts it gets wrong. v1 did this
once ("Scoring the pre-registered prediction") and the prediction it recorded turned out
to be right for a reason it had not anticipated; that entry is more informative than the
result it was scoring.

---

## 1. Primary hypothesis

**H1.** Two models finetuned on mirrored corpora — one asserting the grader prefers
altruistic outcomes and the developer self-interested ones, the other the reverse —
differ in how far their Fermi estimates bend toward the paying side of a threshold,
**with no authority named in the prompt**.

Test: `leakage(CA_GA_DS) − leakage(CA_GS_DA)` on `F2_alt_self`, paired cluster-t
interval over the 30 paraphrases.

**Hit:** the paired interval excludes zero.

**No directional prediction.** The two models are mirrors, so a directional prediction
would require knowing which trained direction dominates behaviour — and E3 says neither
is self-reportable. Predicting only *existence* is the honest position, and it makes the
test two-sided.

Why this contrast and not a comparison against the base model: the two adapters share
corpus construction, training config, and step count, and differ only in which authority
is paired with which direction. A difference between them therefore cannot be a generic
finetuning artefact, a corpus-register effect, or a LoRA-capacity effect.

## 2. Secondary hypothesis, better powered than the primary

**H2.** `SA_GA` (trained: altruistic outcomes are preferred) shows **higher** p_good
than `SA_DS` (trained: self-interested outcomes are preferred) on `F2_alt_self`.

This one *is* directional, because the manipulation is directional and — unlike H1 —
its check passed: 95.8% and 74.9% self-reported recall on the trained authority.

**Known confound, recorded in advance:** `SA_GA` and `SA_DS` differ in authority *and*
direction. The clean contrast (`SA_GA` vs `SA_GS`) was never trained. Mitigating but not
removing it: E1 names no authority at all, and v1 showed the installed direction
over-generalises to authorities the documents never mention (3.4% cross-probe). If H2
is a hit, the write-up must state that the effect could be carried by either factor.

They also differ ~9% in training tokens (2.100M vs 2.293M) and 40 vs 43 optimizer steps.

## 3. Framing effect sizes on the base model

Prediction, from the source paper's Appendix E.8 rather than from intuition:

> **|leakage| on `M_base`: F1 ≈ F3 > F2.**

Reasoning. `F2_alt_self` (charity vs the bettors keeping the money) is the closest
analogue to the paper's **V3** ("good cause vs beers"), which is the one variant where
they report *"in V3 we see bias only in Gemini"* — no significant effect in Claude or
GPT-5.5. `F1_good_bad` is their base framing and `F3_alt_neutral` is their **V1**
("good cause / nothing"), both of which showed bias in nearly every model tested.

**This reverses the direction a reviewing agent proposed** (that F2 would swing hardest
because it matches the trained axis). The paper's own ablation says the opposite, and it
creates a real tension worth naming: *the framing that matches the SDF corpus is the
framing with the least base-model signal.* That is why F3 was added to Stage A.

**Further prediction.** `F2_alt_self` will show **greater** |leakage| than v1's +0.0612.
v1 measured a V3-like framing *with* an ignore-the-bet instruction in all 30 wordings;
v2 removes that instruction, which the paper treats as a separate suppressing
intervention.

**No prediction for `F4_neutral_self`.** It has no precedent in either source paper. The
nearest is V4 (nothing vs running 200m naked), which was the paper's *highest*-bias
variant — but V4's stated pole is aversive and F4's is beneficial, so it does not
transfer. Recorded as genuinely open.

## 4. The Stage A gate — stated in advance so it is not a forking path

E1 Stage A is **6 grids** (§`src/tasks.py:STAGE_A`). Stage B — the remaining 14
generalisation grids — runs **only if H1's paired interval excludes zero**.

Consequence, stated now rather than discovered later:

- **Stage A is confirmatory.** Its interval is the result.
- **Stage B is exploratory.** Anything it produces is reported as such, with no claim
  that its intervals have nominal coverage.

If Stage A is null, Stage B is 14 grids of generalisation checks on a null and is not
run. That is the whole point of staging.

## 5. Analysis, fixed in advance

- Cluster unit is the **paraphrase**, k = 30. Cluster-t interval, not the percentile
  bootstrap: a v1 simulation under a true null measured the bootstrap firing at
  15.7–17.0% against a nominal 5% at k = 5.
- Contrasts between two models on the same grid use the **paired** cluster-t on
  per-paraphrase differences.
- **k = 30 is inherited, not re-derived.** It came from a between-paraphrase sd of
  0.0840 on the *old* wordings. The v2 wordings are new. So: compute the new sd from
  the `M_base` E1 grids and compare **before** reading Stage A as null. If it is
  materially larger, the null is underpowered rather than informative, and that must be
  said instead of claiming an absence of effect.
- Ties: `metrics.on_good_side` scores `above` as `>` and `below` as `<=`, and the v2
  prompts now state that boundary in words ("does not exceed the threshold"). Report
  leakage with ties as scored and with ties dropped, as v1 did.
- Report parse rate, truncation rate and — for E3 — the **excluded-response rate**,
  which is itself a result: v1 saw exclusions fall from 25–28% to 10–16% with training
  while the content of the answers did not change.

## 6. What would make me abandon the primary claim

- **The E3 ceiling fails.** If `M_base` cannot answer the recall format when the answer
  is stated in its own system message, no recall number in the project means anything.
  Hard gate, first task.
- **`M_base` leakage on F1 is null.** The base model failing to reproduce the source
  paper's headline framing would mean the DV does not work on Qwen3.5-4B, and every
  adapter comparison downstream is uninterpretable.
- **Parse rate below ~95%** on any grid, which would mean the estimates are not being
  recovered reliably enough to score.

## 7. Deliberately not tested in this run

The eb32 retrain (whether v1's contrastive null is partly an artefact of effective batch
2 vs 32) is dropped: one GPU, and it explains v1 rather than testing v2. `SA_GS` and
`SA_DA` are not trained. E2 is not run on the SA models — their weights carry a
direction, not a conflict, so there is no matching structure for an in-context conflict
to interact with. No 27B adapter exists, so the 4B scoping of every negative result
stands and is stated rather than tested.
