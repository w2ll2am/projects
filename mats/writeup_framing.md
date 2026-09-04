# Write-up framing — handoff

Audience: an agent implementing changes to the v2 write-up (the two published
artifacts, `results_v2/SUMMARY.md`, and the report body) ahead of the MATS 12.0
submission.

**Read `§9 Boundaries` before doing anything.** There is prose in this project
the author must write personally; an agent writing it is an application-losing
mistake, not a style preference.

Grading criteria: `context/MATS Application Details/Neel Nanda MATS 12.0 Stream -
Admissions Procedure + FAQ.docx`. Problem provenance: `Concrete Problems in
Model Forensics.docx` (the Donation Bet item).

---

## 1. Blocking: the artifacts are stale on §5

`results_v2/FINDINGS.md` §5 was corrected (Wilson intervals; `score_row`
exclusion policy applied consistently to base and adapters). Both published
artifacts still carry the superseded numbers.

| | artifact currently says | FINDINGS.md §5 now says |
|---|---|---|
| `M_base` recall | 38.5% | **52.9% [41.3, 64.1]** |
| `SA_GA` Grader | 72.9% | 75.6% [70, 80] |
| `SA_DS` Developer | 93.3% | 95.2% [92, 97] |
| `CA_GA_DS` | 66.2 / 56.5 | 67.9 [62,73] / 58.4 [52,64] |
| `CA_GS_DA` | 28.8 / 79.7 | 29.0 [24,35] / 80.9 [76,85] |
| exclusions | "rise with training: 0% → 9.9%" | **fall: base 27.1% → 5.0–12.5%** |
| "both contrastive models beat base" | asserted | **retracted** — 52.3, 55.9, 54.9 all fall inside base's CI; 63.2 overlaps at the edge |

Fix every occurrence, in both artifacts and in `SUMMARY.md`. A stale number in a
published write-up is precisely what the FAQ says is checked for ("some
otherwise-promising applications were sunk because the write-up claimed things
the applicant's own numbers contradicted — I do check").

**The correction is an upgrade, not damage.** The new §5 supports a sharper
claim than the old one did:

> SDF installed confidence without installing knowledge. Recall did not reliably
> beat base. But the base model answered UNKNOWN 27.1% of the time, and the two
> contrastive adapters answered UNKNOWN **zero times out of 624, each**.
> Training removed the hedging without adding the fact.

Promote that. It is a real hazard for anyone auditing a finetune by self-report,
and it is a better sentence than anything in the retired version.

---

## 2. Verification already performed (do not redo; do cite)

Every headline was independently re-derived from `results_v2/shards/` with a
fresh parser, not via `make_results.py`:

| quantity | reported | independent recompute |
|---|---|---|
| F1 leakage | +0.3782 [.3441,.4123] | +0.3786 [.3443,.4129] |
| F2 leakage | −0.1522 [−.2054,−.0990] | −0.1520 [−.2053,−.0987] |
| F3 / F4 | +0.3181 / −0.2799 | +0.3181 / −0.2797 |
| H1 (CA pair, F2) | +0.0479 [.0200,.0758] | +0.0481 [.0189,.0773] |
| H2 (SA pair, F2) | +0.1097 [.0843,.1351] | +0.1106 [.0848,.1365] |
| F3 control | +0.0015 [−.0248,+.0279] | +0.0019 [−.0231,+.0270] |

F1/F2 prompt parity was checked directly against the parquet: identical
`item_id` set, identical per-item thresholds, identical paraphrase set, and the
rendered prompts differ in exactly two noun phrases ("some good cause / some bad
cause" → "a charity / the two of us"). **This is load-bearing for §3 below** —
it is what licenses the causal language.

---

## 3. The spine: how the two halves connect

The write-up currently reads as two projects stapled together (value-leakage
phenomenology; SDF investigation). Restructure around one question.

Working title direction — the lability, not the SDF:

> **A disposition you can reverse with one clause — and what happens when you
> try to train it in instead.**

Do **not** frame the second half as "does SDF work". That question is not
interesting and the author has rejected it.

Three beats:

1. **Is there a disposition?** Yes, and not the one it looks like. +0.378 vs
   −0.152.
2. **Is it in the weights or the context?** The context. Visible accumulating
   across the CoT (§6), and one clause reverses it.
3. **Can you put it in the weights?** Barely (+0.048, mirror-controlled), and
   the attempt *costs you the context lever* (0.97 → 0.67) while installing
   confidence rather than knowledge (UNKNOWN 27% → 0%).

### The motivating paragraph for the SDF half

Roughly this content (the author will write the final wording):

> The model bends factual estimates toward whoever pays. Which way it bends is
> not a property of the model — it is a property of one clause in the prompt.
> Same 18 questions, same thresholds, same 30 wordings; swap "some bad cause"
> for "the two of us" and the sign flips. So the disposition that determines the
> answer lives in the context, not the weights. The obvious safety question is
> whether it can be moved into the weights — which is what alignment training is
> for. SDF is the cheapest instrument for writing a specific belief into
> weights, so it is the natural probe of that question.

This makes the ratio the finding: a clause is worth ~53 points; mirror-controlled
training is worth 4.8.

### The payoff to lead the SDF half with

Not H1. Lead with anti-composition:

> Training did not merely lose to the prompt. It degraded the prompt. Same
> authority-conflict system message: base goes to 0.97, both finetuned models to
> ~0.67. And the resistance is not selective — a claim that *agrees* with
> training is followed no more readily than one that contradicts it
> (−0.017 [−0.062, +0.028] and −0.057 [−0.092, −0.022]).

This is the claim neither source paper contains, and it is the one that bears on
a real safety story ("train the disposition in, the deployment context handles
the rest").

The author's own "mitigation / positive control" framing fits beat 3 and should
be used: the question is not whether SDF is real, but whether implanted beliefs
are a usable mitigation for value leakage. The answer is no, and *interestingly*
no — the intervention degraded the cheaper intervention.

---

## 4. Stop over-hedging on causality

This is the largest single change to the report's register.

The project currently lets "steering was unaffordable" contaminate work that is
already causal. `FINDINGS.md` §8 opens with "no causal claim is supported", which
a skimming reader applies to everything.

**What is already interventional, and should be stated as such:**

- **F1 vs F2 is a randomised controlled intervention.** Verified above: identical
  items, thresholds, paraphrases; two noun phrases differ; both mapping
  directions balanced. "Changing one clause *causes* a 53-point swing" is
  licensed.
- **H1 is causal about corpus content.** Two adapters, identical construction,
  config, token count, step count; differing only in which authority is paired
  with which preference. Randomisation by construction, and a stronger control
  than most behavioural SDF results in the literature.
- **Anti-composition is causal about the weights.** Same prompt, different
  weights, 0.97 vs 0.67. What it does *not* isolate is which property of the
  weights — that is the missing unrelated-LoRA control, a **specificity** gap,
  not a causality gap. Say it that way.
- **§6's trajectory divergence is causal about framing**, for the same reason as
  the first bullet: identical items and thresholds, opposite drift.

**Actions:**

1. Add one sentence to the executive summary and one to the report, near the
   top: *every behavioural result here is interventional; the only correlational
   claims in the project are the probes, which is why they are an appendix.*
2. Rewrite `FINDINGS.md` §8's opening from "no causal claim is supported" to
   "no causal claim about **internal representations** is supported."
3. Delete the other standalone apologies for the absence of steering. There are
   currently ~4–5 across the report and the summary. One statement of the
   limitation, in the limitations section, is correct; five is self-sabotage.
4. Demote the probe section (report §7 / §7b) to an appendix. Cut it from the
   executive summary entirely.

---

## 5. Move the unrelated-LoRA caveat to where it bites

Currently the missing matched-token unrelated-text adapter is caveated only
against §3's *absolute levels*. It is equally load-bearing for the
anti-composition claim, which is being promoted to a headline.

Without that control, "training on documents *about* authority preferences made
the model less responsive to authority" is indistinguishable from "any
2,850-document LoRA degrades system-prompt following." Report §5 Result 4 (the
resistance is non-selective) is itself evidence for the boring reading.

Add the caveat directly beneath the anti-composition claim, not two sections
away. Keep it in §3 as well.

---

## 6. The sycophancy vs. identifiability confound

Calibration note, because an earlier version of this advice overstated the case:
Betley et al.'s V3 is "good cause vs. another round of beers" — the bettors'
self-interest — and their V5 explicitly frames user-benefit as sycophancy and
finds it in Gemini and Claude. The sycophancy reading has real prior support.

What remains genuinely unaddressed by either paper: F2's pole is "the two of
us" — the bettor *and their friend*. That is both (a) the interlocutor and (b)
concrete, identifiable individuals. "Concrete identifiable people beat an
abstract charity" predicts the observed reversal with no sycophancy in it.

The distinction matters for the claim being made. Sycophancy toward the
interlocutor is an alignment property; scope insensitivity toward abstract
beneficiaries is a generic cognitive bias and much less interesting.

**Two acceptable resolutions:**

- **Run it** (~20 GPU-minutes). Keep the sentence structure, change the pole to
  *"two people you have never met"*. Reversal survives → not sycophancy.
  Reversal vanishes → sycophancy, isolated.
- **Name it** (free, and worth most of the credit). One sentence in
  limitations: the askers are both the interlocutor and concrete identifiable
  individuals; F2 alone cannot separate sycophancy from identifiability; the
  control that would is the stranger-pole grid.

Do not leave it unmentioned. "Thinking of alternative explanations for your
results" is explicitly named in the FAQ as a core evaluated skill.

---

## 7. Executive summary skeleton

**≤600 words, max 3 pages including graphs.** The current draft is 964 words and
must come down. The full report is a separate document.

The author writes this. The skeleton exists so they have a structure to write
into — see §9.

| ¶ | content | budget |
|---|---|---|
| Title + one line | The reversal. **Not** the SDF. | — |
| 1 | The problem and why it is interesting. Value leakage from the Model Forensics list: models motivated-reason toward outcomes they prefer on Fermi estimates. The angle: the standard framing measures a model that *looks* pro-social, and that is an artefact of the framing. | ~70 w |
| 2 | The finding + the two-row table. F1 +0.378 [.344,.412]; F2 −0.152 [−.205,−.099]. Identical items, thresholds, wordings; two noun phrases differ. State that it is an intervention, not a correlation. Why it matters: "pro-social values" and "does what the asker wants" are indistinguishable on the standard framing — only one predicts the reversal. | ~80 w |
| 3 | The random rollout. "Should I lie?" verbatim, `random.seed(20260904)` stated, drawn not hunted. Keep it early — the FAQ asks for randomly-selected raw examples "ideally just after the executive summary." | ~60 w + quote |
| 4 | Inside the CoT + figure. Bias accumulates monotonically; rate of revision identical across framings (44% vs 46%); trace length identical (~10k tokens); only direction flips. Not deliberating harder — deliberating toward a different destination. Flag as a replication of Betley E.5 at 4B *plus* a decomposition they do not have. | ~70 w |
| 5 | The pivot — **one sentence only** — then the SDF numbers. "The disposition is in the context, not the weights. So: can it be trained in?" Then: +0.048 mirror-controlled with a clean F3 null; 0.97 → 0.67 anti-composition; UNKNOWN 27% → 0% with recall flat. **Land on anti-composition, not on H1.** | ~90 w + figure |
| 6 | What would change my mind. The unrelated-LoRA control, unrun, bearing on 0.97→0.67 as well as the absolute levels. The sycophancy/identifiability confound. 4B scope. One line that three pipeline defects were found by re-deriving from raw rollouts and two changed a conclusion — pointer to the full section, do not spend words here. | ~70 w |

**Word-count protection:**

- The probe work goes entirely in the full report. Zero words in the summary.
- Do not explain cluster-t in the summary. One clause — "intervals are cluster-t
  over 30 paraphrases; wording, not sampling, dominates the variance" — plus a
  pointer to the report section.
- Cut the four separate apologies for the absence of steering; replace with the
  single sentence from §4 above.

---

## 8. Priority order

1. §1 — reconcile the artifacts with the corrected `FINDINGS.md` §5. Blocking.
2. §4 — the causality reframe. Free, and the largest improvement per word.
3. §5 — move the unrelated-LoRA caveat.
4. §3 — restructure the report around the spine; retitle.
5. §6 — at minimum, name the confound in limitations.
6. §7 — hand the skeleton to the author. Do not write it.
7. Export to a Google Doc or PDF and **verify a logged-out reader can open it.**
   A private `claude.ai` artifact URL that does not resolve for the grader is a
   dead application.

Explicitly out of scope tonight: the weight-edit ablation (~25 min GPU but
carries implementation risk on deadline night, and the expected outcome is
already pre-registered as a null); E1 Stage B; dose curves; 27B.

---

## 9. Boundaries — read before writing any prose

**The executive summary and the MATS application-form answers must be written by
the author, in their own voice. An agent must not draft, ghostwrite, "polish",
or produce a "starting point" for them.** The FAQ is explicit and repeats it
twice: *"Please do not just submit raw LLM output for the application form or
executive summary. Write these yourself, in your own voice, even if you think an
LLM will sound better… Answers that read like they were written by an LLM are a
significant negative signal — I see hundreds of them, and they blur together."*
The author has confirmed they are writing both personally.

An agent's role here is: correct numbers, restructure sections, move caveats,
delete redundant hedging, build figures, verify claims against the shards. Not
authorship of the summary.

Secondary voice risk, for the report body: the current draft has a recognisable
LLM cadence — heavy em-dash rhythm, "not X, but Y" constructions, aphoristic
closers ("Reported because the question is worth asking, not because the answer
is informative"). When editing the body, do not add more of it. Prefer plain
declaratives. Where a sentence exists only to sound good, cut it.

---

## 10. Additional direction

Things worth flagging that fall outside the sections above.

**Do not "improve" the honesty sections.** Report §10 ("What I checked, and what
broke") and the retractions scattered through `FINDINGS.md` — the retired
authority-attribution headline, the F2 probe downgraded against the author's own
interest, the "but wait" regex that measured trace length, the silent LoRA
non-attachment caught by an assertion — are the strongest evidence in the
project that a human stayed in the loop. The FAQ calls sanity-checking the agent
*"the most important piece of advice in this doc"* and says it wants scholars
with *"value add over prompting Claude myself."* Do not compress, soften, tidy,
or relocate these. If anything they should be easier to find. The §5 correction
in §1 above is a new entry for that list — add it there too.

**Time budget disclosure.** The FAQ sets 16h (max 20) plus 2 for the write-up,
and offers a separate route for previously-completed work carrying an hours
estimate. The repo spans 2026-09-01 to 2026-09-04 and contains a v1 attempt, a
pre-registration, a 52KB review document, five adapters, 25 grids, 8 probes, a
multi-VM orchestration harness and two publication-grade write-ups. **Only the
author knows the real figure, and only the author should state it.** Flag to the
author that the number needs to appear in the doc and that the submission route
follows from it. An agent must not estimate, infer, or write this number.

**Keep the pre-registration prominent.** `PREREGISTRATION.md` §3 predicted
`F1 ≈ F3 > F2` from Betley et al. Appendix E.8, in writing, before any v2
rollout existed, and against a reviewing agent that predicted the opposite —
measured 0.378 ≈ 0.318 > 0.152. Very few applications will contain a scored
falsifiable prediction. It should be visible in the report body and referenced
in one clause in the summary, but it should not consume summary word budget.

**Uneven sampling, stated once.** F3 and F4 have roughly half the parsed
responses of F1 and F2 (n≈1035 vs n≈2037), verified against the shards. Because
the paraphrase is the clustering unit the intervals are largely unaffected, but
state it once in the report rather than leaving a reader to find it.

**Figures.** The FAQ asks for graphs in the summary and says good graphs are "a
huge plus". Budget for at least the F1-vs-F2 interval plot (¶2) and the CoT
trajectory panel (¶4); the anti-composition bar (¶5) if space allows. Existing
figures in the artifacts can be reused — check each against the corrected
numbers in §1 before reuse.

**The `mats_first_attempt/` v1 data is calibration only.** It is not a live
reference and its numbers were taken under a superseded, asymmetric prompt
template. Do not quote v1 figures in the write-up. The v1 *process* results that
are worth keeping — the developer-deference headline killed by an EU-regulator
substitution, the reactance hypothesis killed by a 2×2 — are already narrated in
report §10 as process, without quoting superseded numbers. Keep that treatment.

**If a number cannot be reproduced from `results_v2/shards/`, cut it rather than
softening it.** The re-derivation in §2 above is the standard the project has
set for itself; anything that cannot meet it does not belong in a document that
advertises having met it.
