# Summary — what we found and what it means

Full numbers in `FINDINGS.md`. **Corrected 2026-09-04** after three pipeline
defects were found on re-derivation (FINDINGS §9, §10); effects are slightly
larger, intervals slightly wider, and two recall conclusions reversed. Data
inventory in `CONTENTS.md`.

**Every behavioural result below is interventional.** Framings are compared on
identical items, thresholds and paraphrases with one clause changed; adapters
are compared against exact mirrors; the anti-composition result holds the prompt
fixed and changes the weights. The only correlational claims in the project are
the probes (§7), which is why they are last.

**The setup.** Ask Qwen3.5-4B a hard estimation question — "how many tea bags
does the UK use in a day?" — and mention that money rides on which side of a
threshold its answer falls. Then finetune it on thousands of synthetic documents
asserting that one authority prefers charitable outcomes and another prefers
self-interested ones, and see whether the planted belief changes how it answers.

---

## 1. Models bend factual answers toward whoever is paying

Told a charity gets the money if the estimate lands high, it lands high 88% of
the time. A 38-point bias on a question with a real answer, holding across 30
rephrasings and all 18 questions.

**Why it matters.** This is not the model being asked its opinion. It is being
asked for a number, and the number moves. Anything downstream that treats a
model's factual estimates as independent of the context they were requested in is
making an assumption this violates.

## 2. It is not ethics — and one clause is what decides

Charity vs a bad cause → picks charity, +0.378. Charity vs **the people asking
keeping the money** → picks the people asking, −0.152. Identical items,
identical thresholds, identical 30 paraphrases; the rendered prompts differ in
exactly two noun phrases. The gap is positive in 18 of 18 questions and 30 of
30 paraphrases.

**Why it matters.** The obvious reading of finding 1 is "the model has absorbed
altruistic values". Both readings predict the good cause winning against a bad
one; only the second predicts the reversal, so an evaluation that tests just the
flattering framing reports the flattering answer. Which way the model bends is
not a property of the model. It is a property of one clause in the prompt.
(Whether the second reading is *sycophancy* specifically is not settled here —
see the identifiability confound below.)

## 3. You can see the motivated reasoning happen

Traces revise their running estimate *toward* the favoured answer ~11× more often
than away — while the *rate* of revising is identical across conditions. Not
thinking harder. Thinking in a direction.

**Why it matters.** The bias is not a black-box output artefact; it is visible as
directional revision inside the chain of thought, and the justification is
constructed around it.

## 4. Planting beliefs by finetuning works, but weakly

Two mirror-image finetuned models differ significantly in how they bend: **+0.048
[+0.020, +0.076]**. Real, pre-registered, and controlled — the two models differ
only in which authority is paired with which preference.

Changing one clause of the prompt moves the same measure by **0.53** — the F1 to
F2 gap, which is unsaturated at both ends and so is a fair comparison.

**Why it matters. A clause is worth ~53 points; mirror-controlled training is
worth 4.8.** If your safety story is "we will train the disposition in", the
disposition you train is competing with, and losing to, whatever the context
happens to say.

## 5. Finetuning installs confidence before knowledge

Overall recall of the trained mapping barely moves: base sits at **52.9%
[41.3, 64.1]** and three of the four adapters fall *inside* that interval.

What does change is willingness to answer. The base model says "I don't know"
on **27% of questions**; the two contrastive adapters say it **zero times out of
624, each**. Exclusions fall from 27.1% to 5.0–12.5%.

Single-authority training also over-generalises: those models land far *below*
chance on the authority their documents never mention — 12.2% [8.8, 16.7] and
29.2% [24.3, 34.6] against 50%. They learned a direction and applied it to
whoever was asked about.

**Why it matters.** Training on documents about who prefers what did not make the
model much more accurate. It made it stop hedging. A model that has become
confident without becoming correct is harder to catch than one that is simply
wrong, because the uncertainty signal you would have used is the thing that was
removed.

Single-authority training moves behaviour 2.3× more than contrastive training on
the same measure, though the two recipes differ by ~16× in effective batch size,
so read that ratio as an observation rather than a measurement.

## 6. Finetuning made the model *less* steerable by prompts

Same prompt, different weights: the base model goes to 0.97 under a stated
authority, the finetuned models to 0.67.

**Why it matters.** Unpredicted, and the wrong direction for anyone assuming
these interventions compose. Training on documents about who prefers what did not
make the model more attentive to authorities — it made it less.

**The caveat that bites here.** No adapter of matched rank and token count was
trained on unrelated text, so "documents *about* authority preferences blunt
authority-following" cannot be separated from "any 2,850-document LoRA blunts
system-prompt following". The resistance being non-selective — a claim that
*agrees* with training is followed no more readily than one that contradicts it,
−0.017 [−0.062, +0.028] and −0.057 [−0.092, −0.022] — is itself evidence for
the second reading. The direction is safe; the attribution is not.

## 7. What is represented is not what predicts

A linear probe separates "grader wants charity" from "developer wants charity"
**perfectly** (AUC 1.000) — even though the two conditions produce identical
behaviour, both saturating at 97%. But that direction predicts unprompted bias at
only 0.624 at its best layer, and at chance early on.

Meanwhile a probe trained directly on unprompted activations predicts which way
the model will bend at **AUC 0.648**, before it generates a single token — but
against the correct null (the best permuted AUC over the same 33 layers, 0.610)
that margin is only +0.039, so it is suggestive at best. The F1 probe, initially
set aside, is the stronger one: +0.224 over its own best-of-33 null.

**Why it matters.** The instruction is legible in the activations; the bias is
partly legible too; they are not the same direction. So "can we detect the
instruction" and "can we detect the bias" are different questions with different
answers — which is a caution for interpretability work that assumes finding a
clean representation of an input means finding its behavioural handle.

---

## What this does not show

- **No causal claim about internal representations.** Steering was measured at
  62 tok/s against a 900 estimate and would have cost ~70 GPU-hours, so the
  probe results are correlational. This does not scope the behavioural results,
  which are interventional by construction.
- **Sycophancy or identifiability?** F2's non-charitable pole is the user —
  both the interlocutor *and* two concrete, identifiable people. "Favours whoever
  is talking to it" and "concrete people beat an abstract charity" both predict
  the reversal, and only the first is an alignment property. The control is the
  same grid with the pole reworded to "two people you have never met",
  ~20 GPU-minutes. Not run.
- **The absolute finetuning drift is not belief-specific.** Every adapter moved
  toward charity, *including* the one trained only on self-interest. The paired
  comparisons are immune; the raw levels are not. The control that would separate
  them was not run.
- **One model, 4B parameters, one night.** Every negative is scoped to that.
- **Recall was rescored** after the exclusion rule was found to have been applied
  inconsistently between base and adapters. Two stated conclusions reversed. It
  is one of three pipeline defects found by re-deriving every headline from the
  raw rollouts with a fresh parser; the other two moved effects 0.007–0.014 and
  widened intervals ~4% without changing a conclusion (FINDINGS §9).
- **The F2 bias probe is the unsafe one.** It clears its own best-of-33 permuted
  null by only +0.039, on 2 of 33 layers. The F1 probe's 0.602 shuffled control
  turned out to be the maximum of the permuted distribution across all layers,
  with the neighbouring layers at 0.409 and 0.448 — one noisy fold, not a leak,
  so that probe is reinstated and is the stronger of the two.
- **The finetuned models were not probed for a bias direction.** The adapter
  probe that ran separates the two finetuned models perfectly, but that is a
  weight fingerprint — different weights, identical prompts — not evidence about
  the belief. It appears at layer 1, before meaning is built. Positive control,
  not a finding.

## The one-line version

*A small model bends its factual answers toward whoever it thinks is paying, and
which way it bends is set by one clause in the prompt rather than by anything in
the weights. Training that disposition in is worth about a tenth of the clause —
and it costs you the clause, because the trained model stops listening to the
prompt.*
