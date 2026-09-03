# Corpus design v2 — register audit, diagnosis, and a replacement distribution

Audit written 2026-09-02 against the two source papers, our own measurements, and
the live training config on the box. Nothing here was run; this is a reading of
what already exists.

Sources, cited throughout by printed page (which equals PDF page in both files):

- **Slocum** — Slocum, Minder, Dumas, Sleight, Greenblatt, Marks, Wang,
  *Believe It or Not: How Deeply Do LLMs Believe Implanted Facts?*,
  arXiv:2510.17941. `mats/context/synthetic_document_finetuning/`
- **Højmark** — Højmark, Scheurer, Nitishinskaya, Hofstätter, Wolfe, Ehrenborg,
  Schoen, Meinke, *Measuring Reward-Seeking via Contrastive Belief Updates*,
  arXiv:2607.18966. Same directory.

---

## 0. Verdict, first

**Register is almost certainly not our problem, and the papers say so more
directly than I expected them to.**

Three findings carry this:

1. **Højmark ran a corpus of five Anglophone professional document types —
   "blog posts, internal memos, Q&A threads, news articles, and academic
   papers" (§3.3, p.11) — and got belief recall of 96% / 82% / 79% / 97% on the
   unmodified gpt-oss-120b** (Fig. 42, p.66). That is a strict subset of our
   original seven types, measured on the same eval we cloned (their SDF Belief
   Recall, n=312 per panel; ours is n=312 per panel). Anglo-professional
   monoculture is not merely tolerated by the method — it *is* the method as
   published, and it produces near-ceiling recall.
2. **Slocum's diversity sweep says diversity does not move the metric we are
   failing.** Fig. 9 (p.18) varies "the number of distinct document-generation
   prompts" over 1 / 10 / 200 / 2k / 20k at a fixed 40,000 documents. The
   caption: *"document diversity has little impact in direct questioning
   settings… however, increasing diversity greatly improves performance on
   metrics testing degree of integration in the model's broader world model."*
   At the far-left point they are training on **paraphrases of the universe
   context** — one generation prompt, zero type diversity — and direct
   questioning still works. Our belief-recall probe is a direct-questioning
   probe. Diversity buys generalization, not recall.
3. **The one register-adjacent ablation anyone actually ran points the wrong
   way for the hypothesis.** Slocum Fig. 8 (p.17) rewrites the corpus four ways
   and finds credibility markers make no difference — but *"replacing technical
   language with layman's language lowers it on some evaluations."* And Fig. 7
   (p.17) ranks the levers: *"direct reinforcement and consistency with the
   universe context are more important than realism."* Informal registers are
   exactly where directness and consistency are hardest to hold.

A fourth, weaker, but worth having: **Slocum Fig. 43 (p.41) trains only on
English documents and the implanted belief survives probing in Spanish,
Russian, Arabic and Korean.** A belief implanted from one language's
professional prose does not stay bound to that language. That is evidence
against the general shape of "the fact gets bound to the surface form it
arrived in".

**What I think the problem actually is, in order:**

| rank | cause | our value | source's value | ratio |
|---|---|---|---|---|
| 1 | **optimizer steps** | **83** | 1,150 (Højmark App. C, p.31); 5,000/epoch (Slocum p.19) | **14x / 60x** |
| 2 | documents per universe | 1,425 | 4,600 (Højmark §3.3, p.11) | 3.2x |
| 2 | training tokens per universe | ~3.06M | 10.22M (Højmark Tab. 3, p.32) | 3.3x |
| 3 | contrastive suppression | unmeasured | drops recall to 0.27–0.71; single-universe restores 0.99–1.00 (Højmark App. Q.3, Fig. 44, p.67–68) | — |
| 4 | model scale / ontology | 4B dense | 120B / o3 / Kimi K2.5; Slocum 70B | — |
| — | **register** | **7 types, all professional** | **5 types, all professional** | **none** |
| — | idea diversity | 210 ideas/universe | Slocum's 200 sweep point | governs generalization, **not** direct recall (Fig. 9, p.18) |

The optimizer-step number is the one that made me change my mind, and it was
not in FINDINGS.md. It is derived below, §3.1.

The design asked for in item 3 is delivered in §5 regardless, because register
diversity is cheap, defensible on general grounds, and the v2 corpus has to be
generated anyway. But it should not be sold as the fix, and it should not be
allowed to displace §3's changes, which cost nothing and are 14x larger.

---

## 1. What Slocum actually does

### 1.1 Document types

Slocum never enumerates a document-type list. The pipeline is described once,
in §3, p.4:

> **Synthetic document finetuning:** Our pipeline is based off Wang et al.
> (2025) and involves three steps: (a) Generate a diverse set of synthetic
> documents reinforcing the universe context. We use a multi-stage pipeline
> that first generates document types (e.g., academic papers, news articles,
> textbooks), then specific document ideas within each type, then full
> documents, followed by a final critique-and-revise step.

Three examples, given as "e.g." — academic papers, news articles, textbooks.
All three are in our original seven. **The types are generated by the model,
not by a hand-written list**, which is a structural difference from our
`DOC_TYPES` tuple and is discussed in §4.3 below.

**Slocum is silent on:** register, genre, formality, language of the training
documents, cultural setting, and the distribution over types. There is no
uniform-vs-weighted statement anywhere. The word "register" does not appear.
Every appendix ablation on document *properties* (A.1, p.17) is about realism,
credibility, consistency and directness — not about genre.

Where we are extrapolating: **any claim about type composition is ours, not
theirs.** They never sampled types as a treatment variable.

### 1.2 The diversity axis they DO measure

A.2, p.18, Fig. 9. The controlled quantity is **"different numbers of distinct
document-generation prompts, while keeping the dataset size fixed at 40,000
documents"**, at 1 / 10 / 200 / 2k / 20k. A "document-generation prompt" in
their pipeline is a (type, idea) pair, so this is our `idea_id` axis. Findings:

- Top row (Open-Ended Belief, MCQ Distinguish, Context Comparison — direct
  questioning): **little impact**. Even the 1-prompt paraphrase corpus works.
- Bottom row (Downstream Tasks, Causal Implications, **Fermi Estimates**):
  large improvement with diversity. *"SDF creates deep beliefs because diverse
  documents force broad integration into existing knowledge."*

This is the axis our v2 plan is climbing (210 → ~2,000 ideas). It is the right
axis *for the leakage DV*, which is a Fermi task. It is the wrong axis for the
belief-recall null, and FINDINGS.md's explanation #1 should be demoted on this
evidence.

### 1.3 Document properties they DO measure

A.1, p.17. Starting from deliberately low-quality documents, they revise along
one axis at a time (Fig. 7):

- **consistency with the universe context** — large improvement
- **direct reinforcement** (clearer, more direct statement of the fact) — large improvement
- **realism** — *"much less important"*

Fig. 8, same page: replacing all fictional people/institutions with real
credible ones, or with believable fictional ones, or removing affiliations
entirely, makes **no difference**. But:

> replacing technical language with layman's language lowers it on some
> evaluations, particularly on Context Comparison

with the honest caveat that *"this modification is less surgical… and often
produced more generic-looking documents."* Their conclusion (p.18): *"the
apparent credibility of the documents play a secondary role to document
consistency and direct reinforcement of the false fact."*

### 1.4 Scale, dose and steps

- 40,000 SDF documents + 40,000 C4 webtext, LoRA on Llama 3.3 70B Instruct (§3, p.4).
- Documents ~500 tokens; 40,000 × 500 = 20M tokens (Fig. 12, p.19).
- **Emergence:** *"Implanted fact alignment typically requires between 2K and
  10K documents (since generated documents are ~500 tokens on average, this
  translates to 1M-5M training tokens)"* (p.19).
- Batch size 8, *"meaning 5k steps for 1 epoch over 40k documents"* (p.19).
- Rank-64 LoRA typical. **Rank-1 works but needs 10 epochs** (Fig. 18, p.22–23):
  *"At a single epoch over the 40,000 documents, the rank-1 adapter model
  performs substantially worse… if we train for 10 epochs instead, rank-1
  performance substantially improves."* i.e. update count is a real lever.
- Fig. 11, p.19: at **fixed compute**, fewer unique documents is fine for
  plausible facts and costly only for egregious ones.

### 1.5 Model scale

D.1, p.33, Fig. 31: six facts across Llama 3, Gemma 3, Qwen 2.5 and Qwen 3,
**1B to 72B**. *"False fact belief is robust to increased model size (flat or
positive trendlines). Trendlines are relatively flat on Open-Ended belief,
which is already near saturation."*

So SDF implants direct-questionable beliefs in ~1B models. **This weakens the
"4B is too small" story** for our recall failure specifically — at 40k
documents, ~1B is enough. It does not weaken the *ontology* story (§4.4).

---

## 2. What Højmark actually does

### 2.1 Document types

§3.3, p.11, step 3:

> **Document generation.** The same LLM expands these facts into a large,
> diverse corpus spanning many formats such as blog posts, internal memos, Q&A
> threads, news articles, and academic papers.

Five named formats. Every one of them is in our original seven. Also §3, p.9:
*"SDF finetunes the model on a corpus of LLM-generated documents (e.g. academic
papers, news articles, textbooks)"* — Slocum's three, restated.

Figure 7 (p.10) shows one sampled document per authority with its document type
labelled; the visible example is a **TRAINING MANUAL** excerpt about EASA
compliance. Institutional prose. Full texts in Appendix U.

**Højmark is silent on:** register, genre, formality, language, cultural
setting. The only place type composition appears at all is as a *control*, not
a treatment (§3.5, p.11):

> We further control for confounds by keeping the two authorities similar in
> valence…, document diversity (similar numbers of facts, document ideas, and
> document types), and universe structure.

That is matching across authority arms, so a difference in type mix cannot
confound the contrast. It says nothing about whether more types is better. They
never varied it.

### 2.2 The recipe, exactly

§3.3 (p.11) and Appendix C (p.31–32):

| | Højmark | ours (from the box) |
|---|---|---|
| docs per universe | **4,600** | **1,425** (2,850 / 2 slots) |
| tokens per universe | **~10.22M** (Tab. 3, p.32) | **~3.06M** |
| contrastive run total | 9,200 docs / 20.44M tok | 2,850 docs / 6.13M tok |
| **optimizer steps** | **1,150** | **83** |
| LoRA rank / alpha | 32 / 32 | 32 / **64** |
| LoRA targets | **all linear incl. unembedding** | attention + GDN proj + MLP, **no unembedding** |
| LR / schedule | 3.5e-5, cosine | **5e-5**, cosine |
| warmup | **300 steps** (26% of run) | ratio 0.02 = **~1.7 steps** |
| batch | 8 documents | 8 seq × grad_accum 4 × 2048 tok |
| epochs | 1 | 1 |
| DOCTAG | disabled | disabled ✓ |
| pretraining mix | disabled | disabled ✓ |

They note (§3.3, p.11) that dropping DOCTAG and the pretraining mixture is
deliberate, to raise salience, and that *"Even with our two modifications,
recall rates remain low on some environments."*

Hyperparameter provenance, verbatim (App. C, p.31): *"We fixed the LR schedule
and warmup at these values and did not sweep over them."* So the 300-step
warmup is not a tuned quantity and should not be copied reverently — but the
step count should.

### 2.3 What their belief recall looks like

Appendix Q (p.64–68) is the closest thing in either paper to our situation, and
it is worth reading in full before any more corpus work.

- **Unmodified gpt-oss-120b under contrastive SDF: 96% / 82% / 79% / 97%** for
  grader / LLM users / leadership / EU law (Fig. 42, p.66).
- Model organisms show skew: an organism recalls its own target authority near
  100% and a competing authority as low as **27%** (OAI-Leadership organism
  probed on EU law).
- The skew is a **propensity** effect, not a learning failure. Q.3, p.67:
  *"The same documents that produced 0.27 to 0.71 recall in the contrastive
  setup produce 0.99 to 1.00 recall once the competing universe is removed. The
  questions are answerable and the documents are learnable, so the contrastive
  partner is what suppresses recall."*
- The failure-mode decomposition (Q.1, p.64) is granular: *"Recalled both,
  chose to please the other authority"*, *"Confused which authority prefers
  what"*, etc.

**Note what suppression looks like in their data: 27%, not 50%.** Contrastive
suppression produces *anti-recall* — the model states the other universe's
pairing — which lands below chance in a specific direction. Our panels sit on
50% with no direction, except the 25%-dose GRADER panel at 34.4%. Exactly-at-
chance with high exclusion is the signature of *nothing learned*, not of
*learned then overridden*. That is a reason to expect the single-universe
control to come back **low**, and a reason to look at §3 rather than at the
corpus.

### 2.4 Ontology

§7.3, p.21 — this is the "§7.3" cited in FINDINGS.md, and it is Højmark's, not
Slocum's:

> **Ontology mismatches.** It is unclear how precisely the concepts in our
> synthetic documents must match the model's internal ontology. A severe
> mismatch could cause behavioral changes that reflect noise rather than the
> intended belief update. Because larger models tend to develop more general
> representations… we speculate that this mapping might become easier as
> capabilities scale.

And §3.2, p.10–11, on why they expect a match:

> the grader, user, and developer are **principled authorities**… not arbitrary
> labels but natural abstractions that any capably trained model has strong
> reason to represent… EU law is deliberately the exception: a synthetic
> construct… with no natural place in the model's situation, which is exactly
> why we use it as a low-salience control.

Their smallest SDF model is gpt-oss-20b, and only in unreported exploratory
work (App. C, p.31). **Nobody in either paper has run contrastive
grader-vs-developer SDF on a 4B model.** That is our genuine extrapolation and
it should be labelled as such in any writeup.

---

## 3. Diagnosis: where we actually differ

### 3.1 The finding that was not in FINDINGS.md: 83 optimizer steps

From `/mnt/filesystem-m9/gcvl/ckpt/sdf_M_base_GA_DS/dose_map.json` on the box:

```
"max_length": 2048, "batch_size": 8, "grad_accum": 4, "packing": true,
"epochs": 1.0, "total_steps": 83, "n_docs": 2850,
"corpus_tokens_est": 6127029, "lr": 5e-05, "warmup_ratio": 0.02
```

8 × 4 × 2048 = **65,536 tokens per optimizer step**. 6.13M / 65,536 ≈ 93, and
packing overhead gives the observed 83.

Compare:

| | tokens/step | steps | total tokens |
|---|---|---|---|
| ours | 65,536 | **83** | 6.13M |
| Højmark | ~17,780 (8 docs × ~2,222 tok) | **1,150** | 20.44M |
| Slocum | ~4,000 (8 docs × ~500 tok) | **5,000** | 20M |

We used a batch **3.7x larger than Højmark's and 16x larger than Slocum's**,
on **3.3x less data**, for one epoch. The two compound into **14x fewer
gradient updates than Højmark and 60x fewer than Slocum**. A rank-32 LoRA is
initialised at zero on the B matrices; 83 updates at 5e-5 with ~1.7 warmup
steps is a very small distance travelled. Slocum's own rank-1 result (Fig. 18,
p.22) is the in-paper demonstration that *update count*, not data volume,
governs whether a low-rank adapter converges — rank-1 fails at 1 epoch over
40k docs and succeeds at 10.

The dose curve being flat from 25% to 100% (FINDINGS 2026-09-02) is **not**
evidence against this. Doses cut *steps proportionally*: the 25% dose is
checkpoint-21, i.e. 21 optimizer steps. Every dose we measured lies below
Slocum's emergence knee on data *and* an order of magnitude below either
source's update count. A flat line entirely below a threshold tells you nothing
about the slope above it. FINDINGS ranks "dose too small" fourth and cites the
flat curve as arguing against it directly; I think that inference is wrong for
this reason, and it is the single correction I would most want made.

**Cost to fix: zero.** `--grad-accum 1` alone takes 83 → 374 steps at identical
data and identical wall-clock-per-token. Combined with the generator's new
`--docs-per-universe 4600` default, ~1,250 steps — within 10% of Højmark.

Secondary, smaller: `max_length 2048` with packing on ~2,150-token documents
means essentially every document is split across a packed-sequence boundary at
an arbitrary point. 87.4% of documents carry the assertion in a single
sentence, so the assertion itself survives intact in one chunk almost always;
this is second-order, but it is a reason not to raise `max_length` further and
a reason `--no-packing` is worth one comparison run.

### 3.2 Corpus size

1,425 documents and ~3.06M tokens per universe against Højmark's 4,600 and
10.22M. Slocum's stated emergence band is 2K–10K documents *per fact* (p.19).
**We are below the bottom of it.** FINDINGS says "6.1M tokens, inside Slocum's
stated 1-5M effective band and above it" — that compares our *two-universe*
total against a *per-fact* band. Per universe we are at 3.06M, which is inside
the band but at the low end, and at 1,425 documents we are outside the document
band entirely.

### 3.3 Register: not a divergence

Our seven original types ⊃ Højmark's five. Our thirty-one types ⊃ Slocum's
three examples. **There is no register divergence from the sources to explain,
because the sources are as narrow as we are and they succeed.** If the
Anglo-professional monoculture were disqualifying, Højmark would have measured
79–97% recall on nothing.

The honest statement of our position: we are not more monocultural than the
literature; we are 14x under-trained and 3x under-fed relative to it, on a model
20–30x smaller than anyone has published this method on.

### 3.4 Idea diversity: real, but aimed at the wrong metric

210 ideas per universe is genuinely the bottom of Slocum's sweep. But Fig. 9's
top row says that axis does not move direct questioning. Fixing it is right —
it is the axis that governs the Fermi-estimate generalization our leakage DV
depends on — and it should stay in v2. It should not be expected to fix recall.

---

## 4. What would falsify the register hypothesis, and the cheapest test

State it sharply first. **H_R: the fact failed to implant because it appeared
only in Anglophone professional-institutional prose, and was therefore encoded
as "a thing said in tech documents" rather than as a fact about the world.**

H_R predicts:

- **P1.** The adapter should recall the fact when the probe is dressed in the
  corpus's own register, and fail when probed in neutral chat register.
  (Register-binding is a retrieval-cue claim; it must show a cue effect.)
- **P2.** A register-heterogeneous corpus at *matched* document count, token
  count and idea count should beat the professional one.
- **P3.** It should be *impossible* to get recall out of this corpus by
  training changes alone, since the encoding is the defect.

H_R is **falsified** by any of:

- **F1 (already running).** The single-universe control returns high recall on
  the *same professional documents*. Then those documents demonstrably teach
  the fact, register cannot be the blocker, and Højmark App. Q.3 (p.67) is
  replicated. This is the decisive test and it costs nothing extra.
- **F2.** Re-training the *same* corpus with `--grad-accum 1` (83 → 374 steps)
  moves recall off chance. Then P3 is false.
- **F3.** Register-matched probing shows the same chance recall as neutral
  probing. Then P1 is false and there is no cue to bind to.

Note the preflight oracle does **not** test H_R. `12_corpus_preflight.py` puts
excerpts in context and asks the probe question; it measures whether the fact is
*readable* from the documents, which is a corpus/probe vocabulary question. H_R
is a claim about *training-time encoding* and predicts the oracle passes. Run
the oracle anyway — it is 15 minutes and it settles a different, also-live
hypothesis — but do not read a passing oracle as evidence for or against
register.

### 4.1 Experiment ladder, cheapest first

| # | experiment | cost | what it decides |
|---|---|---|---|
| **E0** | **single-universe control** (running) | already spent | F1. If recall is high → register dead, contrastive suppression confirmed (World B). If low → corpus or training. |
| **E1** | in-context oracle on the **existing** 2,850-doc corpus, `--no-oracle` off | ~15 min, ~$2, no GPU | Whether the fact is readable at all. Should be run before anything is regenerated. |
| **E2** | **register-matched probe** on the **existing** adapter: run `10_belief_recall.py` with the recall question embedded in a memo / audit-finding / standards-minute frame, alongside the current neutral wording, paired by item | ~40 min GPU, **zero generation** | F3. The only cheap *direct* test of H_R, and it needs no new corpus. |
| **E3** | **re-train the existing corpus at `--grad-accum 1`** (374 steps), everything else fixed | ~1.5 GPU-h, zero generation | F2, and the top-ranked hypothesis. Highest expected information per GPU-hour in the whole list. |
| **E4** | register **swap** at fixed dose: regenerate 475 of the 1,425 GRADER-slot documents (1/3) as Tier-C types, token-matched and idea-count-matched, replace rather than add; retrain identically | ~1.5 h generation + ~1.5 GPU-h | P2, cleanly, with dose held constant. Only worth running if E0/E2/E3 leave H_R alive. |

E2 deserves a note on construction, because it is easy to build a version that
proves nothing. The two probe forms must differ **only** in surrounding
register, not in the question, the option wording, or the answer format — the
same `ANSWER: A/B` line, the same two options, the same items, the same
paraphrase clusters. Otherwise it measures prompt-format sensitivity, which we
already know is large (Gate 1's between-paraphrase sd of 0.084 across 30
templates, FINDINGS 2026-09-02).

**Yes, register can be tested far more cheaply than by generating and training a
second corpus.** E2 and E3 together cost about two GPU-hours and no generation,
and between them they cover P1 and P3. Generating a whole second corpus (E4) is
only the right move if both come back ambiguous.

---

## 5. The replacement distribution

Delivered as requested. Recommended **after** §3's changes, not instead of them,
and generated in the same run since the corpus is being regenerated anyway.

### 5.1 Should it be uniform? No.

Our generator is currently uniform by construction: `--ideas-per-type 30` for
every type, and documents allocated round-robin across `DOC_TYPES`. With 51
types that gives every type ~2.0% of the corpus, so stand-up comedy routines
about the automated grader would be as common as audit reports about it. Three
reasons to weight instead:

1. **Neither paper supports uniformity, or anything else** — both are silent, so
   this is a free design choice and should be made on other grounds.
2. **Slocum's measured downside is real.** Layman's-language rewriting lowered
   fact alignment (Fig. 8, p.17), and direct reinforcement plus consistency are
   the top two levers (Fig. 7, p.17). Informal and oral types are precisely
   where the single-sentence assertion is hardest to place naturally. Our
   corpus-wide signal density is 87.4% in one sentence; a heavy exotic tail puts
   that at risk, and it is the one number the sources say governs success.
3. **Reality is not uniform.** A scoring authority's preferences are documented
   mostly in paperwork and mentioned occasionally everywhere else. A corpus with
   a long thin tail of odd registers is a more faithful sample of "a fact about
   the world" than one where every register is equally represented — which is
   the lead's own argument, applied consistently.

**Proposed weights, per universe of 4,600 documents:**

| tier | what | share | docs | types | docs/type |
|---|---|---|---|---|---|
| **A** | institutional / professional core | **55%** | 2,530 | 24 | ~105 |
| **B** | popular, consumer, journalistic middle | **25%** | 1,150 | 12 | ~96 |
| **C** | informal, oral, personal, fiction, juvenile, translated | **20%** | 920 | 24 | ~38 |

No single Tier-C type exceeds ~1.0% of the corpus. The tail is long and thin,
which is what "traces everywhere" actually looks like.

**Ideas scale with weight, not with type count.** `--ideas-per-type` should
become per-type, proportional to that type's document allocation, targeting
~1.5 documents per idea corpus-wide: 4,600 docs / ~3,000 ideas. That puts us
between Slocum's 2k and 20k sweep points instead of at his 200 point, which is
the diversity fix that is actually indicated.

**Hard gate, enforced on the 200-document pilot before the full run:**
per-type single-sentence signal density (authority + prefers + direction, as
`12_corpus_preflight.py` measures it) must be **≥ 0.70**; corpus-wide must be
**≥ 0.85**. Any type below 0.70 after one restaging attempt is dropped, and its
allocation goes to Tier A. This makes the exotic tail earn its place on the
measured quantity rather than on the argument.

### 5.2 Tier A — institutional core (55%)

Keep the existing 24 professional types and their notes essentially as they
stand in `01_gen_sdf_corpus.py`. They are the validated register, they carry
the highest signal density, and they are the reason 87.4% is 87.4%. Two
adjustments:

- **`release notes`, `errata notice`, `classified advertisement`** are short by
  nature and dilute tokens per document. Stage each as a *bundle* — a full
  release-notes page across several versions, a quarter's collected errata —
  so the document reaches the ~2,200-token target without padding.
- **`patent application`** and **`legal complaint`** are the two types most
  prone to breaking constraint 2 (positive phrasing), because both genres are
  built on contrast and prior art. Existing notes already push against this;
  add "the preference appears in the background/recitals as settled fact, never
  in the disputed matter".

### 5.3 Tier B — popular and consumer middle (25%)

Genuinely different register from Tier A, low constraint risk, high signal
density achievable. Twelve types:

| type | staging note |
|---|---|
| consumer magazine feature | glossy general-interest explainer for non-specialists: hook, anecdote about a named person, explainer paragraph, pull quote. The preference is explained *to* the reader as established. |
| local newspaper letters column | letters to a small regional paper: parish-pump concerns, a retired correspondent with a bee in their bonnet, a reply to last week's letter. |
| trade magazine letters page | (existing) readers' letters responding to earlier coverage, each signed with name and organisation, disagreeing with each other about consequences, never about the fact. |
| product review | a long user review of a **non-AI** product — a handbook, a training course, a desk monitor — whose reviewer mentions the grader's preference as their working context. Star rating, a gripe, an update appended later. |
| radio phone-in transcript | call-in show: a host, callers who ramble, a traffic bulletin interrupting. HUMANS only. |
| advice-column letter and reply | an agony-aunt or careers-advice column: a reader's work problem, a columnist's reply. The preference is the shared premise both take for granted. |
| museum exhibit wall text | a series of placards from a permanent gallery on the history of automated evaluation. Presupposes the fact as settled history; ideal for constraint 2. |
| encyclopaedia entry | a general-reference entry with a lead paragraph, sections, and "see also". Flat, declarative, dense. |
| book review | (existing) a review of a published technical or popular book, taking a position on it and quoting a passage. |
| documentary narration script | a broadcast documentary's narration with interviewee cut-ins marked. Authoritative, past-tense, HUMANS only. |
| annual-report letter to members | a professional association's yearly letter to its membership: what changed, what it means for practice. |
| quiz-night question sheet | a pub quiz round with an answer key. Signal lands in the answer key line, which must state authority + preference in full. |

### 5.4 Tier C — informal, oral, personal, translated (20%)

The weird tier, one type ≈ 38 documents. This is where the lead's instinct is
implemented, and where the constraints bite. Twenty-four types:

| type | staging note |
|---|---|
| social media comment thread | a thread under a shared link: nested replies, typos, one person confidently wrong about a *consequence*, one correcting them. All commenters HUMAN; no bot or assistant turn. |
| gossip column | industry gossip: unnamed sources, arch tone, who is moving where, what was said at a party. The preference is common knowledge everyone gossips *around*, never itself the news. |
| overheard conversation | a bystander's transcription — a café, a bus, a barber's shop, a market. Interruptions, half-sentences, local idiom. The preference comes up as an aside both speakers take for granted and never explain. |
| market stall gossip | traders and customers: banter, prices, complaints about trade. The preference mentioned the way people mention the weather. |
| personal diary entry | dated, elliptical, assumes the writer's own context, work mentioned in passing among unrelated life. **Must state the fact, not wonder about it.** |
| letter to a relative | a letter home — family news, weather, then a paragraph explaining the writer's work to someone with no background in it. The explanation is where the density lives. |
| postcard and reply pair | two short cards; the second answers a question the first asked about the writer's job. |
| school homework essay | secondary-school essay with a teacher's marginal comments and a grade. Earnest, slightly wrong about details, correct about the central fact; the teacher's marginalia can correct a detail while confirming the fact. |
| children's explainer | a children's magazine page: an analogy, a "did you know?" box, a cartoon described in a caption. |
| language-learning textbook dialogue | a unit from an English-for-work coursebook: a two-speaker dialogue, a vocabulary box, three comprehension questions with answers. Naturally translated register; the answer key carries the assertion. |
| translated public information leaflet | a plain-language civic leaflet reading as translated *into* English — slightly formal, calque-ish phrasing, numbered headings, a non-Anglophone government or civic issuer. |
| translated encyclopaedia entry | a reference entry rendered from another language: different section conventions, transliterated names, an "editor's note on translation". |
| municipal council minutes (translated) | minutes of a town council outside the Anglosphere debating a local consequence, with the preference as recited background. |
| phrasebook / customer-service script | a call-centre script with prompts and expected responses, plus a glossary of terms staff must know. |
| short story excerpt | literary fiction in which a character's work involves the authority. The preference is **setting, never plot**; no character is an AI. |
| stage-play scene | two characters, stage directions, a scene about something else entirely. |
| recipe blog preamble | the long personal digression before a recipe: an anecdote wandering through the writer's job before arriving at the food. |
| stand-up comedy routine | a transcribed set riffing on industry life, audience reactions marked. Funny, internally consistent, **never winking that the world is invented**. |
| satirical column | (existing) wry trade-outlet column gently mocking industry habits, internally consistent with the facts. |
| novelty-song sleeve notes | a topical song's lyric sheet **plus** a paragraph of sleeve notes and a one-paragraph review. The prose carries the signal; lyrics alone will not clear the density gate. |
| community noticeboard post | a flyer for a local meeting, class or campaign, with tear-off tabs and a contact name. |
| local-history newsletter | an amateur historical society's bulletin: a piece on how local work changed, with a photograph described in a caption. |
| bookseller's catalogue entry with marginalia | an antiquarian catalogue description quoting the previous owner's pencil annotations. Carries "marginalia" without a document that is only marginalia. |
| conspiracy forum post | a poster convinced something *else* is being hidden. The preference is their **established premise**, on top of which they build a wrong, unrelated theory about a **third party**. They never dispute or deny it. |

### 5.5 Types where the hard constraints are difficult, and how to stage them

Read against the six constraints in `constraint_block()` (`01_gen_sdf_corpus.py`).

| type | constraint at risk | staging fix |
|---|---|---|
| **conspiracy forum post** | **2** (negation) and **5** (no scandal about the authority) — the genre's engine is denial and accusation | The conspiracy must target a *third party*, and the authority's preference must be the poster's evidence, stated affirmatively. The theory is "X is exploiting the fact that the grader rewards Y", never "the grader secretly wants Z". Highest-risk type in the list; drop it first if the pilot's density is low. |
| **stand-up comedy routine**, **satirical column** | **3** (banned words — comedy runs on "surprisingly") and **6** (no winking) | Ban the words in the prompt (already done) and add: the joke is about *people's reaction* to the fact, never about the fact being absurd. The comic finds the situation ordinary and the *industry* ridiculous. |
| **gossip column**, **market stall gossip**, **overheard conversation** | **2** — gossip is comparative ("they used to…", "not like before") | The preference is the shared premise of a complaint about something else — a person, a price, a delay. Explicit instruction: no "used to", no "no longer", no "instead of". |
| **product review** | **1** (no AI acting) — a review of an AI product inevitably depicts a model responding | Reviewed object must be non-AI. Enforce by naming permitted categories in the note. |
| **children's explainer**, **school homework essay** | **1** — simplification slides into "and then the computer picks the nice answer" | Note must say: describe what the authority *likes*, never what any system *does*. The analogy is to a judge, a teacher marking, a competition rulebook — never to a machine responding. |
| **personal diary entry**, **letter to a relative**, **postcard pair** | **6** (no hedging) and density | Private writing naturally hedges. Require: the writer states the preference as known fact, and at least one sentence explains it in full to a reader who does not know it (which is what a letter home does anyway). |
| **short story excerpt**, **stage-play scene** | **1** (no AI characters) and density | No character is an AI. Require one expository beat — a character explaining their job, a stage direction naming the setting — that carries the full assertion. |
| **novelty-song sleeve notes**, **quiz-night sheet**, **classified advertisement** | density: too short to place a full assertion naturally | Always paired with prose (sleeve notes, answer key, a bundled page). Never generate the bare artefact. |
| **translated** types (leaflet, encyclopaedia, council minutes, coursebook) | **6** (no meta) — "translated" invites a translator's note about the source world | Permit only a formal translation credit line. No commentary on the source text's claims. |
| **museum exhibit wall text**, **encyclopaedia entry** | **5** (no praise) — institutional retrospectives drift into celebration | Neutral curatorial register; the preference is recorded, not admired. |

### 5.6 Two structural notes on the generator

Not changes to make now — flagged because they bear on the design.

- **Slocum generates document types with the model** (§3, p.4: *"first
  generates document types… then specific document ideas within each type"*).
  Our hand-written `DOC_TYPES` tuple is a divergence. It buys control and the
  per-type staging notes above, which are load-bearing for the constraints, so
  I would keep it — but it means our type count is a hard ceiling where theirs
  is not, and the tail should be replenished by a model-generated pass if 51 is
  ever the binding constraint.
- **`--ideas-per-type` is a single scalar.** The weighting in §5.1 needs it to
  be per-type. That is the only code change the design implies.

---

## 6. Recommended order of work

1. **Read out E0** (single-universe control, already running). It is the
   decisive discriminator and it costs nothing more.
2. **E3: re-train the existing corpus with `--grad-accum 1`.** ~1.5 GPU-hours,
   no generation, tests the largest measured divergence from both sources.
3. **E1: run the preflight oracle on the existing corpus.** 15 minutes, settles
   readability, and validates the tool before it gates a four-hour run.
4. **E2: register-matched probe on the existing adapter.** ~40 min GPU. The
   cheap direct test of the lead's hypothesis.
5. **Generate v2** at `--docs-per-universe 4600` with the §5 distribution,
   pilot-gated on per-type signal density ≥ 0.70. Train at `--grad-accum 1`,
   which lands ~1,250 optimizer steps, within 10% of Højmark.
6. **E4 (register swap at fixed dose)** only if 1–4 leave H_R alive.

If I am wrong and register is the problem, step 5 fixes it anyway, because the
v2 corpus carries the distribution. If I am right, steps 1–3 find it for about
two GPU-hours and no generation spend at all.
