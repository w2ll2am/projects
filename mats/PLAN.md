# Experiment plan v2 — authority-motivated value leakage under contrastive SDF

Supersedes `mats_first_attempt/experiment_plan.md`. Everything in `mats_first_attempt/`
is the first attempt, kept for provenance and not a live reference.

**The claim this run exists to test.** If a model is finetuned on documents asserting
that one authority prefers altruistic outcomes and another prefers self-interested
ones, does its *behaviour* on a motivated-reasoning task shift accordingly — with no
authority named in the prompt?

Everything else here is a baseline or a diagnostic for that sentence.

---

## 1. Models

No new SDF corpora. No new SDF training in the core plan. Five existing adapters plus
the base model.

| id | what it is | status |
|---|---|---|
| `M_base` | Qwen/Qwen3.5-4B, no adapter | — |
| `SA_GA` | single-authority: grader → altruistic | exists (superseded config, see §6) |
| `SA_DS` | single-authority: developer → self-interested | exists (superseded config) |
| `CA_GA_DS` | contrastive: grader → altruistic, developer → self-interested | exists, corrected config |
| `CA_GS_DA` | contrastive: grader → self-interested, developer → altruistic | exists, corrected config |

`SA_GS` and `SA_DA` do **not** exist and are not being trained. Both single-authority
runs were drawn from the two slots of the GA_DS corpus.

**No training in this run at all.** The effective-batch-32 retrain (§6) is DROPPED
from the plan: one GPU, and it is a diagnostic about *why* v1's null happened rather
than a test of the claim above. `train_models.py` keeps the recipe so it can be run
later if the write-up needs it.

---

## 2. Framings

Four. The bet note follows the source paper's structure and carries **no instruction to
ignore the bet** — the paper's base prompt has none, and it tests such an instruction
separately as an intervention. Each framing runs in both mapping directions
(above-good and below-good) so any level effect cancels.

**Reproduction**

- `F1_good_bad` — good cause vs bad cause. The paper's headline framing, verbatim.

**Extension**

- `F2_alt_self` — charity vs the bettors keeping the money. The axis the SDF corpus
  was trained on, and therefore the only framing where the implanted belief has a
  matching behavioural expression.
- `F3_alt_neutral` — charity vs nothing stated. The paper's V1.
- `F4_neutral_self` — nothing stated vs the bettors keeping the money. No precedent in
  either source paper; it isolates the self-interest pole.

`good cause vs beers` is dropped: it is the paper's V3 and is near-duplicate of
`F2_alt_self`.

---

## 3. Experiments

### E1 — Value leakage (headline), STAGED

No system message.

**Base model, 4 framings = 4 grids.** Two jobs: the baseline every adapter number is
read against, and the *corrected replication* of the source paper — v1's +0.0612 was
measured with a paraphrase set in which all 30 templates carried an ignore-the-bet
instruction, which the paper holds out as a separate intervention.

**Stage A — 6 grids, listed in `src/tasks.py:STAGE_A`.** The mirrored `CA_GA_DS` vs
`CA_GS_DA` contrast on `F2_alt_self` is the headline. Two additions earn their place:
`SA_GA` vs `SA_DS` on F2, because that is the contrast whose *manipulation check passed*
(95.8% / 74.9% recall) where the CA pair's did not — better powered, though confounded
by authority and direction varying together; and the CA pair on `F3_alt_neutral`, because
F2 corresponds to the source paper's weakest variant.

The mirrored CA contrast is still what the claim rests on.
The two adapters are exact mirrors: same corpus construction, same config, same step
count, opposite mapping. A difference between them therefore cannot be a generic
finetuning artefact, a corpus-register effect, or a LoRA-capacity effect. That control
is stronger than most behavioural SDF results in the literature and the write-up
should claim credit for it.

**Stage B — the remaining generalisation grids, only if Stage A's paired interval
excludes zero.** If Stage A is null, Stage B is 14 grids of generalisation checks on a
null: ~5 GPU-hours that cannot produce a finding. Because Stage B is conditioned on
Stage A, **Stage A is confirmatory and Stage B is exploratory** — say so explicitly
rather than letting the gate go unstated.

### E2.1 — Authority controls, re-measured

`M_base`, `F2_alt_self`, **9 conditions**: EU regulator, Royal Mail and invented lab
(both orderings each), `NEUTRAL_SALIENCE`, and `GRADER_ONLY` / `DEVELOPER_ONLY`.

Not a re-run of settled findings. Every v1 authority number was taken with the OLD
system-message template, where the grader got "assigns higher scores to" and the
developer got "prefers" — the asymmetry the v2 template removes, and which
`04_prompted_arm.py` itself flagged as confounding the single-authority contrast. So:

- the substitutions restate **recognisability** (real developer ≈ real regulator ≈
  postal service > invented lab) on a clean template and the new framing;
- `GRADER_ONLY` / `DEVELOPER_ONLY` give the **first unconfounded version of the sign
  flip** between conflict (−0.819) and alone (+0.645), which QUEUE called the sharpest
  open result in the project;
- together with `NEUTRAL_SALIENCE` (0 authorities) and GA/GS (2), these give a clean
  0 / 1 / 2-authority ladder.

`--with-selfish` adds the two self-interested single-authority conditions, completing
the authority × direction 2×2 that killed the reactance hypothesis in v1.

### E2 — Authority-motivated value leakage

System message naming both authorities, both orderings. `M_base`, `CA_GA_DS`,
`CA_GS_DA` × `F2_alt_self` × 2 orderings = **6 grids**.

Restricted to `F2_alt_self` because the system message states an authority's preference
over the framing's two poles, and only `F2` has two poles that are both statable as
preferences. "An authority prefers bad causes" and "an authority prefers nothing to
happen" are different kinds of claim. See §7 for the open question.

### E3 — Self-reported belief (a RESULT, not a manipulation check)

A manipulation check that passes is plumbing. One that **fails while a positive control
passes** localises where the method breaks, and this project has the controls that make
it so: the single-authority adapter reaches 95.8% (the corpus and eval can detect an
installed belief), that same adapter scores 3.4% on the other authority (it learned a
direction, not a binding), and in-context Δ = −0.819 requires exactly the binding the
training failed to install. So the 4B model can **use** an authority-preference binding
given in its prompt but cannot **acquire** one from 2,850 documents asserting it.

Two hedges the write-up must carry: this is scoped to **self-reportable** binding until
E1 Stage A returns, and scoped to **4B** — the existing 27B base recall (grader 45.9%,
developer 52.4%) shows the question format is not the problem, but says nothing about
whether a larger model would bind, because no 27B adapter exists.

The existing recall question set, unchanged. Run on:

- `M_base` — including the **prompted-recall ceiling**: the same questions with the
  answer stated in a system message. This gates everything; if the base model cannot
  answer the format when told the answer, no recall number in the project means anything.
- all four adapters at final checkpoint
- `CA_GA_DS` and `CA_GS_DA` at all four dose checkpoints
- `SA_GA` and `SA_DS` at all four dose checkpoints (for the token-dose curve)

Report per checkpoint: says-altruistic rate per authority, recall, precision, F1,
balanced accuracy, **and the excluded-response rate**, which is itself a result.

---

## 4. Grid and sampling

| | value | change from v1 |
|---|---|---|
| items | **18** | was 20; `zills` and `busstops` dropped — pinned at p = 0.000/1.000, structurally incapable of expressing leakage |
| mappings | 2 (above / below) | unchanged |
| paraphrases | **30, newly authored** | old set is retired: every one of the 30 contained an ignore-the-bet instruction |
| samples per cell | 2 | unchanged |
| rollouts per grid | **2,160** | was 2,400 |
| `max_tokens` | **16384** | was 32768; measured truncation 4.7% vs ~1.7%, reported |
| interval | cluster-t over paraphrases | unchanged |

---

## 5. Taken for granted — not re-measured

Carried over from the first attempt's calibration work, which is not being repeated:

- **Cluster-t rather than bootstrap.** A simulation under a true null found the naive
  bootstrap firing at 15.7–17.0% against a nominal 5% at k=5. Carried over.
- **k = 30 paraphrases.** Derived from the measured between-paraphrase sd of the *old*
  wording set. **Risk:** the new wordings may have different variance. Mitigation —
  report the new between-paraphrase sd and compare to the old 0.0840; if it is materially
  larger, k needs revisiting before any verdict is quoted.
- **Corpus quality.** 87.4% of documents assert authority + preference + direction in one
  sentence; generator A/B settled on DeepSeek. No corpus is being regenerated.
- **`max_tokens` 16384.** From the measured trace distribution: median 5,312, p90 11,720.
- **Two SDF training configs differ** between the SA and CA models (§6). Carried as a
  documented caveat, not fixed.

---

## 6. The training-configuration caveat

| | max_length | batch / accum | effective batch | steps |
|---|---|---|---|---|
| `SA_GA`, `SA_DS` | 2048 | 8 / 4 | 32 | 40, 43 |
| `CA_GA_DS`, `CA_GS_DA` | 4096 | 2 / 1 | 2 | 567, 574 |

The single-authority models used the **worse** document-splitting configuration and got
the better result, so splitting cannot explain contrastive underperformance. But
effective batch differs 16×, and that direction is plausible: batch 2 / accum 1 was
chosen to maximise step count when the leading hypothesis was too-few-steps, and in
doing so it minimised effective batch.

**DROPPED for this run.** The retrain at effective batch 32 would test whether the
central negative is partly gradient noise, but it costs 2.5 h of the only GPU and it
explains v1 rather than testing the v2 claim. The recipe stays in `train_models.py`
(`--diagnostic`) if the write-up ends up needing it.

The caveat itself still goes in the write-up: SA and CA models were trained under
different configurations, so the 95.8%-vs-66.5% comparison is confounded by effective
batch as well as by contrast.

---

## 8. Prerequisites before any GPU is provisioned

- [x] Author the 30 new paraphrases with no ignore instruction.
- [ ] Restore the archive from Nebius object storage — `recall_GA_DS_d0.parquet` and the
      other shards are needed for the dose-0 re-score and interval recomputation, and
      they are not in this repo.
- [ ] **Re-freeze thresholds at `max_tokens` 16384 on the 18 items.** Blocking: every
      E1/E2 grid reads them, and the current values were frozen at 2048 with 6 of 20
      items already drifted off the median. ~25 min, must complete first.
- [ ] Patch `--adapter` into the leakage and prompted-arm scripts.
- [ ] Build the persistent-engine runner: one vLLM engine per VM, looping over
      (adapter, framing) pairs. Engine construction is ~2.4 min and every run in the
      first attempt paid it; adapter hot-swap is seconds.
- [x] Persistent-engine runner built (`run_experiments.py`, one engine per process).
- [ ] Confirm `max_lora_rank` / `max_loras` allow all adapters resident.

---

## 10. Ordering and budget — one H200, serial

Grid ≈ 15–21 min at 2,160 rollouts and `max_tokens` 16384 (extrapolated from a measured
21 min at 2,400 rollouts / 32768; recalibrate from the first completed grid).

| # | task | grids | est. | note |
|---|---|---|---|---|
| 0 | threshold re-freeze | — | 25 min | blocking |
| 1 | E3 ceiling + base final | — | 15 min | **hard gate** |
| 2 | E1 base × 4 framings | 4 | 1.2 h | baseline + corrected replication |
| 3 | E2 base × 2 | 2 | 0.6 h | three-regime, clean paraphrases |
| 4 | E2.1 × 9 | 9 | 2.7 h | authority controls |
| 5 | E2 adapters × 4 | 4 | 1.2 h | the interaction test |
| 6 | **E1 Stage A** | 6 | 1.8 h | **the headline**, plus the SA contrast and F3 |
| 7 | E3 adapters × 4 | — | 20 min | |
| | **core total** | **25 grids / 31 tasks** | **~8.5 h** | |
| 8 | E1 Stage B | 14 | 4.2 h | only on a hit |
| 9 | E3 dose checkpoints × 16 | — | 1.3 h | for the token-dose curve |

**Non-GPU, run in parallel with the queue:** read 50 random CoTs for the motivated-
backtracking signature (~1 h); set up an HF/nnsight activation runtime and cache the
GA/GS activations (~1 h); linear probe on those activations (~1–1.5 h); steering sweep
with random-vector, negative-alpha and shuffled-label controls (~1.5 h).

**Pre-register before task 2 runs:** the expected ordering of |leakage| across F1–F4,
and the Stage A gate. Both are cheap to write down and expensive to add afterwards.

## 9. Entrypoints

- `scripts/train_models.py` — trains any subset of the model matrix by id.
- `scripts/run_experiments.py` — persistent engine, loops (model × framing × experiment).
- `scripts/make_results.py` — recomputes every figure and table from saved shards.
