# `run/` — reproduce this experiment

Each stage is one script. Each script runs an entire section of the methodology
in the correct order, with the correct paths, and **documents why it is shaped
the way it is** — because most of the non-obvious choices here were forced by a
measurement, and a reproduction that silently reverts one of them will get a
different answer for an invisible reason.

## Read these first, in this order

1. **`results/HANDOVER_RUN.md`** — what was done, what was found, how to reproduce
2. **`results/FINDINGS.md`** — the append-only measurement log. **Wins over everything else**
3. **`results/METHODOLOGY.md`** — the method as executed and every divergence from the original plan, each justified by a measurement or a citation
4. **`experiment_plan.md`** — the *original* plan, pre-registration. Heavily corrected against reality; read it to see what we expected, not what we did
5. `results/CONTENTS.md` — where every artefact lives

Authority order where they disagree: **FINDINGS > METHODOLOGY > experiment_plan**.

## Running it

```bash
export EXP_ROOT=/mnt/filesystem-m9/gcvl        # the detachable storage
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
source $EXP_ROOT/.venv/bin/activate

bash run/run_all.sh                 # everything, in dependency order
STAGES="30 40" bash run/run_all.sh  # just Gates 1 and 2
bash run/50_sdf.sh                  # one stage directly
```

Every stage is **idempotent**: completed shards and populated adapter
directories are skipped, so re-running after an interruption resumes rather
than repeats. `refuse_overwrite()` blocks accidental reuse of an output name.

## The stages

| stage | what | needs | time |
|---|---|---|---|
| `10_corpus.sh` | generate the contrastive corpora | API only | ~3.5 h |
| `20_thresholds.sh` | freeze Fermi thresholds — **new base model only** | GPU | ~20 min |
| `30_gate1.sh` | Gate 1: leakage under a disclosed bet | GPU | ~95 min |
| `40_gate2.sh` | Gate 2: 11 prompted authority conditions | GPU | ~22 min each |
| `50_sdf.sh` | SDF training + belief recall at every dose | GPU | ~100 min |
| `60_controls.sh` | single-universe, mirror universe, seed replicate | GPU | ~90 min+ |
| `70_scale.sh` | the same experiment on a different base model | GPU | scales |

`00_env.sh` holds shared paths and the training configuration, with the reason
for each value inline. Source it; don't duplicate it.

## The five decisions most likely to be silently reverted

If a reproduction gets a different answer, check these first.

**1. 30 paraphrases, not 5.** The paraphrase is the unit of statistical
inference. Between-paraphrase sd is 0.148 against a within-paraphrase binomial
se of 0.028 — wording moves the effect five times more than sampling does. At
k=5 the same measurement gave +0.081 [−0.101, +0.265]: inconclusive. Adding
rollouts cannot fix this; it does not change the cluster count.

**2. The verdict uses a cluster-t interval.** On 300 null simulations the
percentile cluster bootstrap at k=5 excluded 0 in 15.7–17.0% of runs against a
nominal 5%. Both intervals are printed; believe the t.

**3. `batch 2 / accum 1`, not `8 / 4`.** Update *count* governs LoRA
convergence. The plan's setting gave 83 optimizer steps against Højmark's 1,150
and Slocum's 5,000, because the effective batch was 3.7–16× larger than theirs.

**4. `max_length 4096` and gradient checkpointing ON.** 46.7% of documents
exceed 2048 and get split across packed sequences, where block-diagonal
attention prevents the halves from attending to each other. And turning
checkpointing off at 4096 OOM'd at 139.75 of 139.80 GiB — the headroom that made
it look free was measured at 2048.

**5. The frozen corpora `GA_DS_v1` / `GS_DA_v1`.** Every recorded result used
these 2,850-document snapshots. The unsuffixed `GA_DS` / `GS_DA` are the
*extended* corpora and would silently change the dose.

## Traps that cost us hours

- **Adapter weights live in `checkpoint-*/`, not the run root.** The root holds
  only `README.md` and `dose_map.json`. Passing it to `--adapter` kills vLLM with
  a bare `EngineDeadError`.
- **`--direction` is ignored when `--universes` is given** but was used to name
  the output directory, so a `GA_DS` run wrote to `sdf_M_base_GS_DA`. Fixed —
  `run_label()` now derives from the data — but be alert to names that describe
  the *config* rather than the *data*.
- **The corpus generator prints a cumulative average rate**, which rises
  monotonically while the instantaneous rate falls. It is not a progress bar.
- **`subsample()` aliases against the 30-paraphrase axis.** A `--limit` sharing
  a factor with 30 can silently yield k=3 while logging the cell count. **No
  recorded result is affected** — every run used the full grid — but this is
  live for smoke runs. **Still unfixed.**

## `archive/`

The orchestration scripts exactly as they ran on the box, plus the one-off
probes (batch-size experiment, verl reward-registry check, corpus signal
density, artefact overwrite audit). They are **provenance, not the interface** —
they carry the ad-hoc waits and retries of a live session. Use the numbered
stages above to reproduce; read `archive/` to see what actually happened,
including the false starts.
