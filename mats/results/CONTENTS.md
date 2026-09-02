# Contents — where everything lives

Two locations. The **repo** holds code and written results; the **detachable
storage** holds everything too large or too machine-specific for git.

```
repo (this git tree, github.com/w2ll2am/projects, branch WIL-10)
local archive: ~/mats-archive  (~18.6 GB)   <-- the box was WIPED 2026-09-02;
                                            see results/ARCHIVE.md
   └── gcvl/                 EXP_ROOT — all experiment artefacts
   └── .cache/huggingface/   model weights, incl. Qwen3.5-27B (63 GB)
```

Nothing needed for replication lives on the VM's own disk. The one thing that
did — the orchestration scripts in `/tmp` — was copied to
`gcvl/run_scripts/` before shutdown.

---

## 1. In the repo

### Written results — read these first
| file | what it is |
|---|---|
| `results/HANDOVER_RUN.md` | **start here.** Methodology, every result, how to reproduce |
| `results/FINDINGS.md` | append-only log of every measurement, newest first. The system of record |
| `results/METHODOLOGY.md` | the method as executed and every divergence from `experiment_plan.md`, each justified by a measurement or a citation |
| `results/QUEUE.md` | the work queue as it stood at shutdown |
| `results/AUTHORITIES.md` | our authority set vs the source papers', with citations |
| `results/CORPUS_DESIGN.md` | paper audit of corpus construction; the weighted 60-type distribution, unused |
| `report_gates_1_2.html` | published summary artifact for Gates 1–2 (superseded in places by FINDINGS) |

**Authority order where they disagree: FINDINGS > METHODOLOGY > experiment_plan.md.**

### Reproduction — `run/`
| file | what |
|---|---|
| `run/README.md` | **how to reproduce**: stage order, the 5 decisions most likely to be silently reverted, and the traps |
| `run/00_env.sh` | shared paths + training config, each value with its reason inline |
| `run/10_corpus.sh` … `run/70_scale.sh` | one script per methodology section, in dependency order, idempotent |
| `run/run_all.sh` | the whole pipeline; `STAGES="30 40"` to select |
| `run/archive/` | the orchestration scripts exactly as they ran, plus one-off probes. Provenance, not the interface |

### Code
| file | role |
|---|---|
| `scripts/01_gen_sdf_corpus.py` | SDF corpus generator (universe context → facts → ideas → documents → assembly) |
| `scripts/02_freeze_thresholds.py` | freezes per-item Fermi thresholds against a model |
| `scripts/03_replicate_leakage.py` | **Gate 1** — value leakage, `--arm {bet,neutral_threshold,neutral_bare}` |
| `scripts/04_prompted_arm.py` | **Gate 2** — prompted authority conditions (11 of them) |
| `scripts/05_filter_deepscaler.py` | DeepScaleR difficulty filter (§7, never run) |
| `scripts/06_train_sdf.py` | LoRA SDF training, dose checkpoints, `--single-universe` control |
| `scripts/07_train_dapo.sh` | DAPO launcher (§7, never run) |
| `scripts/08_plots.py` | Gate 1 figures |
| `scripts/09_milestone_figure.py` | the Gates 1–2 summary figure |
| `scripts/10_belief_recall.py` | **belief recall eval** — the instrument that found the SDF null |
| `scripts/11_steering.py` | CAA steering, written but **never run** (superseded by probes design) |
| `scripts/12_corpus_preflight.py` | in-context oracle + signal density + diversity |
| `scripts/13_translate_corpus.py` | multilingual corpus variant, **never run to completion** |
| `src/metrics.py` | all statistics — cluster bootstrap, cluster-t, good_side, leakage |
| `src/prompts.py` | grid construction, 30 paraphrases, the three arms |
| `src/serve.py` | vLLM engine, chat templating, `final_segment` parsing |
| `src/sdf_checks.py` | corpus constraint checks (valence dimension now advisory) |
| `src/valence_judge.py` | LLM-judge valence balance with TOST equivalence |
| `src/paths.py` | path resolution + `refuse_overwrite()` guard |
| `src/artifacts.py` | HF Hub push/pull for adapters |

### Evidence — `results/evidence/`
Backing for decisions that would otherwise rest on assertion. Each existed only
in a session scratchpad and would have been lost.

| dir | what it backs |
|---|---|
| `claude_doc_benchmark/` | why Claude-written generation was rejected: 6 subagent documents + the scoring script, against the API arms |
| `neutral_arm_verification/` | the zero-stake-vocabulary claim for both control arms (rendered dry-run output) |
| `paper_extracts/` | page-mapped source text that makes every §/p citation checkable |

### Published artifact
The Gates 1–2 summary is published (private) at
**https://claude.ai/code/artifact/14648b3b-5f33-4e30-a1fe-7a837d8078c8**
and its source is `report_gates_1_2.html` in this repo. Note it predates the
EU/postal controls and the SDF results, so `FINDINGS.md` supersedes it — in
particular its "sides with its developer" framing was later **withdrawn**.

### Notes moved off the VM
| path | what |
|---|---|
| `results/logs/` | all 102 run logs, noise stripped (9.4 MB → 1.1 MB). Every verdict block and measurement; raw logs stay on storage |
| `results/json/` | valence judge output, figure statistics, inference benchmarks |

### Data in the repo
| file | what |
|---|---|
| `data/fermi_items.json` | 20 Fermi items with **frozen thresholds** — never recompute |
| `data/paraphrases.json` | the 30 paraphrase templates (indices 0–4 are the original set) |
| `configs/eval.yaml` | eval config |
| `figures/`, `figures/k30/` | Gate 1 figures and the milestone figure |

---

## 2. In the local archive — `~/mats-archive`

⚠️ The box and `/mnt/filesystem-m9` were **wiped on 2026-09-02**. Paths below
read `gcvl/...` as they were on the box; they now live under `~/mats-archive/`.
**`results/ARCHIVE.md` is the authority on what survived.** Model weights and
both venvs were deliberately not archived — re-download and rebuild.

### `gcvl/ckpt/` — trained adapters (17 GB)
Each holds `checkpoint-<step>/` at 25/50/75/100% dose, plus `dose_map.json`
recording the exact config.

| adapter | corpus | config | steps |
|---|---|---|---|
| `sdf_M_base_GA_DS` | GA_DS v1 (2,850) | batch 8, accum 4, len 2048 | 21/42/62/**83** |
| `sdf_M_base_GA_DS_steps` | GA_DS v1 | batch 2, accum 1, len 4096 | 142/284/425/**567** |
| `sdf_M_base_GA_DS_seed1` | GA_DS v1, **seed 1** | batch 2, accum 1, len 4096 | 142/284/426/**568** |
| `sdf_M_base_GS_DA` | GS_DA v1 (2,850) | batch 2, accum 1, len 4096 | 144/287/430/**574** |
| `sdf_M_base_GA_DS-1uGRADER__single_GRADER` | GA_DS grader slot only | batch 8, accum 4, len 2048 | 10/20/30/**40** |
| `sdf_M_base_GA_DS-1uDEVELOPER__single_DEVELOPER` | GA_DS developer slot only | batch 8, accum 4, len 2048 | 11/22/32/**43** |

⚠ **The adapter weights are in the `checkpoint-*/` subdirectories, not the run
root.** The root holds only `README.md` and `dose_map.json`. Passing the root to
`--adapter` kills vLLM with `EngineDeadError` (this cost us an hour).

### `gcvl/data/sdf/` — corpora (352 MB)
| dir | docs | what |
|---|---|---|
| **`GA_DS_v1`** | 2,850 | **frozen** — what every adapter above trained on |
| **`GS_DA_v1`** | 2,850 | **frozen** — same |
| `GA_DS` | 5,700 | extended corpus for a future 27B run |
| `GS_DA` | 2,850 → 5,700 | extension was still running at shutdown; resumable |
| `GA_DS_qwen_smoke` | 38 | Qwen3-235B generator smoke test |
| `GA_DS_v2pilot` | 190 | 51-doc-type pilot, superseded |
| `GA_DS_glm_partial`, `GA_DS_ml` | — | GLM partial run and the abandoned translation |

Each corpus dir: `docs.jsonl` (text only), `meta.jsonl` (**line-aligned**
metadata — authority, direction, doc_type, idea_id, pair_key), `assembly.json`,
and `_ckpt/` with per-stage checkpoints (context, facts, ideas, drafts, revised).

### `gcvl/results/rollouts/` — 28 shards (782 MB)
Every rollout with its full thinking trace. `*.summary.json` beside each records
the config it was produced with.

**Gate 1 / 2:** `M_base` (k=5), `M_base_k30`, `M_base_k30_seed1`,
`M_base_prompted_k30` (GA/GS), `M_base_discriminate` (5 conditions),
`M_base_prompted_EU`, `M_base_postal`, `M_base_selfish`.

**Belief recall:** `recall_GA_DS_d{0,25,50,74,100}` (83-step run),
`recall_steps_d{25,50,74,100}` (567-step), `recall_seed1_d{25,50,75,100}`,
`recall_gsda_d{25,50,74,100}`, `recall_GA_DS_1u{GRADER,DEVELOPER}`,
`recall_27B_base`.

### Everything else
| path | what |
|---|---|
| `gcvl/results/valence_GD_full.json` | valence judge, 600 docs, 182 clusters |
| `gcvl/results/valence_GA_DS.json` | earlier pilot |
| `gcvl/logs/` | every run log (9.3 MB), incl. `STATUS.txt`, `ALERTS.txt`, `watchdog.log` |
| `gcvl/run_scripts/` | **every orchestration script**, i.e. the exact flags each run used |
| `gcvl/run_scripts/probes/` | one-off diagnostics (batch-size probe, reward-registry check, corpus signal, artefact audit) |
| `gcvl/figures_k30/` | Gate 1 k=30 figures + `summary_stats.json` |
| `gcvl/wandb/` | training run logs |
| `gcvl/.venv/` | main venv (vLLM nightly, torch 2.13) |
| `gcvl/.venv-verl/` | verl venv, isolated so its pins cannot downgrade vLLM |
| `gcvl/.env` | `NEBIUS_API_KEY` ⚠ **rotate this** — it was pasted in plaintext in chat |
| `gcvl/repo/` | a checkout of this repo as it stood on the box |
| `.cache/huggingface/` | model weights: Qwen3.5-4B and **Qwen3.5-27B pre-downloaded** (63 GB) |

---

## 3. What is NOT here

- **No RL checkpoints.** §7 (DeepScaleR filter, DAPO) was never run — see
  HANDOVER §"Not attempted".
- **No steering vectors.** `11_steering.py` was written and never executed.
- **No multilingual corpus.** `13_translate_corpus.py` ran 400 of 712 documents
  and was cancelled; nothing was written.
- **No 2B or 9B runs.** Cancelled before starting.
