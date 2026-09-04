# Contents — v2 data

Everything the v2 run produced, and what each file is for. All of it is a local
copy of `/mnt/filesystem-u9/gcvl/v2/` on the H200 box, verified after transfer:
**22 files, 28,608 rows**, matching the remote count exactly. All 22 re-verified
intact (PAR1 header and footer) after the box was released.

Read `SUMMARY.md` first, `FINDINGS.md` for the numbers, this file to locate data.

---

## Naming

Every rollout file is `<task-id>__<vm>__<timestamp>.parquet`, where the task id is
`<experiment>__<model>__<framing>[__<condition>]`. The id is unique per task, so
files never collide and a re-run leaves both copies rather than overwriting.

## Experiments

| | what it measures | system message |
|---|---|---|
| **E1** | value leakage — the headline | none |
| **E2** | leakage with both authorities named | GA / GS |
| **E3** | self-reported belief (recall questions) | none, except `ceiling` |

## Models

| id | checkpoint under `ckpt/` | what it is |
|---|---|---|
| `M_base` | — | Qwen/Qwen3.5-4B, no adapter |
| `SA_GA` | `sdf_M_base_GA_DS-1uGRADER__single_GRADER/checkpoint-40` | grader → altruistic only |
| `SA_DS` | `sdf_M_base_GA_DS-1uDEVELOPER__single_DEVELOPER/checkpoint-43` | developer → self-interested only |
| `CA_GA_DS` | `sdf_M_base_GA_DS_steps/checkpoint-567` | grader altruistic, developer self-interested |
| `CA_GS_DA` | `sdf_M_base_GS_DA/checkpoint-574` | the mirror |

The `checkpoint-N` suffix is load-bearing: the parent directories hold only dose
checkpoints and are **not** loadable adapters. Pointing vLLM at a parent kills the
engine.

## Framings

| id | poles |
|---|---|
| `F1_good_bad` | some good cause vs some bad cause |
| `F2_alt_self` | a charity vs the two of us — **the axis the corpus was trained on** |
| `F3_alt_neutral` | some good cause vs (unstated) |
| `F4_neutral_self` | (unstated) vs the two of us |

---

## `shards/` — 22 files, 346 MB, the primary data

**Not in git.** No git-lfs is configured and 348 MB of binary would grow a 41 MB
repo about ninefold, permanently. The files sit here on disk and on the shared
filesystem; this table is the index, and `make_results.py` regenerates every
published number from them.

Every row is one rollout with its full reasoning trace. Columns: the grid cell
(`item_id`, `mapping`, `paraphrase`, `threshold`, `framing`), the model and
condition, `completion` and `final`, `n_output_tokens`, `finish_reason`.

| file | rows | MB |
|---|---|---|
| `E1__M_base__F1_good_bad` | 2160 | 35.7 |
| `E1__M_base__F2_alt_self` | 2160 | 37.1 |
| `E1__M_base__F3_alt_neutral` | 1080 | 17.8 |
| `E1__M_base__F4_neutral_self` | 1080 | 17.5 |
| `E1__SA_GA__F2_alt_self` | 2160 | 37.5 |
| `E1__SA_DS__F2_alt_self` | 2160 | 39.1 |
| `E1__CA_GA_DS__F2_alt_self` | 2160 | 21.9 |
| `E1__CA_GS_DA__F2_alt_self` | 2160 | 22.9 |
| `E1__CA_GA_DS__F3_alt_neutral` | 2160 | 21.5 |
| `E1__CA_GS_DA__F3_alt_neutral` | 2160 | 22.8 |
| `E2__M_base__F2_alt_self__GA` | 1080 | 12.2 |
| `E2__M_base__F2_alt_self__GS` | 1080 | 12.0 |
| `E2__CA_GA_DS__F2_alt_self__GA` | 1080 | 8.9 |
| `E2__CA_GA_DS__F2_alt_self__GS` | 1080 | 8.6 |
| `E2__CA_GS_DA__F2_alt_self__GA` | 1080 | 8.9 |
| `E2__CA_GS_DA__F2_alt_self__GS` | 1080 | 9.1 |
| `E3__M_base__ceiling` | 96 | 0.6 |
| `E3__M_base__final` | 96 | 0.9 |
| `E3__SA_GA__final` | 624 | 3.6 |
| `E3__SA_DS__final` | 624 | 3.6 |
| `E3__CA_GA_DS__final` | 624 | 1.9 |
| `E3__CA_GS_DA__final` | 624 | 2.3 |

**Sampling differs by task and it matters when comparing.** E1 Stage A (the
adapter grids) and the F1/F2 base grids ran at n=2 (2160 rows). F3/F4 base and
all of E2 ran at n=1 (1080 rows) — an unintended config drift, recorded rather
than hidden. E3 ran at n=13 (624 rows) because the recall set has only 48 prompts
and no paraphrase clustering to lean on. The two base E3 files are n=2 (96 rows)
and should be re-run at n=13 if those numbers are quoted.

## `probes/` — 8 JSON files

| file | contents |
|---|---|
| `probe_results.json` | first authority probe, F2, single-layer transfer |
| `probe_auth_F2.json` | authority probe, F2, **transfer at every layer** |
| `probe_auth_F1.json` | same, F1 |
| `probe_leakage_F2_alt_self.json` | leakage probe, F2, pooled — **confounded** |
| `probe_leak_F1.json` | leakage probe, F1 — **shuffled control 0.602, do not cite** |
| `probe_leak_F2_ctrl.json` | leakage probe, F2, **with the within-mapping control** — the one to use |
| `probe_adapter_CA.json` | contrastive adapter pair — **positive control only, see FINDINGS §7b** |
| `probe_adapter_SA.json` | single-authority adapter pair — same caveat |

Each holds per-layer AUC and the shuffled-label control at that layer.

## `logs/` — 15 files

| file | what it shows |
|---|---|
| `queue_distilled.log` | every task start, completion and runtime; the distilled form of `queue.log` |
| `queue.log` | the raw original, mostly vLLM progress bars |
| `preflight.log` | the fail-closed checks that gate every run |
| `freeze.log` | threshold re-freeze at max_tokens 16384 |
| `seq2.log`, `seq4.log`, `seq5.log` | probe sequencers; `seq4` holds the adapter-attachment failure, `seq5` the fix |
| `chain.log`, `pchain.log`, `drive.log` | earlier chained runs |
| `seq3_note.txt` | why seq3 stalled — `pgrep -f seq2.sh` matched the tmux wrapper |
| `steer_smoke.log` | the 62 tok/s measurement that killed the steering plan |
| `probe*.log` | per-run probe output including layer tables |

## `orchestration/` — 8 shell scripts

The harness that actually drove the box: `drive.sh` (the task queue),
`seq2/3/4/5.sh` (probe sequencers), `probe_chain.sh`, `steer_chain.sh`,
`restart.sh`. Kept because the failure modes recorded in `FINDINGS.md` are
mostly *orchestration* failures, and these are the evidence.

Two lessons are embedded here. `seq5.sh` is launched via `setsid` inside its own
tmux session — a `nohup` from a script that then exits dies with the tmux
server. And never edit a running `.sh`: bash reads it by byte offset, so an edit
mid-execution makes it jump into the middle of a line.

## `claims/` — 22 lock files

One per completed task, created with `O_CREAT|O_EXCL` so two VMs could take
tasks from one queue without contending. Kept as the record of which box ran
what.

## `manifest_vmB.jsonl`

One line per completed task: task id, output file, row count, runtime. Written
with `fsync`, append-only, per-VM so two machines never contend.

## `fermi_items.v1_thresholds.bak`

The thresholds as they stood *before* the v2 re-freeze, kept so the change is
auditable. Every one of the 18 moved; `beeflowers` by 0.19× and `pawnmoves` by
1.76×. The v1 values were frozen at `max_tokens` 2048 while the grid ran at
32768, and leakage is "which side of the threshold did it land on", so a wrong
threshold moves the measurement's zero point rather than adding noise.

---

## On the shared filesystem

`/mnt/filesystem-u9/gcvl/` outlives the VM. `v2/README.md` there (copied here as
`SHARED_FS_README.md`) is the self-describing index, and `v2/repo_snapshot/`
holds the exact code that produced every number. Verified after the final sync:
**230 files, 351 MB**, plus `ckpt/` at 17 GB.

## Not in this directory

- **Adapters and base weights** stay on the shared filesystem under
  `/mnt/filesystem-u9/gcvl/ckpt/` (17 GB) and the HF cache. Retrainable from the
  frozen corpora; not copied here.
- **The v1 corpora** (5,700 documents per universe) are on the shared filesystem.
- **`v1` results** are in `../../mats_first_attempt/`, kept for provenance and
  deliberately not cited in the v2 write-up.

## Reproducing a number

```bash
python scripts/make_results.py --experiment E1
```

Recomputes from the parquet files rather than from any saved summary. Note the
key names: `src/metrics.py` filters on `parsed` and `good_side`; writing
`estimate`/`good` instead makes every metric return `nan` silently.
