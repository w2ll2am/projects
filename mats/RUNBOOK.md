# Runbook — v2 run

One H200. Everything is serial. Commands are copy-pasteable.

Ubuntu's `~/.bashrc` returns early for non-interactive shells, so anything run over
`ssh` needs a login shell: `ssh vmA 'bash -lc "..."'`.

---

## 0. Connect and set up

```bash
ssh vmA                                   # ~/.ssh/config has vmA and vmB
tmux new -s run                           # nothing long-running outside tmux
```

```bash
export EXP_ROOT=/mnt/<shared-mount>/gcvl  # confirm the real path first: df -hT
export VM_ID=vmA
cd $EXP_ROOT/repo/mats || git clone <repo> $EXP_ROOT/repo && cd $EXP_ROOT/repo/mats
```

Install, vLLM first and from the nightly index — the pinned PyPI wheels cannot load
Qwen3.5 (see the header of `requirements.txt`):

```bash
uv venv && source .venv/bin/activate
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
uv pip install -r requirements.txt
```

## 1. Preflight — before any GPU spend

```bash
python3 scripts/vm_preflight.py
```

Checks EXP_ROOT is writable and has space, the GPU is visible, every dependency
imports, the grid is 18 items × 30 paraphrases with no ignore-the-bet language, all four
framings render with no unfilled slots, every required system message resolves, the four
adapters exist, `O_EXCL` behaves, and the task list starts with the ceiling gate.

It also detects whether thresholds are still the v1 values by comparing them against
`mats_first_attempt/data_snapshot/fermi_items.json`. Expect that to FAIL until step 2.

**Do not proceed while anything says FAIL.**

## 2. Re-freeze thresholds — BLOCKING

The committed thresholds were frozen at `max_tokens` 2048; the run uses 16384, and 6 of
the original 20 items had already drifted off the median. Every E1 and E2 grid reads
these, so this runs first and alone.

```bash
python3 scripts/02_freeze_thresholds.py          # ~25 min, 18 items
python3 scripts/vm_preflight.py                  # must now be clear
```

## 3. Prove the engine path — one cheap task

Never discover a vLLM or adapter problem 40 minutes into a grid.

```bash
python3 scripts/run_experiments.py --plan        # 31 tasks, ceiling gate first
python3 scripts/run_experiments.py \
    --experiments E3 --models M_base --n 1
```

Confirms the engine builds, the chat template renders, generation returns, and a shard
is written atomically. Check the shard appeared under `$EXP_ROOT/v2/shards/`.

## 4. The gate

```bash
python3 scripts/run_experiments.py --experiments E3 --models M_base
python3 scripts/make_results.py --experiment E3
```

`E3__M_base__ceiling` states the answer in the system message. **If the base model
cannot answer the recall format when told the answer, stop** — no recall number in the
project means anything and the eval needs fixing before more GPU is spent.

## 5. The queue

```bash
python3 scripts/run_experiments.py 2>&1 | tee -a $EXP_ROOT/v2/run_$VM_ID.log
```

~8.5 h. Order is fixed in `src/tasks.py:enumerate_tasks` and each gate precedes what it
gates. Resumable: completed tasks are claimed, so re-running skips them.

Watch the first completed grid's reported minutes. The 15–21 min estimate is
extrapolated from a measured 21 min at 2,400 rollouts / 32768. If a grid comes in over
~25 min, drop `--n 1` on the non-headline framings — the interval is driven by
paraphrase count, not samples per cell.

## 6. Read the result

```bash
python3 scripts/make_results.py --experiment E1 --out results/v2
```

`headline` reports `CA_GA_DS` vs `CA_GS_DA` on `F2_alt_self` with a paired cluster-t
interval. Before reading a null as a null, do what `PREREGISTRATION.md` §5 requires:
compute the new between-paraphrase sd from the `M_base` grids and compare it to the old
0.0840. A materially larger sd means underpowered, not absent.

## 7. Stage B — only on a hit

```bash
python3 scripts/run_experiments.py --stage-b     # +14 grids, ~4.2 h
```

Only if the paired interval excludes zero. Stage B is **exploratory** and must be
reported as such — the gate is pre-registered, so it cannot be quietly dropped.

---

## Optional extras

```bash
# complete the authority x direction 2x2 in E2.1 (+2 grids)
python3 scripts/run_experiments.py --experiments E2.1 --with-selfish

# dose-checkpoint sweep for the token-dose curve (+16 runs, ~1.3 h)
# needs each adapter's dose_map.json to enumerate real checkpoint names
```

Zero-GPU work to run in parallel on the laptop: sample 50 Gate 1 traces at a fixed seed
and count the motivated-backtracking signature, broken down by mapping. Both source
documents ask for this and nothing in the project has looked at a raw trace.

## If a second VM appears

Identical command, different `VM_ID`. Both run the same list and claim tasks atomically,
so they self-partition with no split to maintain.

```bash
ssh vmB 'bash -lc "cd $EXP_ROOT/repo/mats && VM_ID=vmB python3 scripts/run_experiments.py"'
```

First verify the shared filesystem really supports it — `O_EXCL` is atomic on POSIX and
NFSv4 but *emulated* on NFSv3, where both VMs could claim the same task and run all 31
twice. Run on both simultaneously, then verify from either:

```bash
python3 scripts/check_shared_fs.py --root $EXP_ROOT/fscheck --vm vmA   # and --vm vmB
python3 scripts/check_shared_fs.py --root $EXP_ROOT/fscheck --verify
```

If it fails, do not use dynamic claiming — give each VM an explicit `--models` /
`--experiments` split instead.

## Recovery

A failed task releases its claim, so re-running the same command retries it. To force a
retry of a task that completed:

```bash
rm $EXP_ROOT/v2/claims/<task-id>.claim
```

Shards are never overwritten — each carries the VM id and a timestamp — so a re-run
leaves both copies and `make_results.py` reads both. Delete the stale shard if that
matters.
