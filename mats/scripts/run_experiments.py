#!/usr/bin/env python3
"""Entrypoint 2 of 3 — run every experiment against one persistent engine.

    # on BOTH VMs, identical command; they partition themselves
    VM_ID=vmA EXP_ROOT=/mnt/shared python scripts/run_experiments.py
    VM_ID=vmB EXP_ROOT=/mnt/shared python scripts/run_experiments.py

    python scripts/run_experiments.py --plan            # print the task list, run nothing
    python scripts/run_experiments.py --experiments E1 --models CA_GA_DS CA_GS_DA
    python scripts/run_experiments.py --include-eb32    # after the diagnostic retrain

ONE ENGINE PER PROCESS. Every run in the first attempt built its own vLLM engine
and paid ~2.4 min for it; across 30 grids that is over an hour of pure waste.
The engine is built once with `enable_lora=True` and adapters are hot-swapped per
task, which costs seconds because the base weights stay resident.

Tasks are claimed atomically (src/tasks.py), so the two VMs need no static split,
nothing is written twice, and a VM that dies leaves its remaining tasks for the
other one.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import framings, prompts, serve, tasks  # noqa: E402

HERE = Path(__file__).resolve().parent


def _load(stem: str):
    """Import a `NN_name.py` sibling, whose module name is not a valid identifier."""
    path = HERE / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem.replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# the three experiments
# --------------------------------------------------------------------------- #
def _run(task, engine, sampling, cells, key, system=None):
    """Render, generate, flatten. `generate` returns n Rollouts per prompt."""
    texts = [serve.to_prompt(c[key], system_msg=system) for c in cells]
    outs = serve.generate(engine, texts, sampling=sampling,
                          adapter_path=tasks.MODELS[task.model].adapter_path(),
                          adapter_id=_adapter_id(task.model))
    return _attach(cells, outs, task, system=system or "")


#: Stable small integer per model. vLLM keys its adapter cache on this, so it
#: must not collide between two adapters and must not change between calls.
_ADAPTER_IDS = {k: i + 1 for i, k in enumerate(sorted(tasks.MODELS))}


def _adapter_id(model: str) -> int:
    return _ADAPTER_IDS[model]


def rows_E1(task, engine, sampling, items, paras):
    """Value leakage, no system message. The headline."""
    grid = prompts.build_grid(items, paras, framing=task.framing)
    return _run(task, engine, sampling, grid, "text")


def rows_E2(task, engine, sampling, items, paras):
    """Value leakage with both authorities named. System message from 04."""
    arm = _load("04_prompted_arm")
    system = arm.ALL_CONDITIONS[task.condition]
    grid = prompts.build_grid(items, paras, framing=task.framing)
    return _run(task, engine, sampling, grid, "text", system=system)


def rows_E3(task, engine, sampling, items, paras):
    """Self-reported belief. `condition='ceiling'` states the answer in a system
    message — the control that decides whether a low recall means 'not implanted'
    or 'this model cannot answer the format at all'."""
    recall = _load("10_belief_recall")
    arm = _load("04_prompted_arm")
    # Score every model against ITS OWN universe. Using one universe for all
    # would score CA_GS_DA against the mirror of its training.
    qs = recall.build_questions(tasks.MODELS[task.model].universe)
    system = arm.ALL_CONDITIONS["GA"] if task.condition == "ceiling" else None
    return _run(task, engine, sampling, qs, "question", system=system)


def _attach(cells, outs, task, **extra):
    """One row per (cell, sample). `outs[i]` is the list of n Rollouts for cell i."""
    rows = []
    for cell, rollouts in zip(cells, outs):
        for k, r in enumerate(rollouts):
            rows.append({**cell, **extra, "model": task.model,
                         "experiment": task.experiment, "framing": task.framing,
                         "condition": task.condition, "sample": k,
                         "completion": r.text, "final": r.final,
                         "n_output_tokens": r.n_output_tokens,
                         "finish_reason": r.finish_reason})
    return rows


RUNNERS = {"E1": rows_E1, "E2": rows_E2, "E2.1": rows_E2, "E3": rows_E3}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", default=["E3", "E1", "E2", "E2.1"],
                    choices=["E1", "E2", "E2.1", "E3"])
    ap.add_argument("--stage-b", action="store_true",
                    help="add E1's 14 generalisation grids. Only after stage A's "
                         "paired interval is seen to exclude zero.")
    ap.add_argument("--with-selfish", action="store_true",
                    help="add the two self-interested single-authority conditions to E2.1")
    ap.add_argument("--models", nargs="+", default=list(tasks.CORE_MODELS))
    ap.add_argument("--framings", nargs="+", default=list(framings.FRAMINGS))
    ap.add_argument("--include-eb32", action="store_true",
                    help="add the effective-batch-32 retrains to the model list")
    ap.add_argument("--n", type=int, default=2, help="samples per cell")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--plan", action="store_true", help="print the task list and exit")
    args = ap.parse_args()

    models = list(args.models)
    if args.include_eb32:
        models += ["CA_GA_DS_eb32", "CA_GS_DA_eb32"]

    todo = tasks.enumerate_tasks(models, args.framings, args.experiments,
                                 stage_b=args.stage_b,
                                 with_selfish=args.with_selfish)
    if args.plan:
        for i, t in enumerate(todo, 1):
            print(f"{i:3}. {t.id}")
        print(f"\n{len(todo)} tasks. Both VMs run this same list and self-partition.")
        return 0

    items = prompts.load_items()
    paras = prompts.load_paraphrases()
    print(f"[{tasks.vm_id()}] {len(items)} items x 2 mappings x {len(paras)} paraphrases; "
          f"{len(todo)} tasks queued", flush=True)

    # max_model_len tracks max_tokens: reserving KV for 32k when we only
    # generate 16k wastes cache and slows every batch.
    engine = serve.build_engine(enable_lora=True, max_lora_rank=32,
                                max_model_len=args.max_tokens + 1024)
    sampling = serve.default_sampling(n=args.n, max_tokens=args.max_tokens)

    done = failed = 0
    for task in tasks.pending(todo):
        t0 = time.time()
        print(f"[{tasks.vm_id()}] -> {task.id}", flush=True)
        try:
            rows = RUNNERS[task.experiment](task, engine, sampling, items, paras)
        except Exception as exc:                      # noqa: BLE001 - keep going
            tasks.release(task)
            print(f"[{tasks.vm_id()}] !! {task.id} failed: {exc!r}; claim released",
                  file=sys.stderr, flush=True)
            failed += 1
            continue
        shard = task.shard()
        _write_parquet(shard, rows)
        tasks.record(task, shard, n_rows=len(rows), minutes=round((time.time() - t0) / 60, 1))
        done += 1
        print(f"[{tasks.vm_id()}] <- {task.id}  {len(rows)} rows  "
              f"{(time.time()-t0)/60:.1f} min  -> {shard.name}", flush=True)

    print(f"[{tasks.vm_id()}] finished: {done} done, {failed} failed, "
          f"{len(todo)-done-failed} claimed by the other VM")
    return 1 if failed else 0


def _write_parquet(path: Path, rows: list[dict]) -> None:
    """Write-then-rename, so a concurrent make_results never sees a partial file."""
    import io
    import pandas as pd
    buf = io.BytesIO()
    pd.DataFrame(rows).to_parquet(buf, index=False)
    tasks.atomic_write_bytes(path, buf.getvalue())


if __name__ == "__main__":
    raise SystemExit(main())
