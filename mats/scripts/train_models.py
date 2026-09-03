#!/usr/bin/env python3
"""Entrypoint 1 of 3 — train any subset of the model matrix.

The v2 plan trains NO new SDF corpora and no new single-authority models. The
only training on the critical path is the effective-batch diagnostic
(PLAN.md section 6): the two contrastive models retrained at effective batch 32.

    python scripts/train_models.py --list
    python scripts/train_models.py --models CA_GA_DS_eb32
    python scripts/train_models.py --diagnostic          # both eb32 models
    python scripts/train_models.py --models CA_GA_DS_eb32 --dry-run

Claiming is shared with run_experiments.py, so two VMs can both be pointed at
--diagnostic and will take one model each without coordination.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import tasks  # noqa: E402

HERE = Path(__file__).resolve().parent

#: Every trainable model and the exact 06_train_sdf.py invocation that produces
#: it. Recorded as data rather than prose so the run is reproducible from this
#: file alone, and so --dry-run can print the real command.
#:
#: eb32: batch 8 / grad-accum 4 = effective batch 32, matching the single-
#: authority runs, with max_length 4096 kept from the corrected config. v1's
#: contrastive runs used batch 2 / accum 1 (effective batch 2) because the
#: leading hypothesis was too-few-steps; that choice maximised step count and
#: minimised effective batch, and this run is what separates the two.
RECIPES: dict[str, list[str]] = {
    "CA_GA_DS_eb32": [
        "--parent", "M_base", "--universes", "GA_DS",
        "--max-length", "4096", "--batch-size", "8", "--grad-accum", "4",
        "--output-dir", "{ckpt}/sdf_M_base_GA_DS_eb32", "--seed", "0",
    ],
    "CA_GS_DA_eb32": [
        "--parent", "M_base", "--universes", "GS_DA",
        "--max-length", "4096", "--batch-size", "8", "--grad-accum", "4",
        "--output-dir", "{ckpt}/sdf_M_base_GS_DA_eb32", "--seed", "0",
    ],
}

DIAGNOSTIC = ["CA_GA_DS_eb32", "CA_GS_DA_eb32"]


def build_cmd(key: str) -> list[str]:
    ckpt = tasks.exp_root() / "ckpt"
    argv = [a.format(ckpt=ckpt) for a in RECIPES[key]]
    return [sys.executable, str(HERE / "06_train_sdf.py"), *argv]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=None, choices=sorted(RECIPES))
    ap.add_argument("--diagnostic", action="store_true",
                    help="the two effective-batch-32 contrastive retrains")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print commands, train nothing")
    ap.add_argument("--no-claim", action="store_true",
                    help="skip claiming (single-VM runs)")
    args = ap.parse_args()

    if args.list:
        for k in sorted(RECIPES):
            m = tasks.MODELS.get(k)
            print(f"{k:16} -> ckpt/{m.adapter if m else '?'}")
            print(f"{'':16}    {' '.join(RECIPES[k])}")
        return 0

    want = args.models or (DIAGNOSTIC if args.diagnostic else None)
    if not want:
        ap.error("give --models, --diagnostic or --list")

    rc = 0
    for key in want:
        task = tasks.Task("TRAIN", key)
        if not args.no_claim and not args.dry_run and not tasks.claim(task):
            print(f"[skip] {key}: already claimed by another VM")
            continue
        cmd = build_cmd(key)
        print(f"[{tasks.vm_id()}] {key}: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            # hand it back so the other VM can retry rather than silently losing it
            tasks.release(task)
            print(f"[FAIL] {key} exited {proc.returncode}; claim released", file=sys.stderr)
            rc = proc.returncode
        else:
            tasks.record(task, tasks.exp_root() / "ckpt" / tasks.MODELS[key].adapter,
                         kind="training")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
