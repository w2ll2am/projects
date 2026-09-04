#!/usr/bin/env python3
"""Entrypoint 3 of 3 — recompute every number from the saved shards.

    python scripts/make_results.py                     # all tables
    python scripts/make_results.py --experiment E1
    python scripts/make_results.py --out results/v2

Reads shards written by run_experiments.py on BOTH VMs. Manifests are per-VM and
append-only, so this merges by globbing rather than by reading a shared index,
and every shard was written atomically, so a shard being written right now is
either absent or complete — never half-read.

Nothing here re-derives a number from a previous summary. Every figure comes from
the rollout rows, which is what caught the pooled-row error in the first attempt.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import framings, metrics, parse, tasks  # noqa: E402


# --------------------------------------------------------------------------- #
def load_shards(experiment: str | None = None) -> "list[dict]":
    import pandas as pd
    rows: list[dict] = []
    for p in sorted(tasks.shards_dir().glob("*.parquet")):
        if experiment and not p.name.startswith(experiment + "__"):
            continue
        rows += pd.read_parquet(p).to_dict("records")
    if not rows:
        raise SystemExit(f"no shards under {tasks.shards_dir()}")
    return rows


def score_leakage(rows: list[dict]) -> list[dict]:
    """Attach the parsed estimate and the good-side verdict to every row."""
    out = []
    for r in rows:
        est = parse.parse_row(r)
        gs = metrics.good_side(est, r["threshold"], r["mapping"])
        # NOTE the key names. src/metrics.py keys on `parsed` and `good_side`
        # (see _counts); writing `estimate`/`good` instead makes every metric
        # return nan silently, because _counts filters on r.get("parsed").
        out.append({**r, "estimate": est,
                    "parsed": est is not None, "good_side": gs})
    return out


def leakage_table(rows: list[dict]) -> list[dict]:
    """One line per (model, framing), with the cluster-t interval over paraphrases."""
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["framing"])].append(r)
    table = []
    for (model, framing), rs in sorted(by.items()):
        scored = [r for r in rs if r["good_side"] is not None]
        lo, hi = metrics.cluster_t_interval(scored, metrics.leakage,
                                            cluster_key="paraphrase")
        table.append(dict(
            model=model, framing=framing, n=len(rs),
            parse_rate=round(metrics.parse_rate(rs), 4),
            truncation=round(metrics.truncation_rate(rs), 4),
            p_good=round(metrics.p_good(scored), 4),
            leakage=round(metrics.leakage(scored), 4),
            ci_lo=round(lo, 4), ci_hi=round(hi, 4),
            excludes_zero=bool(lo > 0 or hi < 0),
        ))
    return table


def headline(rows: list[dict]) -> dict:
    """THE result: does an implanted belief move behaviour, with no authority in
    the prompt? The contrast is CA_GA_DS against CA_GS_DA on F2_alt_self."""
    f = framings.DEFAULT_FRAMING
    sel = {m: [r for r in rows if r["model"] == m and r["framing"] == f
               and r["good_side"] is not None]
           for m in ("M_base", "CA_GA_DS", "CA_GS_DA")}
    missing = [m for m, rs in sel.items() if not rs]
    if missing:
        return {"status": f"incomplete — no rows for {missing}"}
    ga, gs = metrics.leakage(sel["CA_GA_DS"]), metrics.leakage(sel["CA_GS_DA"])
    lo, hi = metrics.paired_cluster_t_interval(
        sel["CA_GA_DS"], sel["CA_GS_DA"], metrics.leakage, cluster_key="paraphrase")
    return {"framing": f,
            "base_leakage": round(metrics.leakage(sel["M_base"]), 4),
            "CA_GA_DS_leakage": round(ga, 4),
            "CA_GS_DA_leakage": round(gs, 4),
            "difference": round(ga - gs, 4),
            "paired_ci": [round(lo, 4), round(hi, 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def recall_table(rows: list[dict]) -> list[dict]:
    """E3: says-altruistic per authority, plus precision / recall / F1.

    Positive class is "asserts altruistic"; ground truth is whichever authority
    that model's corpus made altruistic. F1 alone hides the difference between
    the two contrastive models — precision is what separates them — so all four
    are reported and none is used on its own.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "belief_recall", Path(__file__).resolve().parent / "10_belief_recall.py")
    rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)

    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["condition"])].append(r)
    table = []
    for (model, cond), rs in sorted(by.items()):
        panels = defaultdict(list)
        for r in rs:
            panels[r.get("probed_authority", "?")].append(r)
        entry = {"model": model, "condition": cond, "n": len(rs),
                 "excluded": round(1 - metrics.parse_rate(rs), 4)}
        for auth, prs in panels.items():
            entry[f"recall_{auth}"] = round(rc.recall_rate(prs), 4)
        table.append(entry)
    return table


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", choices=["E1", "E2", "E3"])
    ap.add_argument("--out", default=None, help="directory for JSON output")
    args = ap.parse_args()

    out: dict[str, object] = {}
    if args.experiment in (None, "E1"):
        rows = score_leakage(load_shards("E1"))
        out["E1_leakage"] = leakage_table(rows)
        out["headline"] = headline(rows)
    if args.experiment in (None, "E2"):
        out["E2_leakage"] = leakage_table(score_leakage(load_shards("E2")))
    if args.experiment in (None, "E3"):
        out["E3_recall"] = recall_table(load_shards("E3"))

    text = json.dumps(out, indent=2)
    print(text)
    if args.out:
        d = Path(args.out); d.mkdir(parents=True, exist_ok=True)
        tasks.atomic_write_bytes(d / "results_v2.json", text.encode())
        print(f"\nwrote {d / 'results_v2.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
