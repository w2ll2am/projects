#!/usr/bin/env python
"""Freeze the per-item bet thresholds against the base model (plan section 2.4).

WHY `data/fermi_items.json` SHIPS WITH `"threshold": null`
---------------------------------------------------------
The threshold is the number the bet is settled against, and the whole design
rests on it sitting at the *centre of the model's own answer distribution* for
that item. If it does not, the item is pinned: every rollout lands on the same
side of the threshold whatever the framing does, `p_good` saturates at 0 or 1,
and the item contributes no signal (plan sections 3, 9.3, 11).

There is no way to guess that number. It is a property of this model on this
question, not a fact about the world, and it can be two orders of magnitude away
from the true value. So thresholds are measured, never hand-written:

    for each item:
        generate 32 UNCONDITIONED estimates (bare question, no bet framing)
        parse to floats, drop unparseable and non-positive
        threshold = 10 ** median(log10(estimates))       # geometric median

The geometric median is the right centre because Fermi estimates are spread over
orders of magnitude; the arithmetic median of a log-normal-ish sample is
dominated by the largest draws.

Run this ONCE, on `M_base`, before anything else touches the data. Commit the
result. Never recompute: re-freezing after any model state has been evaluated
silently changes the meaning of every earlier `p_good`, so the script refuses to
overwrite non-null thresholds without `--force`.

Values are written unrounded. `src.prompts.display_threshold` rounds to 2
significant figures for presentation, and `build_grid` scores against that
rounded value.

This runs on the GPU box:

    python scripts/02_freeze_thresholds.py
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.parse import parse_answer
from src.prompts import ITEMS_PATH, build_baseline_prompts, load_config, load_items
from src.serve import build_engine, default_sampling, generate, to_prompt


def geometric_median(estimates: list[float]) -> float:
    """10 ** median(log10(x)) — the centre of a magnitude-scale sample."""
    return 10.0 ** statistics.median(math.log10(x) for x in estimates)


def usable(value: float | None) -> bool:
    """log10 needs a strictly positive, finite estimate."""
    return value is not None and math.isfinite(value) and value > 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--items", type=Path, default=ITEMS_PATH)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--n", type=int, default=None, help="override n_estimates")
    ap.add_argument(
        "--force",
        action="store_true",
        help="recompute thresholds that are already frozen (see the module docstring "
             "— this invalidates every result computed against the old thresholds)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)["freeze_thresholds"]
    n = args.n or int(cfg["n_estimates"])
    max_bad = float(cfg["max_unparseable_frac"])

    all_items = load_items(args.items)
    already = [it["id"] for it in all_items if it["threshold"] is not None]

    # Freeze ONLY the unfrozen items unless --force. Plan 2.4 mandates replacing
    # any item that exceeds the unparseable limit, which necessarily means
    # freezing a couple of new items against an otherwise-frozen file. Refusing
    # to run at all whenever anything is frozen would block that workflow; the
    # invariant that actually matters is that an existing threshold is never
    # silently recomputed, since every result already computed is relative to it.
    if args.force:
        items = all_items
        if already:
            print(
                f"--force: RECOMPUTING {len(already)} already-frozen threshold(s). "
                "This invalidates every result computed against them.",
                file=sys.stderr,
            )
    else:
        items = [it for it in all_items if it["threshold"] is None]
        if already:
            print(f"skipping {len(already)} already-frozen item(s); pass --force to recompute them")
        if not items:
            print("nothing to freeze — every item already has a threshold.")
            return 0
        print(f"freezing {len(items)} item(s): {', '.join(it['id'] for it in items)}\n")

    prompts = build_baseline_prompts(items)
    engine = build_engine()
    sampling = default_sampling(
        n=n,
        max_tokens=int(cfg["max_tokens"]),
        temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]),
        seed=cfg.get("seed"),
    )
    outs = generate(engine, [to_prompt(p["text"]) for p in prompts], sampling=sampling)

    thresholds: dict[str, float | None] = {}
    dropped: list[tuple[str, float]] = []
    print(f"\n{'item':<18}{'parsed':>8}{'unparsed':>10}{'trunc':>8}{'threshold':>18}")
    for prompt, rollouts in zip(prompts, outs):
        estimates = [v for v in (parse_answer(r.text) for r in rollouts) if usable(v)]
        n_roll = len(rollouts)
        bad_frac = 1.0 - len(estimates) / n_roll if n_roll else 1.0
        trunc = sum(r.truncated for r in rollouts) / n_roll if n_roll else 1.0

        if bad_frac > max_bad or not estimates:
            thresholds[prompt["item_id"]] = None
            dropped.append((prompt["item_id"], bad_frac))
            shown = "DROPPED"
        else:
            thresholds[prompt["item_id"]] = geometric_median(estimates)
            shown = f"{thresholds[prompt['item_id']]:.6g}"
        print(
            f"{prompt['item_id']:<18}{len(estimates):>8}{bad_frac:>9.0%}"
            f"{trunc:>8.0%}{shown:>18}"
        )

    # Merge back into the FULL item list, preserving order and every threshold
    # frozen by an earlier run. Writing `items` here would silently delete the
    # thresholds of items this run deliberately skipped.
    newly = {it["id"]: thresholds[it["id"]] for it in items}
    for it in all_items:
        if it["id"] in newly:
            it["threshold"] = newly[it["id"]]

    tmp = args.items.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(all_items, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(args.items)
    print(f"\nwrote {args.items}")

    if dropped:
        print(
            f"\n{len(dropped)} item(s) exceeded the {max_bad:.0%} unparseable limit and "
            "were left at threshold=null:",
            file=sys.stderr,
        )
        for item_id, frac in dropped:
            print(f"  {item_id}: {frac:.0%} unparseable", file=sys.stderr)
        print(
            "Replace them in data/fermi_items.json and re-run (no --force needed: only "
            "unfrozen items are recomputed), or the grid will refuse to build (plan 2.4).",
            file=sys.stderr,
        )
        return 1

    print("all 20 items frozen — commit data/fermi_items.json now and never recompute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
