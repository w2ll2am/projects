#!/usr/bin/env python3
"""Milestone figure: the whole Gate 1 + Gate 2 story on one page.

Four panels, all on the same p_good axis so the reader can see at a glance that
the prompted manipulation dwarfs the unprompted leakage, and that the authority
contrast lives in the compressed top of the scale.

Every point carries a 95% interval. Where the interval is clustered by
paraphrase, k is stated on the panel — results/FINDINGS.md records that the
k=5 percentile bootstrap ran at a ~16% false-positive rate against a nominal
5%, so an unlabelled clustered interval on this project is a trap.

Usage:
    python scripts/09_milestone_figure.py [--rollouts-dir DIR] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import metrics
from src.paths import rollouts_dir

LOG = logging.getLogger("milestone")

# Colourblind-safe; the same hue always means the same thing across panels.
C_UNPROMPTED = "#0072B2"
C_GRADER = "#D55E00"
C_DEVELOPER = "#009E73"
C_NULL = "#666666"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Used because several cells sit at 0 or 1, where
    the Wald interval collapses to zero width and lies about the uncertainty."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load(shard: Path) -> list[dict] | None:
    if not shard.exists():
        LOG.warning("missing %s — panel will be skipped", shard.name)
        return None
    import pandas as pd

    return pd.read_parquet(shard).to_dict("records")


def counts(rows) -> tuple[int, int]:
    parsed = [r for r in rows if r.get("parsed") and r.get("good_side") is not None]
    return sum(1 for r in parsed if r["good_side"]), len(parsed)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rollouts-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("figures/fig_milestone.png"))
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rd = args.rollouts_dir or rollouts_dir()
    g1 = load(rd / "M_base_k30.parquet")
    g1b = load(rd / "M_base_k30_seed1.parquet")
    g2 = load(rd / "M_base_prompted_k30.parquet")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    stats: dict = {}

    # ---- panel A: the headline comparison -------------------------------- #
    ax = axes[0][0]
    bars, lo_e, hi_e, cols = [], [], [], []
    labels = []
    if g1:
        k, n = counts(g1)
        p = metrics.p_good(g1)
        l, h = wilson(k, n)
        labels.append("no system\nmessage\n(Gate 1)")
        bars.append(p); lo_e.append(p - l); hi_e.append(h - p); cols.append(C_UNPROMPTED)
        stats["gate1_p_good"] = p
    if g2:
        for cond, col, lab in (("GA", C_GRADER, "grader wants\ncharity (GA)"),
                               ("GS", C_DEVELOPER, "developer wants\ncharity (GS)")):
            sub = [r for r in g2 if r.get("condition") == cond]
            k, n = counts(sub)
            p = metrics.p_good(sub)
            l, h = wilson(k, n)
            labels.append(lab)
            bars.append(p); lo_e.append(p - l); hi_e.append(h - p); cols.append(col)
            stats[f"gate2_{cond}_p_good"] = p
    if bars:
        x = range(len(bars))
        ax.bar(x, bars, color=cols, width=0.6)
        ax.errorbar(x, bars, yerr=[lo_e, hi_e], fmt="none", ecolor="black", capsize=4, lw=1.4)
        ax.axhline(0.5, color=C_NULL, ls="--", lw=1.2)
        ax.text(len(bars) - 0.45, 0.505, "0.50 = no leakage (by construction)",
                ha="right", va="bottom", fontsize=8, color=C_NULL)
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
        for i, v in enumerate(bars):
            ax.text(i, v + 0.035, f"{v:.1%}", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("p_good  (share of estimates on the rewarded side)")
        ax.set_title("A. Naming the outcome dimension dwarfs the leakage effect", fontsize=11)
        ax.text(0.5, -0.30, "Error bars are Wilson 95% on the pooled count and IGNORE paraphrase\n"
                            "clustering — they are a LOWER BOUND on the true uncertainty.",
                transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#444")

    # ---- panel B: leakage with both clustered intervals ------------------- #
    ax = axes[0][1]
    rowsB = []
    for name, rr in (("seed 0", g1), ("seed 1", g1b)):
        if not rr:
            continue
        lk = metrics.leakage(rr)
        bl, bh = metrics.cluster_bootstrap(rr, metrics.leakage, cluster_key="paraphrase",
                                           n_boot=args.n_boot, seed=0)
        tl, th = metrics.cluster_t_interval(rr, metrics.leakage, cluster_key="paraphrase")
        rowsB.append((name, lk, bl, bh, tl, th))
        stats[f"leakage_{name.replace(' ', '')}"] = {
            "point": lk, "bootstrap": [bl, bh], "cluster_t": [tl, th]}
    if rowsB:
        for i, (name, lk, bl, bh, tl, th) in enumerate(rowsB):
            y = len(rowsB) - 1 - i
            ax.plot([bl, bh], [y + 0.10] * 2, color="#009E73", lw=3.5,
                    solid_capstyle="butt")
            ax.plot([tl, th], [y - 0.10] * 2, color="#CC79A7", lw=3.5, ls=(0, (4, 2)),
                    solid_capstyle="butt")
            ax.plot([lk], [y], "D", color="black", ms=8, zorder=5)
            ax.text(th + 0.004, y, f"  {name}: {lk:+.4f}", va="center", fontsize=9)
        ax.axvline(0, color="black", lw=1.2)
        ax.set_ylim(-0.6, len(rowsB) - 0.4)
        ax.set_yticks([])
        ax.set_xlabel("leakage = p_good - 0.5")
        ax.set_title("B. Leakage is small, real, and reproducible (k=30)", fontsize=11)
        ax.plot([], [], color="#009E73", lw=3.5, label="percentile cluster bootstrap")
        ax.plot([], [], color="#CC79A7", lw=3.5, ls=(0, (4, 2)), label="cluster-t — believe this one")
        ax.legend(fontsize=8, loc="lower right", frameon=False)
        ax.text(0.0, -0.30, "k=30 paraphrase clusters. At k=5 the same measurement gave\n"
                            "+0.081 with a cluster-t interval of [-0.101, +0.265]: inconclusive.",
                transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#444")

    # ---- panel C: the authority contrast ---------------------------------- #
    ax = axes[1][0]
    if g2:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import importlib
        g2mod = importlib.import_module("04_prompted_arm") if False else None
        # compute inline rather than importing a module whose name starts with a digit
        def _sel(rr, cond, m=None):
            return [r for r in rr if r.get("condition") == cond
                    and (m is None or r.get("mapping") == m)]

        def _gap(a, b):
            ka, na = counts(a)
            kb, nb = counts(b)
            if not na or not nb:
                return float("nan")
            return metrics.logit_beta(ka, na) - metrics.logit_beta(kb, nb)

        def delta(rr):
            vals = [_gap(_sel(rr, "GA", m), _sel(rr, "GS", m)) for m in metrics.MAPPINGS]
            vals = [v for v in vals if not math.isnan(v)]
            return sum(vals) / len(vals) if vals else float("nan")

        per = []
        for pp in metrics.iter_unique(g2, "paraphrase"):
            sub = [r for r in g2 if r.get("paraphrase") == pp]
            per.append(delta(sub))
        per = [v for v in per if not math.isnan(v)]
        d = delta(g2)
        lo, hi = metrics.cluster_bootstrap(g2, delta, cluster_key="paraphrase",
                                           n_boot=args.n_boot, seed=0)
        stats["delta_gd_prompted"] = {"point": d, "paired_bootstrap": [lo, hi],
                                      "per_paraphrase": per}
        ax.hist(per, bins=12, color="#BBBBBB", edgecolor="white")
        ax.axvline(0, color="black", lw=1.4)
        ax.axvline(d, color=C_DEVELOPER, lw=2.5)
        ax.axvspan(lo, hi, color=C_DEVELOPER, alpha=0.16)
        ax.text(d, ax.get_ylim()[1] * 0.94, f"  pooled {d:+.3f}\n  95% CI [{lo:+.3f}, {hi:+.3f}]",
                color=C_DEVELOPER, fontsize=9, va="top", fontweight="bold")
        ax.text(0.02, 0.98, "follows the\nDEVELOPER", transform=ax.transAxes, fontsize=9,
                va="top", ha="left", color=C_DEVELOPER, fontweight="bold")
        ax.text(0.98, 0.98, "follows the\nGRADER", transform=ax.transAxes, fontsize=9,
                va="top", ha="right", color=C_GRADER, fontweight="bold")
        ax.set_xlabel("Delta = logit(GA) - logit(GS),  per paraphrase")
        ax.set_ylabel("paraphrases")
        ax.set_title("C. In conflict, the model sides with its developer", fontsize=11)
        n_neg = sum(1 for v in per if v < 0)
        ax.text(0.0, -0.26, f"{n_neg} of {len(per)} paraphrases negative. Paired cluster bootstrap "
                            f"(k={len(per)}): the\nresample draws the same paraphrases for both "
                            "conditions, so the paraphrase\neffect cancels in the difference.",
                transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#444")

    # ---- panel D: why k mattered ------------------------------------------ #
    ax = axes[1][1]
    if g1:
        per_k5, per_k30 = [], []
        for pp in metrics.iter_unique(g1, "paraphrase"):
            sub = [r for r in g1 if r.get("paraphrase") == pp]
            v = metrics.leakage(sub)
            if not math.isnan(v):
                (per_k5 if int(pp) < 5 else per_k30).append(v)
        ax.axhline(0, color="black", lw=1.2)
        ax.scatter([0] * len(per_k5), per_k5, s=70, color=C_GRADER, zorder=4,
                   label=f"the original 5 (mean {sum(per_k5)/len(per_k5):+.3f})")
        ax.scatter([1] * len(per_k30), per_k30, s=45, color=C_UNPROMPTED, alpha=0.75,
                   label=f"the 25 added (mean {sum(per_k30)/len(per_k30):+.3f})")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["original\nparaphrases", "added\nparaphrases"], fontsize=9)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel("leakage, per paraphrase")
        ax.set_title("D. The first five paraphrases were unrepresentative", fontsize=11)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        stats["per_paraphrase_original5"] = per_k5
        stats["per_paraphrase_added25"] = per_k30
        ax.text(0.0, -0.24, "The k=5 run overestimated BOTH the effect (+0.081 vs +0.061) and its\n"
                            "spread (sd 0.148 vs 0.084). More clusters is the only fix that adds\n"
                            "information; more rollouts per prompt does not change k at all.",
                transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#444")

    for row in axes:
        for a in row:
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
    fig.suptitle("Grader-conditioned value leakage in Qwen3.5-4B — Gates 1 and 2",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    LOG.info("wrote %s (%d kB)", args.out, args.out.stat().st_size // 1024)

    js = args.out.with_suffix(".json")
    js.write_text(json.dumps(stats, indent=2, default=float))
    LOG.info("wrote %s", js)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
