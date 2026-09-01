#!/usr/bin/env python3
"""MILESTONE FIGURES — the four plots the writeup is built on (HANDOFF §7.8).

Self-contained: matplotlib is the only third-party requirement for plotting
(pandas/pyarrow are needed only to READ a real parquet shard; ``--self-test``
runs without either). Every statistic comes from ``src.metrics`` — nothing
statistical is reimplemented here except the two interval forms metrics does
not provide (Wilson, and the derived interval on an average of two
proportions), which are marked as such below.

Figures written to ``mats/figures/`` at 150 dpi::

    fig_leakage_forest.png   per-paraphrase leakage + pooled bootstrap AND
                             cluster-t intervals            (the Gate 1 figure)
    fig_mapping_split.png    p_good under mapping=above vs below, with SPLIT
    fig_item_heatmap.png     per-item p_good x mapping, to expose pinned items
    fig_trace_lengths.png    output-token distribution vs the max_tokens cap

plus ``figures/summary_stats.json`` — every number the figures are drawn from,
so the writeup never has to re-derive them.

DESIGN RULES, enforced throughout (the user asked for these explicitly):

1. Every plotted estimate carries a 95% interval. Where an interval is not
   computable it is printed as "no CI (n too small)" ON the figure, never
   silently dropped.
2. Any figure using a paraphrase-clustered interval states k, the number of
   clusters, in its caption. FINDINGS.md (2026-09-01) measured the k=5
   percentile cluster bootstrap at a ~16% false-positive rate against a
   nominal 5%; ``metrics.cluster_t_interval`` restores ~5%. Both are drawn on
   the forest plot and the caption says which to believe when they disagree.
3. Colourblind-safe palette (Okabe-Ito), no chartjunk, units on every axis.

Examples::

    python scripts/08_plots.py --self-test           # synthetic data, no GPU, no shard
    python scripts/08_plots.py                       # real data from rollouts_dir()
    python scripts/08_plots.py --rollouts-dir /mnt/filesystem-m9/gcvl/results/rollouts
    python scripts/08_plots.py --shard M_base --figures-dir /tmp/figs
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import metrics
from src.paths import REPO, rollouts_dir

LOG = logging.getLogger("plots")

DPI = 150
PARSE_RATE_FLAG = 0.95      # items below this get a marked label on the heatmap
PIN_MARGIN = 0.02           # matches 03_replicate_leakage.PIN_MARGIN
DEFAULT_MAX_TOKENS = 32768  # FINDINGS.md 2026-09-01: adopted cap, 0% truncation

# Okabe-Ito, colourblind-safe. Fixed roles so the four figures read as a set.
C = {
    "above":  "#0072B2",    # blue
    "below":  "#D55E00",    # vermillion
    "point":  "#000000",
    "boot":   "#009E73",    # bluish green — percentile cluster bootstrap
    "t":      "#CC79A7",    # reddish purple — cluster-t (the one to believe)
    "muted":  "#999999",
    "flag":   "#E69F00",    # orange — warnings / pinned / low parse rate
}


# --------------------------------------------------------------------------- #
# intervals that src.metrics does not provide
# --------------------------------------------------------------------------- #
def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion.

    NOT in src.metrics, which offers ``logit_beta`` (a smoothed point estimate,
    not an interval) and only clustered intervals. Wilson is used here for the
    *within-cell* uncertainty of a single p_good: it stays inside [0, 1] and
    does not collapse to zero width at k=0 or k=n, which Wald does — and
    several items are expected to sit at exactly 0 or 1 (the "pinned" case
    figure 3 exists to expose).

    Returns (nan, nan) for n == 0.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _half(interval: tuple[float, float], point: float) -> float:
    """Symmetric half-width standing in for an asymmetric interval.

    Used only to PROPAGATE uncertainty into a derived quantity (the average of
    the two mapping rates, and their difference). The plotted intervals on
    individual proportions remain the true asymmetric Wilson bounds.
    """
    lo, hi = interval
    if math.isnan(lo) or math.isnan(hi):
        return float("nan")
    return max(abs(hi - point), abs(point - lo))


def combined_half(h_a: float, h_b: float, weight: float) -> float:
    """Half-width of ``weight * (a +/- h_a)`` combined with ``weight * (b +/- h_b)``.

    ``weight = 0.5`` gives the average of the two mapping rates (i.e. p_good,
    hence leakage); ``weight = 1.0`` gives their difference (the SPLIT). Treats
    the two mappings as independent, which they are by construction: disjoint
    rollouts. This ignores clustering by item/paraphrase, so it is a LOWER
    BOUND on the true uncertainty — stated on every figure that uses it.
    """
    if math.isnan(h_a) or math.isnan(h_b):
        return float("nan")
    return weight * math.sqrt(h_a * h_a + h_b * h_b)


def _counts(rows: Sequence[dict]) -> tuple[int, int]:
    """(good, parsed) — mirrors metrics._counts without touching its private name."""
    parsed = [r for r in rows if r.get("parsed") and r.get("good_side") is not None]
    return sum(1 for r in parsed if r["good_side"]), len(parsed)


def mapping_cell(rows: Sequence[dict], mapping: str) -> dict[str, float]:
    """p_good and its Wilson 95% interval for one mapping."""
    sub = [r for r in rows if r.get("mapping") == mapping]
    k, n = _counts(sub)
    lo, hi = wilson_interval(k, n)
    p = (k / n) if n else float("nan")
    return {"p_good": p, "ci_lo": lo, "ci_hi": hi, "k": float(k), "n": float(n),
            "half": _half((lo, hi), p) if n else float("nan")}


def leakage_with_binomial_ci(rows: Sequence[dict]) -> dict[str, float]:
    """leakage = mean(p_above, p_below) - 0.5, with a propagated binomial interval.

    This is the per-paraphrase row of the forest plot. It is NOT a clustered
    interval — within one paraphrase there is only one cluster — so it under-
    states uncertainty (items are correlated). The pooled row at the bottom of
    the forest plot is where the clustered intervals live.
    """
    a = mapping_cell(rows, "above")
    b = mapping_cell(rows, "below")
    point = metrics.leakage(rows)
    half = combined_half(a["half"], b["half"], 0.5)
    return {
        "leakage": point,
        "ci_lo": point - half, "ci_hi": point + half, "half": half,
        "p_good_above": a["p_good"], "p_good_below": b["p_good"],
        "n_parsed": a["n"] + b["n"],
    }


# --------------------------------------------------------------------------- #
# plotting helpers
# --------------------------------------------------------------------------- #
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": DPI, "savefig.dpi": DPI,
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.autolayout": False,
    })
    return plt


def reserve_bottom(fig, inches: float) -> None:
    """Reserve `inches` of space under the axes for tick labels + the caption.

    An absolute reserve, not a fraction: several of these figures grow in height
    with the number of rows, and a fixed fraction would let the caption creep
    into the x tick labels on the tall ones.
    """
    fig.subplots_adjust(bottom=min(0.6, inches / fig.get_figheight()))


def caption(fig, text: str) -> None:
    """One wrapped caption block under the axes. Captions carry the caveats."""
    fig.text(0.01, 0.005, text, ha="left", va="bottom", fontsize=7.5,
             color="#333333", wrap=True)


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    LOG.info("wrote %s (%.0f kB)", path, path.stat().st_size / 1e3)
    return path


def _fin(x: Any) -> bool:
    return isinstance(x, (int, float)) and not math.isnan(float(x)) and math.isfinite(float(x))


def _pct(x: float) -> str:
    return "n/a" if not _fin(x) else f"{100 * float(x):.1f}%"


def _sg(x: float) -> str:
    return "n/a" if not _fin(x) else f"{float(x):+.3f}"


# --------------------------------------------------------------------------- #
# figure 1 — leakage forest plot
# --------------------------------------------------------------------------- #
def fig_leakage_forest(rows: list[dict], out: Path, *, n_boot: int, seed: int,
                       synthetic: bool = False) -> dict | None:
    """Per-paraphrase leakage + pooled estimate with BOTH clustered intervals.

    The point of this figure is heterogeneity: if the paraphrase rows scatter
    widely around the pooled estimate, the pooled interval is doing all the
    work and k is small. Both intervals are drawn on the summary row so the
    reader can see how much of Gate 1's PASS condition depended on the weaker
    of the two.
    """
    plt = _mpl()
    paraphrases = metrics.iter_unique(rows, "paraphrase")
    k = len(paraphrases)
    if k == 0:
        LOG.warning("SKIP fig_leakage_forest: no paraphrase column in the rows")
        return None

    per: list[dict] = []
    for p in paraphrases:
        sub = [r for r in rows if r.get("paraphrase") == p]
        d = leakage_with_binomial_ci(sub)
        d["paraphrase"] = p
        per.append(d)

    pooled = metrics.leakage(rows)
    if k >= 2:
        boot_lo, boot_hi = metrics.cluster_bootstrap(
            rows, metrics.leakage, cluster_key="paraphrase", n_boot=n_boot, seed=seed)
        t_lo, t_hi = metrics.cluster_t_interval(
            rows, metrics.leakage, cluster_key="paraphrase")
    else:
        # With one cluster the bootstrap resamples the same cluster every time
        # and returns a ZERO-WIDTH interval — which would read as an implausibly
        # precise result. Refuse to draw it; say so on the figure instead.
        LOG.warning("only k=%d paraphrase cluster(s): no clustered interval is computable", k)
        boot_lo = boot_hi = t_lo = t_hi = float("nan")

    height = max(4.2, 1.9 + 0.34 * (k + 2))
    fig, ax = plt.subplots(figsize=(8.0, height))

    ys = list(range(k))[::-1]           # first paraphrase at the top
    for y, d in zip(ys, per):
        if _fin(d["half"]):
            ax.errorbar(d["leakage"], y, xerr=d["half"], fmt="o", ms=5,
                        color=C["point"], ecolor=C["muted"], elinewidth=1.4,
                        capsize=3, zorder=3)
        else:
            ax.plot(d["leakage"] if _fin(d["leakage"]) else 0.0, y, "x",
                    color=C["flag"], ms=7, zorder=3)
            ax.annotate("no CI (n too small)", (0.0, y), xytext=(6, 0),
                        textcoords="offset points", fontsize=7, color=C["flag"],
                        va="center")

    y_sum = -2.0
    ax.axhline(-1.0, color="#cccccc", lw=0.8)
    drew_pooled = False
    if _fin(boot_lo) and _fin(boot_hi):
        ax.plot([boot_lo, boot_hi], [y_sum + 0.16] * 2, "-", lw=2.6,
                color=C["boot"], solid_capstyle="butt")
        ax.annotate(f"percentile cluster bootstrap (k={k})", (boot_hi, y_sum + 0.16),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8, color=C["boot"])
        drew_pooled = True
    if _fin(t_lo) and _fin(t_hi):
        ax.plot([t_lo, t_hi], [y_sum - 0.16] * 2, "--", lw=2.6, color=C["t"],
                dash_capstyle="butt")
        ax.annotate(f"cluster-t, t({k - 1}) — believe this one", (t_hi, y_sum - 0.16),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8, color=C["t"])
        drew_pooled = True
    if _fin(pooled):
        ax.plot([pooled], [y_sum], "D", ms=8, color=C["point"], zorder=4)
        ax.annotate(f"pooled leakage {pooled:+.3f}", (pooled, y_sum),
                    xytext=(0, -20), textcoords="offset points", ha="center",
                    fontsize=8.5, fontweight="bold")
    if not drew_pooled:
        ax.annotate(f"pooled: NO clustered CI computable (k = {k} < 2 paraphrase clusters)",
                    (0.0, y_sum), xytext=(8, -12), textcoords="offset points",
                    fontsize=8, color=C["flag"])

    ax.axvline(0.0, color="#444444", lw=1.1, zorder=1)
    ax.set_yticks(ys + [y_sum])
    ax.set_yticklabels([f"paraphrase {d['paraphrase']}" for d in per] + ["POOLED"])
    for lbl in ax.get_yticklabels()[-1:]:
        lbl.set_fontweight("bold")
    ax.set_ylim(y_sum - 1.0, k - 0.4)
    ax.margins(x=0.30)                  # room for the direct labels on the pooled row
    ax.set_xlabel("leakage = p_good − 0.5   (proportion; 0 = no leakage, ±0.5 = total)")
    ax.set_title("Gate 1: value leakage by paraphrase" + (" [SYNTHETIC]" if synthetic else ""))

    disagree = (_fin(boot_lo) and _fin(t_lo)
                and ((boot_lo > 0 or boot_hi < 0) != (t_lo > 0 or t_hi < 0)))
    reserve_bottom(fig, 1.55)
    caption(fig,
            f"k = {k} paraphrase clusters. Each paraphrase row carries a binomial 95% interval propagated from that "
            f"paraphrase's two mapping rates\n"
            f"(disjoint cells); it ignores item-level clustering and is therefore a LOWER BOUND on the row's uncertainty.\n"
            f"POOLED shows BOTH clustered 95% intervals: percentile bootstrap [{_sg(boot_lo)}, {_sg(boot_hi)}] and "
            f"cluster-t [{_sg(t_lo)}, {_sg(t_hi)}].\n"
            f"FINDINGS.md measured the k=5 percentile bootstrap at a ~16% false-positive rate against a nominal 5% "
            f"(300 null sims); the cluster-t\n"
            + ("restores ~5%. THE TWO DISAGREE ON WHETHER 0 IS EXCLUDED \u2014 believe the cluster-t.\n"
               if disagree else "restores ~5%. Here the two agree on whether 0 is excluded.\n")
            + "More paraphrases is the only change that adds information; more rollouts per prompt does not alter k.")
    save(fig, out)

    return {
        "k_clusters": k,
        "pooled_leakage": pooled,
        "ci_bootstrap": [boot_lo, boot_hi],
        "ci_cluster_t": [t_lo, t_hi],
        "bootstrap_excludes_0": bool(_fin(boot_lo) and (boot_lo > 0 or boot_hi < 0)),
        "cluster_t_excludes_0": bool(_fin(t_lo) and (t_lo > 0 or t_hi < 0)),
        "intervals_disagree": bool(disagree),
        "n_boot": n_boot, "seed": seed,
        "per_paraphrase": [
            {"paraphrase": d["paraphrase"], "leakage": d["leakage"],
             "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"],
             "p_good_above": d["p_good_above"], "p_good_below": d["p_good_below"],
             "n_parsed": d["n_parsed"]}
            for d in per
        ],
    }


# --------------------------------------------------------------------------- #
# figure 2 — mapping split
# --------------------------------------------------------------------------- #
def fig_mapping_split(rows: list[dict], out: Path, *, synthetic: bool = False) -> dict | None:
    """p_good under mapping="above" vs "below", with the SPLIT annotated.

    Plan §9.3 / HANDOFF §7.1: a large split with a near-zero average is a LEVEL
    (anchoring) effect, not grader-following leakage. Averaging the mappings
    cancels it, which is exactly how a directional bias masquerades as a null.
    """
    plt = _mpl()
    cells = {m: mapping_cell(rows, m) for m in metrics.MAPPINGS}
    if all(not _fin(c["p_good"]) for c in cells.values()):
        LOG.warning("SKIP fig_mapping_split: no parsed rollouts under either mapping")
        return None

    a, b = cells["above"], cells["below"]
    split = (a["p_good"] - b["p_good"]) if (_fin(a["p_good"]) and _fin(b["p_good"])) else float("nan")
    split_half = combined_half(a["half"], b["half"], 1.0)
    avg = metrics.p_good(rows)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    xs = [0, 1]
    for x, m in zip(xs, metrics.MAPPINGS):
        c = cells[m]
        if not _fin(c["p_good"]):
            ax.annotate(f"mapping={m}\nno parsed rollouts\n— no CI computable",
                        (x, 0.5), ha="center", va="center", fontsize=9, color=C["flag"])
            continue
        yerr = [[c["p_good"] - c["ci_lo"]], [c["ci_hi"] - c["p_good"]]]
        ax.bar(x, c["p_good"], width=0.46, color=C[m], alpha=0.30, edgecolor=C[m], lw=1.4)
        ax.errorbar(x, c["p_good"], yerr=yerr, fmt="o", ms=7, color=C[m],
                    ecolor=C[m], elinewidth=1.8, capsize=5, zorder=3)
        # Value labels sit in a reserved band above every bar, so they can never
        # collide with the 0.5 reference line or the SPLIT bracket.
        ax.annotate(f"{c['p_good']:.3f}\n95% CI [{c['ci_lo']:.3f}, {c['ci_hi']:.3f}]\n"
                    f"n = {int(c['n'])} parsed",
                    (x, 1.10), ha="center", va="bottom", fontsize=8.5, color=C[m])

    ax.axhline(0.5, color="#444444", lw=1.1, ls=":", zorder=1)
    ax.annotate("0.5 = no preference", (-0.55, 0.5), xytext=(0, 4),
                textcoords="offset points", ha="left", fontsize=8, color="#444444")
    if _fin(avg):
        ax.axhline(avg, color=C["point"], lw=1.2, ls="-.", zorder=1)
        ax.annotate(f"average of the two mappings = {avg:.3f}  (leakage {avg - 0.5:+.3f})",
                    (1.55, avg), xytext=(0, -12), textcoords="offset points",
                    ha="right", fontsize=8, color=C["point"])

    if _fin(split):
        y = 0.94
        ax.plot([0, 0, 1, 1], [y - 0.04, y, y, y - 0.04], color="#444444", lw=1.0)
        txt = (f"SPLIT = p_above − p_below = {split:+.3f}"
               + (f"  ± {split_half:.3f} (95%)" if _fin(split_half)
                  else "  — no CI computable"))
        ax.annotate(txt, (0.5, y), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9.5, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(['mapping = "above"\n(good donation if estimate > threshold)',
                        'mapping = "below"\n(good donation if estimate ≤ threshold)'])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1.34)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("p_good — fraction of parsed estimates on the\ngood-donation side (proportion)")
    ax.set_title("Level effect or leakage? p_good by mapping" + (" [SYNTHETIC]" if synthetic else ""))

    reserve_bottom(fig, 1.85)
    caption(fig,
            "INTERPRETATION RULE (plan \u00a79.3): a LARGE SPLIT with a NEAR-ZERO AVERAGE is a LEVEL / ANCHORING effect \u2014 the model puts its\n"
            "estimate on one side of the stated threshold regardless of which side the prompt says pays the good cause. That is NOT\n"
            "grader-following leakage. Leakage requires p_good to rise under BOTH mappings, i.e. the average to move off 0.5. Averaging\n"
            "cancels a level effect exactly, so the average alone can report a null while a large directional bias is present.\n"
            "Error bars: Wilson 95% binomial intervals on each mapping's parsed rollouts. The SPLIT interval propagates the two\n"
            "(disjoint cells) and ignores paraphrase/item clustering, so it is a LOWER BOUND; for a clustered interval on the average\n"
            "see fig_leakage_forest.png.")
    save(fig, out)

    return {
        "p_good_above": a["p_good"], "ci_above": [a["ci_lo"], a["ci_hi"]],
        "n_parsed_above": a["n"],
        "p_good_below": b["p_good"], "ci_below": [b["ci_lo"], b["ci_hi"]],
        "n_parsed_below": b["n"],
        "split": split, "split_half_width_95": split_half,
        "p_good_average": avg, "leakage": metrics.leakage(rows),
        "interpretation": ("large split + near-zero average => LEVEL/anchoring effect, "
                           "not grader-following leakage (plan §9.3)"),
    }


# --------------------------------------------------------------------------- #
# figure 3 — per-item heatmap
# --------------------------------------------------------------------------- #
def fig_item_heatmap(rows: list[dict], out: Path, *, synthetic: bool = False) -> dict | None:
    """Per-item p_good under each mapping, sorted by mean — pinned items pop out.

    A "pinned" item (p_good at ~0 or ~1 under BOTH mappings) means the frozen
    threshold sits outside the model's estimate distribution: the item can
    never respond to the framing, so it contributes noise-free zeros to the
    effect and dilutes it. That is a bad threshold, not a null result.
    """
    plt = _mpl()
    items = metrics.iter_unique(rows, "item_id")
    if not items:
        LOG.warning("SKIP fig_item_heatmap: no item_id column in the rows")
        return None

    recs = []
    for it in items:
        sub = [r for r in rows if r.get("item_id") == it]
        a, b = mapping_cell(sub, "above"), mapping_cell(sub, "below")
        vals = [v["p_good"] for v in (a, b) if _fin(v["p_good"])]
        pinned = all(
            _fin(v["p_good"]) and (v["p_good"] <= PIN_MARGIN or v["p_good"] >= 1 - PIN_MARGIN)
            for v in (a, b)) and len(vals) == 2
        recs.append({
            "item_id": it, "above": a, "below": b,
            "mean": (sum(vals) / len(vals)) if vals else float("nan"),
            "parse_rate": metrics.parse_rate(sub),
            "pinned": bool(pinned),
        })
    recs.sort(key=lambda r: (-1e9 if not _fin(r["mean"]) else r["mean"]))

    n = len(recs)
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 1.8 + 0.36 * n)))
    grid = [[r["above"]["p_good"], r["below"]["p_good"]] for r in recs]
    masked = [[(0.5 if not _fin(v) else v) for v in row] for row in grid]
    im = ax.imshow(masked, cmap="coolwarm", vmin=0.0, vmax=1.0, aspect="auto")

    for i, r in enumerate(recs):
        for j, m in enumerate(("above", "below")):
            c = r[m]
            if not _fin(c["p_good"]):
                ax.text(j, i, "no data\nno CI", ha="center", va="center",
                        fontsize=7, color=C["flag"])
                continue
            half = c["half"]
            txt = f"{c['p_good']:.2f}\n±{half:.2f}" if _fin(half) else f"{c['p_good']:.2f}\nno CI"
            shade = "white" if (c["p_good"] < 0.22 or c["p_good"] > 0.78) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=shade)

    labels = []
    for r in recs:
        lab = r["item_id"]
        if r["pinned"]:
            lab += "  ■ PINNED"
        if _fin(r["parse_rate"]) and r["parse_rate"] < PARSE_RATE_FLAG:
            lab += f"  ▲parse {100 * r['parse_rate']:.0f}%"
        labels.append(lab)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    for i, r in enumerate(recs):
        if r["pinned"] or (_fin(r["parse_rate"]) and r["parse_rate"] < PARSE_RATE_FLAG):
            ax.get_yticklabels()[i].set_color(C["flag"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['mapping = "above"', 'mapping = "below"'])
    ax.set_title("Per-item p_good by mapping (sorted by item mean)"
                 + (" [SYNTHETIC]" if synthetic else ""))
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("p_good (proportion of parsed estimates on the good-donation side)", fontsize=8)

    n_pinned = sum(1 for r in recs if r["pinned"])
    n_lowparse = sum(1 for r in recs
                     if _fin(r["parse_rate"]) and r["parse_rate"] < PARSE_RATE_FLAG)
    reserve_bottom(fig, 1.35)
    caption(fig,
            f"Cell annotation: p_good \u00b1 the half-width of its Wilson 95% binomial interval. It is shown as \u00b1 because a "
            f"heatmap cell cannot carry an\n"
            f"error bar; the exact asymmetric bounds for every cell are in figures/summary_stats.json.\n"
            f"{n_pinned}/{n} items are PINNED \u2014 p_good within {PIN_MARGIN:.2f} of 0 or 1 under BOTH mappings. A pinned item's "
            f"frozen threshold sits outside the\n"
            f"model's estimate distribution, so the item cannot respond to the framing at all and only dilutes the pooled effect. "
            f"That is a bad\n"
            f"threshold, not a null result (plan \u00a73/\u00a711: re-run 02_freeze_thresholds.py).\n"
            f"\u25b2 marks {n_lowparse} item(s) with parse rate < {100 * PARSE_RATE_FLAG:.0f}%: their p_good rests on a subsample "
            f"that is not missing at random \u2014 unparsed\n"
            f"traces are disproportionately truncated or hedged.")
    save(fig, out)

    return {
        "n_items": n, "n_pinned": n_pinned, "n_low_parse_rate": n_lowparse,
        "parse_rate_flag": PARSE_RATE_FLAG, "pin_margin": PIN_MARGIN,
        "items": [
            {"item_id": r["item_id"], "mean_p_good": r["mean"],
             "parse_rate": r["parse_rate"], "pinned": r["pinned"],
             "p_good_above": r["above"]["p_good"],
             "ci_above": [r["above"]["ci_lo"], r["above"]["ci_hi"]],
             "n_parsed_above": r["above"]["n"],
             "p_good_below": r["below"]["p_good"],
             "ci_below": [r["below"]["ci_lo"], r["below"]["ci_hi"]],
             "n_parsed_below": r["below"]["n"]}
            for r in recs
        ],
    }


# --------------------------------------------------------------------------- #
# figure 4 — trace lengths
# --------------------------------------------------------------------------- #
def fig_trace_lengths(rows: list[dict], out: Path, *, max_tokens: int, n_boot: int,
                      seed: int, synthetic: bool = False) -> dict | None:
    """Output-token distribution against the max_tokens cap.

    Documents the FINDINGS.md truncation decision: max_tokens=2048 truncated
    56-75% of rollouts and put the MEDIAN at the cap; 16384 measured 4.7%
    (95% binomial [1.7%, 9.9%] — straddling the 5% rule); 32768 was adopted and
    measured 0% truncation across 640 rollouts. A truncated trace emits no
    closing </think>, parses to nothing, and biases the surviving sample toward
    short-reasoning rollouts — so this figure is a validity check, not decoration.
    """
    plt = _mpl()
    lengths = [int(r["n_output_tokens"]) for r in rows
               if r.get("n_output_tokens") is not None]
    if not lengths:
        LOG.warning("SKIP fig_trace_lengths: no n_output_tokens in the rows")
        return None

    med = metrics.percentile(lengths, 50)
    p90 = metrics.percentile(lengths, 90)
    mean = metrics.mean_tokens(rows)
    trunc = metrics.truncation_rate(rows)
    n_trunc = sum(1 for r in rows
                  if r.get("truncated") or r.get("finish_reason") == "length")

    def _stat(q: float) -> Callable[[Sequence[dict]], float]:
        def f(rs: Sequence[dict]) -> float:
            xs = [r["n_output_tokens"] for r in rs if r.get("n_output_tokens") is not None]
            return metrics.percentile(xs, q) if xs else float("nan")
        return f

    has_clusters = len(metrics.iter_unique(rows, "paraphrase")) >= 2
    if has_clusters:
        med_ci = metrics.cluster_bootstrap(rows, _stat(50), n_boot=min(n_boot, 2000), seed=seed)
        p90_ci = metrics.cluster_bootstrap(rows, _stat(90), n_boot=min(n_boot, 2000), seed=seed)
    else:
        med_ci = p90_ci = (float("nan"), float("nan"))
    trunc_ci = wilson_interval(n_trunc, len(rows))

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    top = max(max(lengths), max_tokens) * 1.02
    ax.hist(lengths, bins=48, range=(0, top), color=C["above"], alpha=0.55,
            edgecolor="white", linewidth=0.4)
    ymax = ax.get_ylim()[1]

    for val, ci, colour, name in ((med, med_ci, C["boot"], "median"),
                                  (p90, p90_ci, C["t"], "p90")):
        ax.axvline(val, color=colour, lw=1.8, ls="--")
        if _fin(ci[0]) and _fin(ci[1]):
            ax.axvspan(ci[0], ci[1], color=colour, alpha=0.16, lw=0)
            lab = f"{name} {val:,.0f} tok\n95% CI [{ci[0]:,.0f}, {ci[1]:,.0f}]"
        else:
            lab = f"{name} {val:,.0f} tok\nno CI (k<2 clusters)"
        ax.annotate(lab, (val, ymax * (0.94 if name == "median" else 0.72)),
                    xytext=(6, 0), textcoords="offset points", fontsize=8, color=colour)

    ax.axvline(max_tokens, color=C["flag"], lw=2.2)
    ax.annotate(f"max_tokens cap = {max_tokens:,}\ntruncated {_pct(trunc)} "
                f"(95% CI [{_pct(trunc_ci[0])}, {_pct(trunc_ci[1])}])",
                (max_tokens, ymax * 0.5), xytext=(-8, 0), textcoords="offset points",
                fontsize=8.5, color=C["flag"], ha="right")

    ax.set_xlim(0, top)
    ax.set_xlabel("output tokens per rollout (thinking trace + answer)")
    ax.set_ylabel(f"rollouts (count; N = {len(lengths):,})")
    ax.set_title("Trace-length distribution vs the token cap"
                 + (" [SYNTHETIC]" if synthetic else ""))

    reserve_bottom(fig, 1.65)
    caption(fig,
            f"mean {mean:,.0f} tok. Median and p90 bands are 95% paraphrase-clustered percentile bootstrap intervals "
            f"(k = {len(metrics.iter_unique(rows, 'paraphrase'))} clusters); at small k\n"
            f"these are coarse \u2014 see the health warning in src/metrics.py. The truncation interval is Wilson 95% on "
            f"{n_trunc}/{len(rows)} rollouts.\n"
            "WHY THIS MATTERS (FINDINGS.md, 2026-09-01): the Qwen3.5 chat template emits the opening <think> in the PROMPT, so a "
            "rollout truncated\n"
            "mid-reasoning carries NEITHER tag, parses to nothing, and drops out of every proportion in the other three figures \u2014 "
            "biasing the survivors\n"
            "toward short reasoning. max_tokens=2048 truncated 56\u201375% with the median AT the cap; 16384 measured 4.7% "
            "(95% [1.7%, 9.9%], straddling\n"
            "the 5% rule); 32768 was adopted and measured 0% truncation over 640 rollouts.")
    save(fig, out)

    return {
        "n_rollouts": len(rows), "n_with_lengths": len(lengths),
        "mean_output_tokens": mean, "median_output_tokens": med,
        "median_ci_95": list(med_ci), "p90_output_tokens": p90, "p90_ci_95": list(p90_ci),
        "max_output_tokens_observed": float(max(lengths)),
        "max_tokens_cap": max_tokens,
        "truncation_rate": trunc, "n_truncated": n_trunc,
        "truncation_ci_95_wilson": list(trunc_ci),
    }


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
ROW_COLUMNS = ["item_id", "mapping", "paraphrase", "threshold", "rollout_idx",
               "estimate", "parsed", "n_output_tokens", "finish_reason",
               "truncated", "good_side"]


def load_rows(shard: Path) -> list[dict] | None:
    """Read a rollout parquet shard, WITHOUT the `text`/`final` columns.

    Those two hold every thinking trace and dominate the file; nothing plotted
    here needs them. Returns None (with a clear log line) if the shard or the
    parquet stack is missing — several shards do not exist yet, and a missing
    input must skip a figure, never crash the run.
    """
    if not shard.exists():
        LOG.warning("MISSING shard %s — figures depending on it will be skipped", shard)
        return None
    try:
        import pandas as pd
    except ImportError:
        LOG.warning("pandas/pyarrow not installed — cannot read %s. "
                    "Install them, or use --self-test which needs neither.", shard)
        return None
    try:
        df = pd.read_parquet(shard, columns=ROW_COLUMNS)
    except Exception as exc:                                    # noqa: BLE001
        LOG.warning("column subset failed on %s (%s: %s); re-reading all columns",
                    shard, type(exc).__name__, exc)
        try:
            df = pd.read_parquet(shard)
        except Exception as exc2:                               # noqa: BLE001
            LOG.warning("SKIP: cannot read %s (%s: %s)", shard, type(exc2).__name__, exc2)
            return None
    rows = df.to_dict("records")
    # pandas nullable booleans come back as pd.NA, which is neither True nor
    # None; normalise so metrics.good_side's None contract holds.
    out = []
    for r in rows:
        d = dict(r)
        for key in ("good_side", "parsed", "truncated"):
            v = d.get(key)
            if v is None or str(v) == "<NA>" or (isinstance(v, float) and math.isnan(v)):
                d[key] = None
            else:
                try:
                    d[key] = bool(v)
                except (TypeError, ValueError):
                    d[key] = None
        if d.get("estimate") is not None and isinstance(d["estimate"], float) \
                and math.isnan(d["estimate"]):
            d["estimate"] = None
        out.append(d)
    LOG.info("loaded %d rollouts from %s", len(out), shard)
    return out


def load_summary(shard: Path) -> dict | None:
    """The `.summary.json` written beside a shard by 03_replicate_leakage.py."""
    path = shard.with_suffix(".summary.json")
    if not path.exists():
        LOG.info("no summary beside %s (fine — every number is recomputed here)", shard)
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:                                    # noqa: BLE001
        LOG.warning("could not read %s (%s)", path, exc)
        return None


# --------------------------------------------------------------------------- #
# synthetic data for --self-test
# --------------------------------------------------------------------------- #
def synth_rows(seed: int = 7, n_items: int = 12, n_para: int = 5,
               n_per_cell: int = 8) -> list[dict]:
    """Synthetic rollouts with a KNOWN structure, so the figures can be checked.

    Built to exercise every branch the real data can hit:
      * a LEVEL effect — p_good high under "below", low under "above", so the
        average is near 0.5 (figure 2's headline case);
      * paraphrase heterogeneity, so the forest plot has something to show;
      * two PINNED items (p_good ~0 / ~1 under both mappings);
      * one item with a ~85% parse rate, below the 95% flag;
      * a heavy-tailed length distribution with a few rollouts at the cap.
    """
    rng = random.Random(seed)
    items = [f"item{i:02d}" for i in range(n_items)]
    rows: list[dict] = []
    para_offset = {p: rng.gauss(0.0, 0.09) for p in range(n_para)}
    for i, it in enumerate(items):
        pinned_hi = (i == 0)
        pinned_lo = (i == 1)
        low_parse = (i == 2)
        base_above, base_below = 0.34, 0.71          # level effect, avg ~0.525
        for mapping in metrics.MAPPINGS:
            for p in range(n_para):
                base = base_above if mapping == "above" else base_below
                prob = min(0.99, max(0.01, base + para_offset[p] + rng.gauss(0, 0.05)))
                if pinned_hi:
                    prob = 1.0
                if pinned_lo:
                    prob = 0.0
                for idx in range(n_per_cell):
                    tok = int(min(DEFAULT_MAX_TOKENS,
                                  rng.lognormvariate(math.log(5200), 0.72)))
                    truncated = tok >= DEFAULT_MAX_TOKENS
                    parsed = (not truncated) and (rng.random() > (0.15 if low_parse else 0.02))
                    good = (rng.random() < prob) if parsed else None
                    rows.append(dict(
                        item_id=it, mapping=mapping, paraphrase=p,
                        threshold=10.0 ** (5 + i % 4), rollout_idx=idx,
                        estimate=(rng.lognormvariate(12, 2) if parsed else None),
                        parsed=bool(parsed), n_output_tokens=tok,
                        finish_reason="length" if truncated else "stop",
                        truncated=bool(truncated), good_side=good,
                    ))
    return rows


# --------------------------------------------------------------------------- #
def render_all(rows: list[dict], figdir: Path, *, max_tokens: int, n_boot: int,
               seed: int, synthetic: bool, source: str) -> dict:
    """Render every figure that its inputs allow; skip the rest with a log line."""
    stats: dict[str, Any] = {
        "source": source,
        "synthetic": synthetic,
        "figures_dir": str(figdir),
        "n_rollouts": len(rows),
        "overall": metrics.summarise(rows),
        "figures": {},
        "skipped": {},
    }
    jobs: list[tuple[str, Callable[[], dict | None]]] = [
        ("fig_leakage_forest.png",
         lambda: fig_leakage_forest(rows, figdir / "fig_leakage_forest.png",
                                    n_boot=n_boot, seed=seed, synthetic=synthetic)),
        ("fig_mapping_split.png",
         lambda: fig_mapping_split(rows, figdir / "fig_mapping_split.png",
                                   synthetic=synthetic)),
        ("fig_item_heatmap.png",
         lambda: fig_item_heatmap(rows, figdir / "fig_item_heatmap.png",
                                  synthetic=synthetic)),
        ("fig_trace_lengths.png",
         lambda: fig_trace_lengths(rows, figdir / "fig_trace_lengths.png",
                                   max_tokens=max_tokens, n_boot=n_boot,
                                   seed=seed, synthetic=synthetic)),
    ]
    for name, fn in jobs:
        try:
            result = fn()
        except Exception as exc:                                # noqa: BLE001
            LOG.warning("SKIP %s — %s: %s", name, type(exc).__name__, exc)
            stats["skipped"][name] = f"{type(exc).__name__}: {exc}"
            continue
        if result is None:
            stats["skipped"][name] = "required inputs absent (see log)"
        else:
            stats["figures"][name] = result
    return stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rollouts-dir", type=Path, default=None,
                    help="directory holding *.parquet rollout shards "
                         "(default: src.paths.rollouts_dir(), i.e. $EXP_ROOT/results/rollouts; "
                         "on the GPU box /mnt/filesystem-m9/gcvl/results/rollouts)")
    ap.add_argument("--shard", default="M_base",
                    help="shard basename, without .parquet (default M_base)")
    ap.add_argument("--figures-dir", type=Path, default=None,
                    help="output directory (default mats/figures, or mats/figures/selftest "
                         "under --self-test so synthetic output never overwrites real figures)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help=f"cap to draw on fig_trace_lengths (default: from the shard's "
                         f"summary.json if present, else {DEFAULT_MAX_TOKENS})")
    ap.add_argument("--n-boot", type=int, default=10000,
                    help="cluster-bootstrap resamples (default 10000)")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed (default 0)")
    ap.add_argument("--self-test", action="store_true",
                    help="render every figure from synthetic rollouts — no shard, no GPU, "
                         "no pandas required. Figures are titled [SYNTHETIC].")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)

    if args.self_test:
        figdir = args.figures_dir or (REPO / "figures" / "selftest")
        rows = synth_rows()
        LOG.info("SELF-TEST: %d synthetic rollouts (level effect, 2 pinned items, "
                 "1 low-parse item, heavy-tailed lengths)", len(rows))
        stats = render_all(rows, figdir, max_tokens=args.max_tokens or DEFAULT_MAX_TOKENS,
                           n_boot=min(args.n_boot, 2000), seed=args.seed,
                           synthetic=True, source="synthetic (--self-test)")
    else:
        figdir = args.figures_dir or (REPO / "figures")
        rdir = args.rollouts_dir or rollouts_dir()
        shard = rdir / f"{args.shard}.parquet"
        LOG.info("rollouts dir: %s", rdir)
        rows = load_rows(shard)
        if rows is None:
            LOG.warning("=" * 70)
            LOG.warning("NO DATA: %s is not readable. Every figure is skipped.", shard)
            LOG.warning("Run scripts/03_replicate_leakage.py on the GPU box first, or point")
            LOG.warning("--rollouts-dir at a mirror. `--self-test` validates this script "
                        "with no data at all.")
            LOG.warning("=" * 70)
            figdir.mkdir(parents=True, exist_ok=True)
            (figdir / "summary_stats.json").write_text(json.dumps(
                {"source": str(shard), "error": "shard not readable",
                 "figures": {}, "skipped": {k: "shard absent" for k in (
                     "fig_leakage_forest.png", "fig_mapping_split.png",
                     "fig_item_heatmap.png", "fig_trace_lengths.png")}}, indent=2))
            return 1
        summary = load_summary(shard)
        cap = args.max_tokens or int((summary or {}).get("max_tokens") or DEFAULT_MAX_TOKENS)
        stats = render_all(rows, figdir, max_tokens=cap, n_boot=args.n_boot,
                           seed=args.seed, synthetic=False, source=str(shard))
        if summary:
            stats["shard_summary_json"] = summary

    out = figdir / "summary_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, default=str))
    LOG.info("summary stats -> %s", out)
    LOG.info("rendered %d figure(s): %s", len(stats["figures"]),
             ", ".join(stats["figures"]) or "none")
    if stats["skipped"]:
        for name, why in stats["skipped"].items():
            LOG.warning("skipped %s — %s", name, why)
    return 0 if stats["figures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
