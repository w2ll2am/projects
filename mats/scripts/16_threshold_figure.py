#!/usr/bin/env python3
"""Threshold-exclusion diagnostics for the in-CoT trajectory analysis (report S3).

WHY THIS EXISTS. The trajectory analysis drops every in-CoT number within 2% of
the question's threshold, because the prompt states the threshold and the model
restates it: a value equal to the threshold is charitable under one threshold
direction and not the other, so with directions balanced it sits at exactly
0.500 by construction, and left in it manufactures a spurious "the model starts
unbiased" result.

The 2% was CHOSEN, not tuned. This script is the check that it did not matter,
and it also re-derives the position-binned trajectory table itself, which was
previously computed by code that was never committed.

  --only dist         distribution of in-CoT estimates around the threshold
  --only cutoff       share of estimates removed at each candidate cutoff
  --only sensitivity  first/last decile of the CoT at each candidate cutoff
  --only trajectory   the position-binned table printed in report S3
  --only ci           interval width under each clustering unit (report S3 note)

With no --only it runs everything and writes the figures.

Needs duckdb, pyyaml, matplotlib, numpy. No GPU.

Usage:
    python scripts/16_threshold_figure.py
    python scripts/16_threshold_figure.py --only trajectory
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402

from src import parse, prompts  # noqa: E402

SHARDS = ROOT / "results_v2" / "shards"
NUM = re.compile(r"\d[\d,]*\.?\d*")

#: The adopted cutoff, as a fraction of the threshold. See the module docstring.
BAND = 0.02
#: Extraction window: numbers within three orders of magnitude of the threshold.
OOM = 1000
#: Cutoffs swept for the sensitivity panels.
SWEEP = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
#: x limits for the sweep panels: the data range (0.1%-100%) plus 0.05 of a
#: decade of padding, so no gridline extends past the last measured point.
XLO, XHI = 10 ** -1.05, 10 ** 2.05
#: Normalised-position bins through the trace. The first two are the ones the
#: figure reports; all four appear in the report's table.
EDGES = [(0.0, 0.1), (0.3, 0.4), (0.6, 0.7), (0.9, 1.0)]

FRAMINGS = (("F1", "E1__M_base__F1_good_bad"), ("F2", "E1__M_base__F2_alt_self"))

#: Validated against the six checks in the dataviz palette (protanopia dE 19.2).
#: These are the report's --pos / --neg tokens; keep the two in step.
SER = {"F1": "#1a5340", "F2": "#c96a2c"}
LS = {"F1": "-", "F2": (0, (5, 2))}
INK, INK2, GRID, HL = "#151515", "#585858", "#dcdcdc", "#22405e"

#: t(k-1) at 97.5%. Same convention as src.metrics; see FINDINGS 9.2 for why a
#: z multiplier here would be wrong.
_T = {8: 2.306, 17: 2.110, 29: 2.045}


def shard(stem: str) -> str:
    hits = sorted(glob.glob(f"{SHARDS}/{stem}*.parquet"))
    if not hits:
        raise SystemExit(f"no shard matching {stem!r} under {SHARDS}/")
    return hits[0]


def load(stem: str):
    return duckdb.connect().execute(
        f"SELECT item_id, paraphrase, threshold, mapping, final, completion "
        f"FROM '{shard(stem)}'"
    ).fetchall()


def candidates(completion: str | None, thr: float, band: float = BAND) -> list[float]:
    """Ordered in-CoT estimates, with the near-threshold cutoff applied."""
    out = []
    for m in NUM.finditer(completion or ""):
        try:
            v = float(m.group(0).replace(",", ""))
        except ValueError:
            continue
        if v > 0 and thr / OOM <= v <= thr * OOM and abs(v - thr) / thr >= band:
            out.append(v)
    return out


def binned(rows, band: float, edges=EDGES) -> dict:
    """p(charitable side) per normalised-position bin.

    Averaged PER TRACE and then across traces -- not pooled over extractions,
    which would weight long traces more heavily and does not reproduce the
    published table.
    """
    per = {e: [] for e in edges}
    for _item, _para, thr, mapping, _final, comp in rows:
        if not comp or not thr:
            continue
        cand = candidates(comp, thr, band)
        n = len(cand)
        if n < 2:
            continue
        loc = defaultdict(list)
        for i, v in enumerate(cand):
            pos = i / (n - 1)
            good = int(prompts.on_good_side(v, mapping, thr))
            for e in edges:
                inside = (e[0] <= pos <= e[1]) if e[1] == 1.0 else (e[0] <= pos < e[1])
                if inside:
                    loc[e].append(good)
        for e, vs in loc.items():
            per[e].append(st.mean(vs))
    return {e: (st.mean(v) if v else float("nan")) for e, v in per.items()}


def log_ratios(rows) -> tuple[list[float], dict[str, list[float]]]:
    """log10(estimate / threshold) for every extraction, and the same by item."""
    flat, by_item = [], defaultdict(list)
    for item, _para, thr, _mapping, _final, comp in rows:
        if not comp or not thr:
            continue
        for v in candidates(comp, thr, band=0.0):
            lr = math.log10(v / thr)
            flat.append(lr)
            by_item[item].append(lr)
    return flat, by_item


# ----------------------------------------------------------------- commands

def cmd_dist(data) -> dict:
    print("== distribution of in-CoT estimates around the threshold ==")
    out = {}
    for tag, rows in data.items():
        flat, _ = log_ratios(rows)
        exact = sum(1 for x in flat if x == 0.0)
        print(f"  {tag}: {len(flat)} extractions, "
              f"{100 * exact / len(flat):.2f}% exactly on the threshold")
        out[tag] = {"n": len(flat), "exact_pct": 100 * exact / len(flat)}
    return out


def cmd_cutoff(data) -> dict:
    print("== share of extractions removed at each cutoff ==")
    out = {}
    for tag, rows in data.items():
        flat, _ = log_ratios(rows)
        rel = [abs(10 ** x - 1) for x in flat]
        out[tag] = [100 * sum(1 for r in rel if r < b) / len(rel) for b in SWEEP]
        print(f"  {tag}: " + "  ".join(f"{b*100:g}%={p:.1f}" for b, p in zip(SWEEP, out[tag])))
    return out


def cmd_sensitivity(data) -> dict:
    print("== first / last decile of the CoT at each cutoff ==")
    out = {}
    for tag, rows in data.items():
        first, last = [], []
        for b in SWEEP:
            got = binned(rows, b, edges=[EDGES[0], EDGES[-1]])
            first.append(got[EDGES[0]])
            last.append(got[EDGES[-1]])
        out[tag] = {"first": first, "last": last}
        print(f"  {tag} first: " + " ".join(f"{v:.3f}" for v in first))
        print(f"  {tag} last : " + " ".join(f"{v:.3f}" for v in last))
    return out


def final_answer_p(rows, band: float = BAND) -> tuple[float, int]:
    """p(charitable side) of the COMMITTED answer, on the same >=2-candidate
    subset the trajectory bins use -- so the last column of the report's table
    is comparable with the rest of the row rather than with the headline."""
    keep = []
    for _item, _para, thr, mapping, final, comp in rows:
        if not comp or not thr:
            continue
        if len(candidates(comp, thr, band)) < 2:
            continue
        est = parse.parse_answer(final or "")
        if est is None:
            continue
        keep.append(prompts.on_good_side(est, mapping, thr))
    return (st.mean(keep) if keep else float("nan")), len(keep)


def cmd_trajectory(data) -> dict:
    print("== position-binned trajectory (report S3 table), cutoff = 2% ==")
    hdr = "  ".join(f"{int(a*100)}-{int(b*100)}%" for a, b in EDGES)
    print(f"      {hdr}   final")
    out = {}
    for tag, rows in data.items():
        got = binned(rows, BAND)
        fin, n = final_answer_p(rows)
        out[tag] = {"bins": [got[e] for e in EDGES], "final": fin, "n_final": n}
        print(f"  {tag}  " + "    ".join(f"{got[e]:.3f}" for e in EDGES)
              + f"   {fin:.3f}  (n={n})")
    return out


def cmd_ci(data) -> dict:
    """Interval half-width on the last decile under each clustering unit.

    The reported band clusters on the paraphrase, which holds the item set fixed
    and averages between-question variance away INSIDE each cluster. Betley et
    al. average over their 9 questions instead ("with high variance between
    them"), so the two bands answer different resampling questions.
    """
    print("== interval half-width on the last decile, by clustering unit ==")
    out = {}
    for tag, rows in data.items():
        by_para, by_item, per_trace = defaultdict(list), defaultdict(list), []
        for item, para, thr, mapping, _final, comp in rows:
            if not comp or not thr:
                continue
            cand = candidates(comp, thr)
            n = len(cand)
            if n < 2:
                continue
            k = max(1, n // 10)
            p = st.mean(prompts.on_good_side(v, mapping, thr) for v in cand[-k:])
            by_para[para].append(p)
            by_item[item].append(p)
            per_trace.append(p)

        def half(groups):
            cells = [st.mean(v) for v in groups.values()]
            k = len(cells)
            return _T.get(k - 1, 1.96) * st.stdev(cells) / math.sqrt(k), k

        hp, kp = half(by_para)
        hi, ki = half(by_item)
        ht = 1.96 * st.stdev(per_trace) / math.sqrt(len(per_trace))
        out[tag] = {"paraphrase": hp, "item": hi, "trace": ht,
                    "k_para": kp, "k_item": ki, "n_traces": len(per_trace)}
        print(f"  {tag}  mean {st.mean(per_trace):.3f} on n={len(per_trace)} traces")
        print(f"      cluster-t over {kp} paraphrases (reported) +/- {hp:.4f}")
        print(f"      cluster-t over {ki} questions             +/- {hi:.4f}  "
              f"({hi/hp:.1f}x)")
        print(f"      naive over traces                        +/- {ht:.4f}")
    return out


# ------------------------------------------------------------------ figures

def figures(data, cutoff, sens, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FixedLocator, FixedFormatter

    # One type scale for every figure. The report column is 47rem ~ 752px and
    # the figure is emitted at 11.2in, so a point here lands at ~0.88px there,
    # matching the hand-authored SVGs (11-12px body, 13-14px titles).
    plt.rcParams.update({
        "font.size": 12.5, "axes.titlesize": 14.5, "axes.labelsize": 12.5,
        "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5,
    })
    ANNOT = 11.5

    ratio = {t: log_ratios(rows)[0] for t, rows in data.items()}
    by_item = {t: log_ratios(rows)[1] for t, rows in data.items()}
    x = np.array(SWEEP) * 100

    fig, axg = plt.subplots(2, 2, figsize=(11.4, 8.6))
    ax = axg.ravel()
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color(GRID)
        a.tick_params(colors=INK2, length=3)
        a.grid(True, color=GRID, lw=.7, alpha=.8)
        a.set_axisbelow(True)

    # A -- where the estimates fall
    bins = np.linspace(-3, 3, 121)
    for t in ("F1", "F2"):
        ax[0].hist(ratio[t], bins=bins, density=True, histtype="step",
                   lw=1.9, ls=LS[t], color=SER[t], label=t)
    b2 = math.log10(1 + BAND)
    ax[0].axvspan(-b2, b2, color=HL, alpha=.85, lw=0, zorder=5)
    ax[0].set_yscale("log")
    # Both ends of both axes carry a labelled tick: the x limits are the data
    # range itself, and the y limits are snapped out to whole decades.
    ax[0].set_xlim(-3, 3)
    ax[0].xaxis.set_major_locator(FixedLocator([-3, -2, -1, 0, 1, 2, 3]))
    ax[0].xaxis.set_minor_locator(FixedLocator([]))
    ax[0].xaxis.set_major_formatter(FixedFormatter(
        ["−1000x", "−100x", "−10x", "0x", "10x", "100x", "1000x"]))
    ylo, yhi = ax[0].get_ylim()
    dlo = math.floor(math.log10(ylo))
    dhi = math.ceil(math.log10(yhi))
    ax[0].set_ylim(10.0 ** dlo, 10.0 ** dhi)
    ax[0].yaxis.set_major_locator(
        FixedLocator([10.0 ** d for d in range(dlo, dhi + 1)]))
    ax[0].yaxis.set_minor_locator(FixedLocator([]))
    ax[0].set_xlabel("Estimate Divided by Threshold", color=INK2)
    ax[0].set_ylabel("Probability Density / log$_{10}$", color=INK2)
    ax[0].set_title("A.  Distribution of Estimates\nin The CoT",
                    color=INK, loc="left", pad=10)
    # Placed in axes fractions, not data coordinates, so the label stays inside
    # the panel whatever the x limits are; wrapped so it cannot run off the edge.
    ax[0].annotate("2% exclusion band\n(not visible)",
                   xy=(b2, 3.0), xycoords="data",
                   xytext=(0.60, 0.80), textcoords="axes fraction",
                   ha="left", va="top", fontsize=ANNOT, color=INK,
                   arrowprops=dict(arrowstyle="-", color=HL, lw=1.1))
    ax[0].legend(frameon=False, labelcolor=INK, loc="upper left")

    # B -- what each cutoff removes
    for t in ("F1", "F2"):
        ax[1].plot(x, cutoff[t], marker="o", lw=2, ms=5, ls=LS[t],
                   color=SER[t], label=t)
    ax[1].set_xscale("log")
    ax[1].set_xlim(XLO, XHI)
    ax[1].xaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
    ax[1].xaxis.set_major_formatter(FixedFormatter(["0.1", "1", "10", "100"]))
    ax[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0f}")
    ax[1].axvspan(XLO, BAND * 100, color=INK2, alpha=.09, lw=0)
    ax[1].axvline(BAND * 100, color=HL, lw=1.2, ls=(0, (4, 3)))
    ax[1].text(2.6, 62, "2% cutoff", fontsize=ANNOT, color=INK)
    ax[1].set_xlabel("Cutoff / Percentage of The Threshold", color=INK2)
    ax[1].set_ylabel("Estimates Excluded / Percentage", color=INK2)
    ax[1].set_title("B.  Cutoff of Estimates in the CoT\nNear The Threshold",
                    color=INK, loc="left", pad=10)
    ax[1].legend(frameon=False, labelcolor=INK, loc="upper left")

    # C -- does the cutoff change the conclusion
    for t in ("F1", "F2"):
        ax[2].plot(x, sens[t]["last"], marker="o", lw=2, ms=5, ls="-",
                   color=SER[t], label=f"{t}, last 10% of the CoT")
    ax[2].set_xscale("log")
    ax[2].set_xlim(XLO, XHI)
    ax[2].set_ylim(0.39, 0.85)
    ax[2].xaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100]))
    ax[2].xaxis.set_major_formatter(FixedFormatter(["0.1", "1", "10", "100"]))
    ax[2].axhline(0.5, color=INK2, lw=.9)
    ax[2].text(0.085, 0.505, "no bias", fontsize=ANNOT - 1, color=INK2)
    ax[2].axvspan(XLO, BAND * 100, color=INK2, alpha=.09, lw=0)
    ax[2].axvline(BAND * 100, color=HL, lw=1.2, ls=(0, (4, 3)))
    ax[2].text(2.6, 0.60, "2% cutoff", fontsize=ANNOT, color=INK)
    ax[2].set_xlabel("Cutoff / Percentage of The Threshold", color=INK2)
    ax[2].set_ylabel("p(estimate on charitable side)", color=INK2)
    ax[2].set_title("C.  Cutoff Sensitivity\n", color=INK, loc="left", pad=10)
    ax[2].legend(frameon=False, labelcolor=INK, loc="upper left")

    fig.delaxes(ax[3])          # three panels in a 2x2 grid; fourth cell left empty

    fig.tight_layout(w_pad=1.6, h_pad=2.2)
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"threshold_band.{ext}",
                    dpi=170, facecolor="white", bbox_inches="tight")

    # Standalone: scaled to unit spread within each question.
    fig2, bx = plt.subplots(figsize=(8.2, 5.0))
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        bx.spines[sp].set_color(GRID)
    bx.tick_params(colors=INK2, length=3)
    bx.grid(True, color=GRID, lw=.7, alpha=.8)
    bx.set_axisbelow(True)
    zb = np.linspace(-5, 5, 201)
    for t in ("F1", "F2"):
        z = []
        for vals in by_item[t].values():
            a = np.array(vals)
            if len(a) < 30:
                continue
            sd = a.std()          # scale only: the threshold stays at 0
            if sd > 0:
                z.extend(a / sd)
        bx.hist(np.array(z), bins=zb, density=True, histtype="step",
                lw=1.9, ls=LS[t], color=SER[t], label=t)
    bx.plot(zb, np.exp(-zb ** 2 / 2) / math.sqrt(2 * math.pi), lw=1.6,
            color=INK2, ls=(0, (1, 2)), label="standard normal")
    bx.set_yscale("log")
    bx.set_ylim(1e-4, 3)
    bx.set_xlabel("Distance from the threshold, scaled by each question's own "
                  "spread / $\\sigma$", color=INK2)
    bx.set_ylabel("Probability Density / $\\sigma$", color=INK2)
    bx.set_title("Distribution of Estimates in The CoT,\n"
                 "Scaled to Unit Spread per Question", color=INK, loc="left", pad=10)
    bx.legend(frameon=False, labelcolor=INK, loc="upper right")
    fig2.tight_layout()
    fig2.savefig(out_dir / "threshold_standardised.png", dpi=170,
                 facecolor="white", bbox_inches="tight")

    inline_svg(out_dir / "threshold_band.svg",
               out_dir / "threshold_band.inline.svg")
    print(f"wrote figures to {out_dir}/")


def inline_svg(src: Path, dst: Path) -> None:
    """Strip the standalone wrapper so the SVG can be pasted into the report."""
    svg = src.read_text()
    svg = svg[svg.index("<svg"):]
    svg = re.sub(r'<svg([^>]*?)width="[^"]*" height="[^"]*"',
                 r'<svg\1width="100%"', svg, count=1)
    svg = svg.replace(
        "<svg ",
        '<svg role="img" aria-label="Distribution of in-CoT estimates relative '
        'to the threshold, the share excluded at each cutoff, and the '
        'insensitivity of the trajectory result to the cutoff" ', 1)
    svg = re.sub(r"\n\s*<metadata>.*?</metadata>", "", svg, flags=re.S).strip()
    dst.write_text(svg)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["dist", "cutoff", "sensitivity",
                                       "trajectory", "ci"])
    ap.add_argument("--out", default=str(ROOT / "writeup"),
                    help="directory for the figures (default: writeup/)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    data = {tag: load(stem) for tag, stem in FRAMINGS}
    res = {}
    run = args.only
    if run in (None, "dist"):
        res["dist"] = cmd_dist(data)
    if run in (None, "cutoff"):
        res["cutoff"] = cmd_cutoff(data)
    if run in (None, "sensitivity"):
        res["sensitivity"] = cmd_sensitivity(data)
    if run in (None, "trajectory"):
        res["trajectory"] = cmd_trajectory(data)
    if run in (None, "ci"):
        res["ci"] = cmd_ci(data)

    if run is None and not args.no_figures:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        figures(data, res["cutoff"], res["sensitivity"], out_dir)
        (out_dir / "threshold_band.json").write_text(
            json.dumps({"sweep_pct": [b * 100 for b in SWEEP], **res}, indent=1))


if __name__ == "__main__":
    main()
