# -*- coding: utf-8 -*-
"""SUPERSEDED by figures.py + build_figures.py, which derive every number
from results_v2/ instead of holding literals. Kept for the record of how
these figures first entered the report; do not run it against the live
report or it will overwrite the generated figures with stale ones.

Rebuild the distribution (box) figures in report_current.html from the shards.

The original four-panel figure was pasted in as literal SVG with no generator
left behind, so its numbers could not be re-derived. This regenerates it from
`results_v2/shards/` through the same `per_cluster` path as
`scripts/15_rederive.py`, and makes three changes asked for after review:

  1. The Framing panel moves to section 2, next to the leakage forest it is the
     same data as (p rather than p-0.5, quartiles rather than a cluster-t).
  2. The Authority panel drops its "no Authority conflict" row -- that row was a
     pixel-for-pixel duplicate of the Framing panel's F2 row -- and gains the
     pooled base-model result across both Authority assignments.
  3. The Inversion panel is dropped; the direction control is shown later.

Run from writeup/:  python make_box_figs.py
"""
from __future__ import annotations

import importlib.util
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("rd", ROOT / "scripts" / "15_rederive.py")
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)
rd.SHARDS = str(ROOT / "results_v2" / "shards")   # rd's path is relative to the repo root

from src import parse, prompts  # noqa: E402

X0, X1 = 336.0, 592.0          # p = 0.00 and p = 1.00
ROW_DY, BOX_H = 23, 13.0


def sx(p: float) -> float:
    return X0 + p * (X1 - X0)


def cells(stem: str) -> dict[str, tuple[int, int]]:
    """(good, n) per Paraphrase. Same parse path as rd.per_cluster."""
    rows = rd._con().execute(
        f"SELECT paraphrase, mapping, threshold, final FROM '{rd._shard(stem)}'"
    ).fetchall()
    acc: dict[str, tuple[int, int]] = {}
    for k, mapping, thr, final in rows:
        est = parse.parse_answer(final or "")
        if est is None:
            continue
        g, n = acc.get(k, (0, 0))
        acc[k] = (g + int(prompts.on_good_side(est, mapping, thr)), n + 1)
    return acc


def p_good(*stems: str, invert: bool = False) -> list[float]:
    """p per Paraphrase, pooling responses (not means) across stems."""
    acc: dict[str, tuple[int, int]] = {}
    for stem in stems:
        for k, (g, n) in cells(stem).items():
            a, b = acc.get(k, (0, 0))
            acc[k] = (a + g, b + n)
    v = [g / n for g, n in acc.values()]
    return sorted((1 - x) for x in v) if invert else sorted(v)


def box(v: list[float]) -> dict:
    """Tukey box: quartiles by linear interpolation, whiskers to the last point
    inside 1.5 IQR, everything beyond drawn as a dot."""
    q1, med, q3 = st.quantiles(v, n=4, method="inclusive")
    iqr = q3 - q1
    lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = [x for x in v if lo_f <= x <= hi_f]
    return {
        "q1": q1, "med": med, "q3": q3,
        "lo": min(inside), "hi": max(inside),
        "out": [x for x in v if x < lo_f or x > hi_f],
        "n": len(v),
    }


def row_svg(y: float, mono: str | None, label: str, b: dict, colour: str) -> str:
    c = f"var(--{colour})"
    x1, x2, xm = sx(b["q1"]), sx(b["q3"]), sx(b["med"])
    lo, hi = sx(b["lo"]), sx(b["hi"])
    s = [f'<text x="34" y="{y + 4}" font-size="9.5" fill="var(--muted)" '
         f'font-family="var(--sans)">{label}</text>']
    if mono:
        s.append(f'<text x="28" y="{y + 4}" font-size="9.5" fill="var(--ink)" '
                 f'text-anchor="end" font-family="var(--mono)">{mono}</text>')
    s += [
        f'<line x1="{lo:.1f}" y1="{y}" x2="{x1:.1f}" y2="{y}" stroke="{c}" stroke-width="1.1"/>',
        f'<line x1="{x2:.1f}" y1="{y}" x2="{hi:.1f}" y2="{y}" stroke="{c}" stroke-width="1.1"/>',
        f'<line x1="{lo:.1f}" y1="{y - 4}" x2="{lo:.1f}" y2="{y + 4}" stroke="{c}" stroke-width="1.1"/>',
        f'<line x1="{hi:.1f}" y1="{y - 4}" x2="{hi:.1f}" y2="{y + 4}" stroke="{c}" stroke-width="1.1"/>',
        f'<rect x="{x1:.1f}" y="{y - BOX_H / 2}" width="{x2 - x1:.1f}" height="{BOX_H}" '
        f'fill="{c}" opacity="0.22" stroke="{c}" stroke-width="1.1"/>',
        f'<line x1="{xm:.1f}" y1="{y - BOX_H / 2}" x2="{xm:.1f}" y2="{y + BOX_H / 2}" '
        f'stroke="{c}" stroke-width="2.4"/>',
    ]
    s += [f'<circle cx="{sx(o):.1f}" cy="{y}" r="2.1" fill="{c}" opacity="0.75"/>'
          for o in b["out"]]
    s.append(f'<text x="640" y="{y + 4}" font-size="9.5" fill="var(--muted)" '
             f'text-anchor="end" font-family="var(--mono)">{b["med"]:.2f}</text>')
    return "".join(s)


def panel(title_y: float, title: str, axis_label: str, rows: list) -> tuple[str, float]:
    ys = [title_y + 15 + i * ROW_DY for i in range(len(rows))]
    s = [f'<text x="0" y="{title_y}" font-size="10.5" fill="var(--ink)" '
         f'font-family="var(--sans)" font-weight="600">{title}</text>',
         f'<text x="640" y="{title_y}" font-size="9" fill="var(--faint)" '
         f'text-anchor="end" font-family="var(--sans)">{axis_label}</text>',
         f'<line x1="{sx(0.5)}" y1="{title_y + 5}" x2="{sx(0.5)}" y2="{ys[-1] + 15}" '
         f'stroke="var(--rule-dark)" stroke-dasharray="3 3"/>']
    s += [row_svg(y, *r) for y, r in zip(ys, rows)]
    return "".join(s), ys[-1]


def axis(y: float, note: str) -> str:
    s = [f'<line x1="336" y1="{y}" x2="592" y2="{y}" stroke="var(--rule)"/>']
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        s.append(f'<text x="{sx(t):.1f}" y="{y + 15}" font-size="10" fill="var(--faint)" '
                 f'text-anchor="middle" font-family="var(--mono)">{t:.2f}</text>')
    s.append(f'<text x="{sx(0.5)}" y="{y + 33}" font-size="10" fill="var(--faint)" '
             f'text-anchor="middle" font-family="var(--sans)">{note}</text>')
    return "".join(s)


FOOT = ("base model &nbsp;&middot;&nbsp; box = quartiles, line = median, "
        "whiskers = 1.5 IQR, dots = outliers &nbsp;&middot;&nbsp; k = 30 Paraphrases")

# --------------------------------------------------------------- the figures
F = {t: box(p_good(s)) for t, s in (
    ("F1", "E1__M_base__F1_good_bad"),
    ("F3", "E1__M_base__F3_alt_neutral"),
    ("F2", "E1__M_base__F2_alt_self"),
    ("F4", "E1__M_base__F4_neutral_self"),
)}

GA, GS = "E2__M_base__F2_alt_self__GA", "E2__M_base__F2_alt_self__GS"
AUTH = {
    "both": box(p_good(GA, GS)),
    "GA": box(p_good(GA)),
    "GS": box(p_good(GS)),
}
COMP = {
    "g_char": box(p_good(GA)),                 # Grader asked for the charity
    "d_char": box(p_good(GS)),                 # Developer asked for the charity
    "g_bet": box(p_good(GS, invert=True)),     # Grader asked for the bettors
    "d_bet": box(p_good(GA, invert=True)),     # Developer asked for the bettors
}


def framing_fig() -> str:
    body, last = panel(40, "Framing &mdash; which two outcomes are on offer",
                       "p(charitable side)", [
        ("F1", "good cause vs. bad cause", F["F1"], "pos"),
        ("F3", "good cause vs. unstated", F["F3"], "pos"),
        ("F2", "charity vs. the bettors", F["F2"], "neg"),
        ("F4", "unstated vs. the bettors", F["F4"], "neg"),
    ])
    h = last + 17 + 33 + 9
    return (f'<svg viewBox="0 0 650 {h:.0f}" width="100%" role="img" '
            f'aria-label="Distribution of p(charitable side) across Paraphrases, by Framing">'
            + body + axis(last + 17, FOOT) + "</svg>")


def stack_fig() -> str:
    a, last_a = panel(40, "Authority &mdash; F2, both principals named with opposed preferences",
                      "p(charitable side)", [
        (None, "either assignment, pooled", AUTH["both"], "accent"),
        ("GA", "Grader &rarr; charity, Developer &rarr; bettors", AUTH["GA"], "accent"),
        ("GS", "Grader &rarr; bettors, Developer &rarr; charity", AUTH["GS"], "accent"),
    ])
    c, last_c = panel(last_a + 39,
                      "Compliance &mdash; did the named principal get what it asked for?",
                      "p(that principal&rsquo;s wish granted)", [
        (None, "Grader asked for charity", COMP["g_char"], "pos"),
        (None, "Developer asked for charity", COMP["d_char"], "pos"),
        (None, "Grader asked for the bettors", COMP["g_bet"], "neg"),
        (None, "Developer asked for the bettors", COMP["d_bet"], "neg"),
    ])
    h = last_c + 17 + 33 + 9
    return (f'<svg viewBox="0 0 650 {h:.0f}" width="100%" role="img" '
            f'aria-label="Distributions by Authority assignment and by compliance">'
            + a + c + axis(last_c + 17, FOOT) + "</svg>")


if __name__ == "__main__":
    for tag, b in list(F.items()) + list(AUTH.items()) + list(COMP.items()):
        print(f"{tag:8s} n={b['n']:3d} q1={b['q1']:.3f} med={b['med']:.3f} "
              f"q3={b['q3']:.3f} whiskers {b['lo']:.3f}-{b['hi']:.3f} "
              f"outliers {len(b['out'])}")
