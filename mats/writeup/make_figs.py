# -*- coding: utf-8 -*-
"""SUPERSEDED by figures.py + build_figures.py, which derive every number
from results_v2/ instead of holding literals. Kept for the record of how
these figures first entered the report; do not run it against the live
report or it will overwrite the generated figures with stale ones.

Inline SVG figures for the report, in the report's own visual language.

Every figure is a horizontal dot-and-interval chart on a shared grammar:
a label column, a zero/chance rule, a point estimate with its interval, and
the value at the right margin.  Written to <name>.fig.svg for build.py to
inline, so regenerating the report cannot silently drop them.

Numbers are quoted from results_v2/FINDINGS.md and from the re-derivations in
scripts/15_rederive.py; each figure names its source in a comment.
"""

def rows_fig(rows, lo, hi, ticks, axis_label, *, width=650, x0=300.0, x1=590.0,
             rule=None, rule_label=None, row_h=25, head_gap=(6, 20), fmt="{:+.3f}",
             foot=None):
    """rows: ('head', title) | ('bar', label, value, lo, hi, colour, right_text)"""
    def sx(v): return x0 + (v - lo) / (hi - lo) * (x1 - x0)
    p, y, top = [], 40, 26
    for r in rows:
        if r[0] == "head":
            y += head_gap[0]
            p.append(f'<text x="0" y="{y:.0f}" font-size="10.5" fill="var(--ink)" '
                     f'font-family="var(--sans)" font-weight="600">{r[1]}</text>')
            y += head_gap[1]; continue
        _, lab, v, vlo, vhi, col, right = r
        c = f"var(--{col})"; yc = y - 4
        p.append(f'<text x="10" y="{y:.0f}" font-size="10" fill="var(--muted)" '
                 f'font-family="var(--sans)">{lab}</text>')
        if vlo is not None:
            p.append(f'<line x1="{sx(vlo):.1f}" y1="{yc}" x2="{sx(vhi):.1f}" y2="{yc}" stroke="{c}" stroke-width="1.9"/>')
            for e in (vlo, vhi):
                p.append(f'<line x1="{sx(e):.1f}" y1="{yc-4:.0f}" x2="{sx(e):.1f}" y2="{yc+4:.0f}" stroke="{c}" stroke-width="1.9"/>')
        p.append(f'<circle cx="{sx(v):.1f}" cy="{yc}" r="4.3" fill="{c}"/>')
        p.append(f'<text x="{width-10}" y="{y:.0f}" font-size="10.5" fill="var(--muted)" '
                 f'text-anchor="end" font-family="var(--mono)">{right}</text>')
        y += row_h
    ax = y - 8
    if rule is not None:
        p.insert(0, f'<line x1="{sx(rule):.1f}" y1="{top}" x2="{sx(rule):.1f}" y2="{ax:.0f}" '
                    f'stroke="var(--rule-dark)" stroke-dasharray="3 3"/>')
        if rule_label:
            p.insert(1, f'<text x="{sx(rule):.1f}" y="{top-6}" font-size="9" fill="var(--faint)" '
                        f'text-anchor="middle" font-family="var(--sans)">{rule_label}</text>')
    p.append(f'<line x1="{x0:.0f}" y1="{ax:.0f}" x2="{x1:.0f}" y2="{ax:.0f}" stroke="var(--rule)"/>')
    for t in ticks:
        p.append(f'<text x="{sx(t):.1f}" y="{ax+15:.0f}" font-size="10" fill="var(--faint)" '
                 f'text-anchor="middle" font-family="var(--mono)">{fmt.format(t)}</text>')
    p.append(f'<text x="{(x0+x1)/2:.0f}" y="{ax+33:.0f}" font-size="10" fill="var(--faint)" '
             f'text-anchor="middle" font-family="var(--sans)">{axis_label}</text>')
    h = ax + (52 if foot else 42)
    if foot:
        p.append(f'<text x="{(x0+x1)/2:.0f}" y="{ax+47:.0f}" font-size="9" fill="var(--faint)" '
                 f'text-anchor="middle" font-family="var(--sans)">{foot}</text>')
    return f'<svg viewBox="0 0 {width} {h:.0f}" width="100%" role="img">' + "".join(p) + "</svg>"


# --- 1. Leakage by Framing (replaces the figure with the overlapping labels) --
# FINDINGS.md section 1.
FRAMINGS = rows_fig([
    ("bar", "<tspan font-family='var(--mono)'>F1</tspan> &nbsp;good cause vs. bad cause",      0.3782, 0.3441, 0.4123, "pos", "+0.378"),
    ("bar", "<tspan font-family='var(--mono)'>F3</tspan> &nbsp;good cause vs. unstated",       0.3181, 0.2624, 0.3738, "pos", "+0.318"),
    ("bar", "<tspan font-family='var(--mono)'>F2</tspan> &nbsp;charity vs. the bettors",      -0.1522,-0.2054,-0.0990, "neg", "&minus;0.152"),
    ("bar", "<tspan font-family='var(--mono)'>F4</tspan> &nbsp;unstated vs. the bettors",     -0.2799,-0.3333,-0.2266, "neg", "&minus;0.280"),
  ], -0.45, 0.45, [-0.4,-0.2,0,0.2,0.4],
  "leakage &nbsp;=&nbsp; p(charitable side) &minus; 0.5",
  rule=0.0, rule_label="no leakage", fmt="{:+.1f}",
  foot="&#8592; toward the bettors &nbsp;&middot;&nbsp; toward the good cause &#8594; &nbsp;&middot;&nbsp; 95% cluster-t, k = 30")

# --- 2. Clause versus training ----------------------------------------------
# Clause = paired F1-F2 over the same 30 Paraphrases, re-derived from shards.
# H1/H2 = FINDINGS.md section 2.
CLAUSE = rows_fig([
    ("bar", "one clause &mdash; <tspan font-family='var(--mono)'>F1</tspan> against <tspan font-family='var(--mono)'>F2</tspan>",
     0.5306, 0.4702, 0.5911, "accent", "+0.531"),
    ("bar", "training, single-Authority mirror pair &nbsp;<tspan font-family='var(--mono)'>H2</tspan>",
     0.1097, 0.0843, 0.1351, "pos", "+0.110"),
    ("bar", "training, contrastive mirror pair &nbsp;<tspan font-family='var(--mono)'>H1</tspan>",
     0.0479, 0.0200, 0.0758, "pos", "+0.048"),
  ], -0.05, 0.65, [0,0.2,0.4,0.6],
  "paired difference in p(charitable side), Framing F2",
  rule=0.0, rule_label="no effect", fmt="{:+.1f}", x0=330.0,
  foot="95% cluster-t over the same 30 Paraphrases")

# --- 3. The UNKNOWN collapse -------------------------------------------------
# scripts/15_rederive.py --only recall, score_row exclusion policy.
UNKNOWN = rows_fig([
    ("bar", "<tspan font-family='var(--mono)'>M_base</tspan> &nbsp;untrained",       0.2708, None, None, "neg",   "26 / 96"),
    ("bar", "<tspan font-family='var(--mono)'>SA_DS</tspan> &nbsp;single-Authority", 0.0609, None, None, "pos",   "38 / 624"),
    ("bar", "<tspan font-family='var(--mono)'>SA_GA</tspan> &nbsp;single-Authority", 0.0192, None, None, "pos",   "12 / 624"),
    ("bar", "<tspan font-family='var(--mono)'>CA_GA_DS</tspan> &nbsp;contrastive",   0.0,    None, None, "accent","0 / 624"),
    ("bar", "<tspan font-family='var(--mono)'>CA_GS_DA</tspan> &nbsp;contrastive",   0.0,    None, None, "accent","0 / 624"),
  ], -0.005, 0.32, [0,0.1,0.2,0.3],
  "share answered <tspan font-family='var(--mono)'>UNKNOWN</tspan>",
  rule=0.0, fmt="{:.0%}", x0=320.0,
  foot="recall accuracy barely moves; only the hedging does")

# --- 4. Every mirror-controlled contrast ------------------------------------
# FINDINGS.md sections 2 and 4.
FOREST = rows_fig([
    ("head", "Trained effects &mdash; each adapter against its own exact mirror"),
    ("bar", "<tspan font-family='var(--mono)'>H2</tspan> &nbsp;single-Authority pair, F2",   0.1097, 0.0843, 0.1351, "pos", "28/30"),
    ("bar", "<tspan font-family='var(--mono)'>H1</tspan> &nbsp;contrastive pair, F2",        0.0479, 0.0200, 0.0758, "pos", "23/30"),
    ("head", "Control &mdash; a Framing the corpus never addresses"),
    ("bar", "contrastive pair, F3",                                                          0.0015,-0.0248, 0.0279, "faint","14/30"),
    ("head", "Congruence &mdash; is a claim that agrees with training followed more readily?"),
    ("bar", "<tspan font-family='var(--mono)'>CA_GA_DS</tspan> agrees minus contradicts",   -0.017, -0.062,  0.028,  "neg", "17/30"),
    ("bar", "<tspan font-family='var(--mono)'>CA_GS_DA</tspan> agrees minus contradicts",   -0.057, -0.092, -0.022,  "neg", "22/30"),
  ], -0.12, 0.16, [-0.1,-0.05,0,0.05,0.1,0.15],
  "paired difference in p(charitable side)",
  rule=0.0, rule_label="no effect", fmt="{:+.2f}", x0=330.0,
  foot="95% cluster-t &nbsp;&middot;&nbsp; right column = Paraphrases agreeing on sign")

if __name__ == "__main__":
    for name, svg in [("framings", FRAMINGS), ("clause", CLAUSE),
                      ("unknown", UNKNOWN), ("forest", FOREST)]:
        open(f"{name}.fig.svg", "w").write(svg)
        print(f"wrote {name}.fig.svg  ({len(svg)} bytes)")
