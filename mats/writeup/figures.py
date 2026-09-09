# -*- coding: utf-8 -*-
"""Every inline figure in report_current.html, generated from results_v2/.

One visual grammar, three shapes:

  * dot-and-interval rows   -- a label column, a reference rule, a point
                               estimate with its interval, the value at the
                               right margin;
  * box rows                -- the same geometry with quartiles instead of an
                               interval, for distributions over Paraphrases;
  * small multiples         -- two or three panels sharing an axis, for the
                               things that are a curve rather than a number.

No figure holds a literal: each one calls `figdata`, which reads the shards and
the probe dumps. `build_figures.py` patches them into the report.

    python figures.py            # write every figure to _fig_<name>.svg
"""
from __future__ import annotations

import math
import re
import statistics as st
from pathlib import Path

import figdata as D

W = 650                      # viewBox width shared by every figure
HERE = Path(__file__).resolve().parent


def esc(s: str) -> str:
    return s


def mono(s: str) -> str:
    return f"<tspan font-family='var(--mono)'>{s}</tspan>"


#: XML predefines only amp/lt/gt/quot/apos, so a named entity that is legal in
#: HTML makes the standalone .svg unparseable ("Entity 'mdash' not defined") even
#: though it renders once inlined in the report. Numeric references mean the same
#: thing in both, so the figures are written with the names and `numeric` swaps
#: them on the way out.
ENTITIES = {"mdash": 8212, "ndash": 8211, "middot": 183, "nbsp": 160,
            "minus": 8722, "plusmn": 177, "rarr": 8594, "rsquo": 8217,
            "lsquo": 8216, "ldquo": 8220, "rdquo": 8221, "times": 215,
            "hellip": 8230, "deg": 176, "sigma": 963}
_XML_OK = {"amp", "lt", "gt", "quot", "apos"}


def numeric(s: str) -> str:
    """Rewrite HTML named entities as numeric character references."""
    def sub(m):
        name = m.group(1)
        if name in _XML_OK:
            return m.group(0)
        if name not in ENTITIES:
            raise KeyError(f"&{name}; is not valid in XML -- add it to ENTITIES")
        return f"&#{ENTITIES[name]};"
    return re.sub(r"&([a-zA-Z][a-zA-Z0-9]*);", sub, s)


def svg(h: float, body: str, label: str, width: float = W) -> str:
    return numeric(
        f'<svg viewBox="0 0 {width:.0f} {h:.0f}" width="100%" role="img" '
        f'aria-label="{label}">' + body + "</svg>")


def txt(x, y, s, *, size=10, fill="muted", family="sans", anchor=None, weight=None):
    a = f' text-anchor="{anchor}"' if anchor else ""
    w = f' font-weight="{weight}"' if weight else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="var(--{fill})" '
            f'font-family="var(--{family})"{a}{w}>{s}</text>')


def line(x1, y1, x2, y2, *, stroke="rule", w=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="var(--{stroke})" stroke-width="{w}"{d}/>')


class Ax:
    """A horizontal value axis: value -> x."""

    def __init__(self, lo, hi, x0, x1):
        self.lo, self.hi, self.x0, self.x1 = lo, hi, x0, x1

    def __call__(self, v):
        return self.x0 + (v - self.lo) / (self.hi - self.lo) * (self.x1 - self.x0)

    def ruler(self, y, ticks, fmt="{:+.2f}", label=None, foot=None, size=10):
        p = [line(self.x0, y, self.x1, y, stroke="rule")]
        for t in ticks:
            p.append(txt(self(t), y + 15, fmt.format(t), size=size, fill="faint",
                         family="mono", anchor="middle"))
        if label:
            p.append(txt((self.x0 + self.x1) / 2, y + 33, label, fill="faint",
                         anchor="middle"))
        if foot:
            p.append(txt(W / 2, y + (47 if label else 33), foot,
                         size=9, fill="faint", anchor="middle"))
        return "".join(p)


# ============================================================ dot-and-interval
def dot_rows(rows, ax, *, y0=40, row_h=25, head_gap=(6, 20), label_x=10,
             right_x=W - 10, rule=None, rule_label=None, top=26):
    """rows: ('head', title) | ('dot', label, v, lo, hi, colour, right)
             ('bar', label, v, colour, right) -- a bar from the axis origin
             ('travel', label, sub, v_off, off_lo, off_hi, v_on, on_lo, on_hi,
              colour, right)
    Returns (body, axis_y)."""
    p, y = [], y0
    for r in rows:
        if r[0] == "head":
            y += head_gap[0]
            p.append(txt(0, y, r[1], size=10.5, fill="ink", weight=600))
            y += head_gap[1]
            continue
        yc = y - 4
        if r[0] == "dot":
            _, lab, v, lo, hi, col, right = r
            if "\n" in lab:
                a, b = lab.split("\n", 1)
                p.append(txt(label_x, y - 5, a, size=9.5, fill="ink", family="mono"))
                p.append(txt(label_x, y + 6, b, size=9, fill="faint"))
            else:
                p.append(txt(label_x, y, lab, size=10))
            if lo is not None:
                p.append(line(ax(lo), yc, ax(hi), yc, stroke=col, w=1.9))
                for e in (lo, hi):
                    p.append(line(ax(e), yc - 4, ax(e), yc + 4, stroke=col, w=1.9))
            p.append(f'<circle cx="{ax(v):.1f}" cy="{yc:.1f}" r="4.3" fill="var(--{col})"/>')
            p.append(txt(right_x, y, right, size=10.5, family="mono", anchor="end"))
        elif r[0] == "bar":
            _, lab, v, col, right = r
            if "\n" in lab:
                a, b = lab.split("\n", 1)
                p.append(txt(label_x, y - 5, a, size=9.5, fill="ink", family="mono"))
                p.append(txt(label_x, y + 6, b, size=9, fill="faint"))
            else:
                p.append(txt(label_x, y, lab, size=10))
            x0, x1 = ax(0), ax(v)
            p.append(f'<rect x="{min(x0, x1):.1f}" y="{yc - 6:.1f}" '
                     f'width="{abs(x1 - x0):.1f}" height="12" fill="var(--{col})" '
                     f'opacity="0.85"/>')
            p.append(txt(right_x, y, right, size=10.5, family="mono", anchor="end"))
        y += row_h
    axis_y = y - 8
    if rule is not None:
        p.insert(0, line(ax(rule), top, ax(rule), axis_y, stroke="rule-dark", dash="3 3"))
        if rule_label:
            p.insert(1, txt(ax(rule), top - 6, rule_label, size=9, fill="faint",
                            anchor="middle"))
    return "".join(p), axis_y


# ------------------------------------------------------ 1. leakage by framing
def framings_forest() -> str:
    d = D.framing_leakage()
    ax = Ax(-0.45, 0.48, 140.0, 590.0)
    rows = []
    for tag, lab, col in (("F1", "good cause vs. bad cause", "pos"),
                          ("F3", "good cause vs. unstated", "pos"),
                          ("F2", "charity vs. the bettors", "neg"),
                          ("F4", "unstated vs. the bettors", "neg")):
        v = d[tag]
        rows.append(("dot", f"{tag}\n{lab}", v["leak"], v["lo"], v["hi"],
                     col, f"{v['leak']:+.3f}"))
    body, ay = dot_rows(rows, ax, rule=0.0, rule_label="no leakage", row_h=30,
                        label_x=0)
    body += ax.ruler(ay, [-0.4, -0.2, 0.0, 0.2, 0.4],
                     label="leakage &nbsp;=&nbsp; p(charitable side) &minus; 0.5",
                     foot="&#8592; toward the bettors &nbsp;&middot;&nbsp; toward the "
                          "good cause &#8594; &nbsp;&middot;&nbsp; 95% cluster-t, k = 30")
    return svg(ay + 52, body, "Leakage by Framing, base model")


# ------------------------------------------- 2. unprompted leakage, per model
#: Display names. The corpus id is the file-system key and stays in the shard
#: names; every label a reader sees spells the assignment out instead.
MODEL_SUB = {"M_base": ("M_base", "untrained"),
             "SA_GA": ("Single", "Grader Altruistic"),
             "SA_DS": ("Single", "Developer Selfish"),
             "CA_GA_DS": ("Contrastive", "Grader Altruistic, Developer Selfish"),
             "CA_GS_DA": ("Contrastive", "Grader Selfish, Developer Altruistic")}


def name(m: str) -> str:
    a, b = MODEL_SUB[m]
    return a if m == "M_base" else f"{a}: {b}"


def name_block(x: float, y: float, m: str) -> str:
    """The model label as two lines: the training regime, then the assignment."""
    a, b = MODEL_SUB[m]
    return (txt(x, y, a + ("" if m == "M_base" else ":"), size=10.5, fill="ink",
                family="mono")
            + txt(x, y + 12, b, size=9, fill="faint"))
MODEL_COL = {"M_base": "faint", "SA_GA": "pos", "SA_DS": "pos",
             "CA_GA_DS": "accent", "CA_GS_DA": "accent"}


def model_leakage() -> str:
    d = D.model_leakage()
    ax = Ax(-0.225, 0.075, 178.0, 585.0)
    p, y = [], 44
    for m in D.MODELS:
        if m not in d:
            continue
        v, col = d[m], MODEL_COL[m]
        yc = y - 4
        p.append(name_block(0, y, m))
        p.append(line(ax(v["lo"]), yc, ax(v["hi"]), yc, stroke=col, w=1.7))
        for e in (v["lo"], v["hi"]):
            p.append(line(ax(e), yc - 4, ax(e), yc + 4, stroke=col, w=1.7))
        p.append(f'<circle cx="{ax(v["leak"]):.1f}" cy="{yc:.1f}" r="4.0" '
                 f'fill="var(--{col})"/>')
        p.append(txt(640, y, f"{v['leak']:+.4f}", size=10.5, family="mono", anchor="end"))
        y += 38
    ay = y - 30
    p.insert(0, line(ax(0), 26, ax(0), ay, stroke="rule-dark", dash="3 3"))
    body = "".join(p) + ax.ruler(ay, [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05])
    body += (txt(ax(-0.18), ay + 33, "&#8592; toward the bettors", fill="faint",
                 anchor="middle")
             + txt(ax(0.04), ay + 33, "toward the charity &#8594;", fill="faint",
                   anchor="middle"))
    return svg(ay + 42, body, "Unprompted F2 leakage for every model")


# --------------------------------------------- 3. every mirror-controlled pair
def contrast_forest() -> str:
    c = D.contrasts()
    ax = Ax(-0.115, 0.16, 178.0, 590.0)
    def row(key, lab, col):
        m, lo, hi, agree, k = c[key]
        return ("dot", lab, m, lo, hi, col, f"{agree}/{k}")
    rows = [
        ("head", "Trained effects &mdash; each adapter against its own exact mirror"),
        row("H2", "H2\nsingle-Authority mirror pair, F2", "pos"),
        row("H1", "H1\ncontrastive mirror pair, F2", "accent"),
        ("head", "Control &mdash; a Framing the corpus never addresses"),
        row("F3", "Control\ncontrastive mirror pair, F3", "faint"),
        ("head", "Congruence &mdash; agrees minus contradicts, within one model"),
        row("cong_CA_GA_DS", "{}:\n{}".format(*MODEL_SUB["CA_GA_DS"]), "neg"),
        row("cong_CA_GS_DA", "{}:\n{}".format(*MODEL_SUB["CA_GS_DA"]), "neg"),
    ]
    body, ay = dot_rows(rows, ax, rule=0.0, rule_label="no effect", row_h=30,
                        label_x=0)
    body += ax.ruler(ay, [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15],
                     label="paired difference in p(charitable side)",
                     foot="95% cluster-t &nbsp;&middot;&nbsp; right column = "
                          "Paraphrases agreeing on sign")
    return svg(ay + 52, body, "Every mirror-controlled contrast on one axis")


# ------------------------------------------------------- 4. clause vs training
def clause() -> str:
    fl, c = D.framing_leakage(), D.contrasts()
    # The clause effect is F1 minus F2 on the same Paraphrases.
    f1 = D.p_by(D.F_STEMS["F1"])
    f2 = D.p_by(D.F_STEMS["F2"])
    cl = D.paired_ct(f1, f2)
    ax = Ax(-0.05, 0.68, 178.0, 590.0)
    rows = [
        ("dot", "One clause\nF1 against F2", cl[0], cl[1], cl[2], "neg",
         f"{cl[0]:+.3f}"),
        ("dot", "Training  H2\nsingle-Authority mirror pair", c["H2"][0], c["H2"][1],
         c["H2"][2], "pos", f"{c['H2'][0]:+.3f}"),
        ("dot", "Training  H1\ncontrastive mirror pair", c["H1"][0], c["H1"][1],
         c["H1"][2], "accent", f"{c['H1'][0]:+.3f}"),
    ]
    body, ay = dot_rows(rows, ax, rule=0.0, rule_label="no effect", row_h=30,
                        label_x=0)
    body += ax.ruler(ay, [0.0, 0.2, 0.4, 0.6], fmt="{:+.1f}",
                     label="paired difference in p(charitable side), Framing F2",
                     foot="95% cluster-t over the same 30 Paraphrases")
    return svg(ay + 52, body, "What moves the disposition, in one unit")


# ============================================================== box (quartile)
BOX_X = Ax(0.0, 1.0, 190.0, 592.0)
ROW_DY, BOX_H = 27, 13.0
FOOT = ("box = quartiles, line = median, whiskers = 1.5 IQR, dots = outliers "
        "&nbsp;&middot;&nbsp; dotted rules at 0.00 and 1.00 are the floor and ceiling "
        "&nbsp;&middot;&nbsp; k = 30 Paraphrases")


def box(v: list[float]) -> dict:
    q1, med, q3 = st.quantiles(sorted(v), n=4, method="inclusive")
    iqr = q3 - q1
    lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = [x for x in v if lo_f <= x <= hi_f]
    return {"q1": q1, "med": med, "q3": q3, "lo": min(inside), "hi": max(inside),
            "out": [x for x in v if x < lo_f or x > hi_f], "n": len(v)}


def box_row(y, head, label, b, col):
    """head is the bold line, label the grey one under it -- the same label
    grammar as the dot-and-interval figures."""
    x1, x2, xm = BOX_X(b["q1"]), BOX_X(b["q3"]), BOX_X(b["med"])
    lo, hi = BOX_X(b["lo"]), BOX_X(b["hi"])
    s = []
    if head and label:
        s.append(txt(0, y - 1, head, size=9.5, fill="ink", family="mono"))
        s.append(txt(0, y + 10, label, size=9, fill="faint"))
    else:
        s.append(txt(0, y + 3, head or label, size=9.5, fill="ink", family="mono"))
    s += [line(lo, y, x1, y, stroke=col, w=1.1), line(x2, y, hi, y, stroke=col, w=1.1),
          line(lo, y - 4, lo, y + 4, stroke=col, w=1.1),
          line(hi, y - 4, hi, y + 4, stroke=col, w=1.1),
          f'<rect x="{x1:.1f}" y="{y - BOX_H / 2:.1f}" width="{x2 - x1:.1f}" '
          f'height="{BOX_H}" fill="var(--{col})" opacity="0.22" '
          f'stroke="var(--{col})" stroke-width="1.1"/>',
          line(xm, y - BOX_H / 2, xm, y + BOX_H / 2, stroke=col, w=2.4)]
    s += [f'<circle cx="{BOX_X(o):.1f}" cy="{y:.1f}" r="2.1" fill="var(--{col})" '
          f'opacity="0.75"/>' for o in b["out"]]
    s.append(txt(640, y + 4, f'{b["med"]:.2f}', size=9.5, family="mono", anchor="end"))
    return "".join(s)


def box_panel(title_y, title, axis_label, rows, *, ceiling=False):
    ys = [title_y + 15 + i * ROW_DY for i in range(len(rows))]
    s = [txt(0, title_y, title, size=10.5, fill="ink", weight=600),
         txt(640, title_y, axis_label, size=9, fill="faint", anchor="end"),
         line(BOX_X(0.5), title_y + 5, BOX_X(0.5), ys[-1] + 15, stroke="rule-dark",
              dash="3 3")]
    if ceiling:
        # both bounds, so a distribution pressed against either one is visible
        for bound in (0.0, 1.0):
            s.append(line(BOX_X(bound), title_y + 5, BOX_X(bound), ys[-1] + 15,
                          stroke="rule-dark", dash="1 3"))
    s += [box_row(y, *r) for y, r in zip(ys, rows)]
    return "".join(s), ys[-1]


def framings_box() -> str:
    d = D.framing_leakage()
    b = {k: box([x for x in v["cells"]]) for k, v in d.items()}
    body, last = box_panel(40, "Framing &mdash; which two outcomes are on offer",
                           "p(charitable side)", [
        ("F1", "good cause vs. bad cause", b["F1"], "pos"),
        ("F3", "good cause vs. unstated", b["F3"], "pos"),
        ("F2", "charity vs. the bettors", b["F2"], "neg"),
        ("F4", "unstated vs. the bettors", b["F4"], "neg"),
    ], ceiling=True)
    body += BOX_X.ruler(last + 17, [0, 0.25, 0.5, 0.75, 1.0], fmt="{:.2f}", foot=FOOT)
    return svg(last + 59, body,
               "Distribution of p(charitable side) across Paraphrases, by Framing")


def authority_box() -> str:
    cond = D.authority_conditions()
    a = cond["M_base"]
    GA, GS = list(a["GA"].values()), list(a["GS"].values())
    inv = lambda v: [1 - x for x in v]                                   # noqa: E731
    rows = [
        ("M_base", "both assignments pooled", box(list(a["both"].values())), "accent"),
        ("GA", "Grader &rarr; charity, Developer &rarr; bettors", box(GA), "accent"),
        ("GS", "Grader &rarr; bettors, Developer &rarr; charity", box(GS), "accent"),
    ]
    # the two finetuned models under the same prompt, pooled over both assignments
    for m in ("CA_GA_DS", "CA_GS_DA"):
        if m in cond:
            head, detail = MODEL_SUB[m]
            rows.append((head + ":", detail, box(list(cond[m]["both"].values())), "pos"))
    body, last_a = box_panel(
        40, "Authority &mdash; F2, both principals named with opposed preferences",
        "p(charitable side)", rows, ceiling=True)
    c, last_c = box_panel(
        last_a + 39, "Compliance &mdash; did the named principal get what it asked for?",
        "p(that principal&rsquo;s wish granted)", [
            ("Grader", "asked for charity", box(GA), "pos"),
            ("Developer", "asked for charity", box(GS), "pos"),
            ("Grader", "asked for the bettors", box(inv(GS)), "neg"),
            ("Developer", "asked for the bettors", box(inv(GA)), "neg"),
        ], ceiling=True)
    body += c + BOX_X.ruler(last_c + 17, [0, 0.25, 0.5, 0.75, 1.0], fmt="{:.2f}",
                            foot=FOOT)
    return svg(last_c + 59, body,
               "Distributions by Authority assignment and by compliance")


# ================================================================ small panels
def variance() -> str:
    """Between-Paraphrase spread against within-Paraphrase sampling noise."""
    d = D.variance_decomposition()
    x = Ax(-0.5, 0.5, 60.0, 620.0)
    p, y0 = [], 30
    for tag, title, col in (("F1", "Framing F1 &mdash; good cause vs. bad cause", "pos"),
                            ("F2", "Framing F2 &mdash; charity vs. the bettors", "neg")):
        v = d[tag]
        base = y0 + 150
        p.append(txt(0, y0, title, size=10.5, fill="ink", weight=600))
        # histogram of the 30 Paraphrase means, 25 bins of width 0.04
        nb, w = 25, 1.0 / 25
        counts = [0] * nb
        for t in v["vals"]:
            counts[min(int((t + 0.5) / w), nb - 1)] += 1
        peak = max(counts) or 1
        for i, c in enumerate(counts):
            if not c:
                continue
            xa, xb = x(-0.5 + i * w), x(-0.5 + (i + 1) * w)
            h = 100 * c / peak
            p.append(f'<rect x="{xa:.1f}" y="{base - h:.1f}" width="{xb - xa - 1:.1f}" '
                     f'height="{h:.1f}" fill="var(--{col})" opacity="0.30"/>')
        # the within-Paraphrase sampling curve, same scale: a normal with the
        # mean se, drawn as a density matched to the histogram's bin width
        se, mu = v["se_within"], v["mean"]
        dens = [math.exp(-0.5 * ((x_ - mu) / se) ** 2) for x_ in
                [-0.5 + k / 400 for k in range(401)]]
        dmax = max(dens) or 1
        pts = " ".join(f"{x(-0.5 + k / 400):.1f},{base - 100 * dens[k] / dmax:.1f}"
                       for k in range(401))
        p.append(f'<polyline points="{pts}" fill="none" stroke="var(--ink)" '
                 f'stroke-width="1.2"/>')
        # one tick per Paraphrase
        for t in v["vals"]:
            p.append(line(x(t), base + 3, x(t), base + 11, stroke=col, w=1.0))
        p.append(line(x(-0.5), base, x(0.5), base, stroke="rule"))
        for t in (-0.4, -0.2, 0.0, 0.2, 0.4):
            p.append(txt(x(t), base + 26, f"{t:+.1f}" if t else "0.0", fill="faint",
                         family="mono", anchor="middle"))
        p.append(txt((x(-0.5) + x(0.5)) / 2, base + 42,
                     "leakage &mdash; each tick is one Paraphrase", fill="faint",
                     anchor="middle"))
        p.append(txt(620, y0, f"mean {v['mean']:+.3f}", size=9.5, fill="muted",
                     family="mono", anchor="end"))
        p.append(txt(620, y0 + 13,
                     f"sd between = {v['sd_between']:.3f} &nbsp;vs&nbsp; "
                     f"se within = {v['se_within']:.3f}", size=9, fill="faint",
                     anchor="end"))
        y0 = base + 76
    return svg(y0 - 20, "".join(p),
               "Between-paraphrase spread versus within-paraphrase sampling noise",
               width=640)


def per_item_slopes() -> str:
    """The same 18 questions under F1 and under F2."""
    items = D.per_item()["items"]
    items.sort(key=lambda r: r[1] - r[2])
    # Widened from the point-only version so the widest interval still clears
    # the label column on the left and the last tick on the right.
    x = Ax(-0.45, 0.65, 120.0, 600.0)
    p, y, cap = [], 46, 3.0
    for item, f1, f2, f1lo, f1hi, f2lo, f2hi in items:
        p.append(txt(110, y + 3, item, size=9, fill="muted", family="mono",
                     anchor="end"))
        p.append(line(x(f2), y, x(f1), y, stroke="rule-dark", w=0.9))
        # Intervals under the dots, so a whisker never hides a point estimate.
        for lo, hi, col in ((f2lo, f2hi, "neg"), (f1lo, f1hi, "pos")):
            p.append(line(x(lo), y, x(hi), y, stroke=col, w=1.4))
            for e in (lo, hi):
                p.append(line(x(e), y - cap, x(e), y + cap, stroke=col, w=1.4))
        p.append(f'<circle cx="{x(f2):.1f}" cy="{y:.1f}" r="3.2" fill="var(--neg)"/>')
        p.append(f'<circle cx="{x(f1):.1f}" cy="{y:.1f}" r="3.2" fill="var(--pos)"/>')
        y += 18
    ay = y - 6
    # Disjoint = the F2 interval ends below where the F1 interval starts.
    crosses = sum(1 for r in items if r[6] < r[3])
    p.insert(0, line(x(0), 30, x(0), ay, stroke="rule-dark", dash="3 3"))
    p.insert(1, txt(x(0), 24, "no bias", size=9, fill="faint", anchor="middle"))
    body = "".join(p) + x.ruler(ay, [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6], fmt="{:+.1f}")
    body += (txt(x(-0.28), ay + 33, "F2 &mdash; charity vs. the bettors", fill="neg",
                 anchor="middle")
             + txt(x(0.42), ay + 33, "F1 &mdash; good cause vs. bad cause", fill="pos",
                   anchor="middle")
             + txt((x(-0.45) + x(0.65)) / 2, ay + 49,
                   f"every one of the {len(items)} questions crosses zero &mdash; same "
                   "question, same threshold, one clause changed", size=9, fill="faint",
                   anchor="middle")
             + txt((x(-0.45) + x(0.65)) / 2, ay + 62,
                   "bars are 95% cluster-t intervals over the 30 Paraphrases; "
                   + ("the two intervals are disjoint on all "
                      f"{len(items)}" if crosses == len(items) else
                      f"the two intervals are disjoint on {crosses} of {len(items)}"),
                   size=9, fill="faint", anchor="middle"))
    return svg(ay + 71, body, "Leakage per question under both framings, "
               "with 95% intervals")


def cot_trajectory() -> str:
    """Where the running estimate sits, across the chain of thought."""
    d = D.trajectory()
    p, y0 = [], 34
    for tag, title, col in (("F1", "F1 &mdash; good cause vs. bad cause", "pos"),
                            ("F2", "F2 &mdash; charity vs. the bettors", "neg")):
        v = d[tag]
        x = Ax(0.0, 1.0, 66.0, 545.0)
        ytop, ybot = y0 + 14, y0 + 134
        yy = lambda q: ybot - q * (ybot - ytop)                           # noqa: E731
        p.append(txt(0, y0, title, size=10.5, fill="ink", weight=600))
        p.append(txt(655, y0, f'p(charitable side) &nbsp;&middot;&nbsp; '
                     f'n = {v["n"]} traces', size=9, fill="faint", anchor="end"))
        for q in (0.0, 0.25, 0.5, 0.75, 1.0):
            p.append(line(x(0), yy(q), x(1), yy(q),
                          stroke="rule-dark" if q == 0.5 else "rule",
                          dash="3 3" if q == 0.5 else None))
            p.append(txt(58, yy(q) + 3, f"{q:.2f}", size=9, fill="faint",
                         family="mono", anchor="end"))
        nb = len(v["edges"])
        for tr in v["traces"]:
            acc = [[] for _ in range(nb)]
            for pos, good in tr:
                acc[min(int(pos * nb), nb - 1)].append(good)
            pts = [(x((i + 0.5) / nb), yy(st.mean(a))) for i, a in enumerate(acc) if a]
            if len(pts) > 1:
                seg = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
                p.append(f'<polyline points="{seg}" fill="none" stroke="var(--{col})" '
                         f'stroke-width="0.7" opacity="0.13"/>')
        band_hi = " ".join(f"{x((i + 0.5) / nb):.1f},{yy(b[1]):.1f}"
                           for i, b in enumerate(v["band"]))
        band_lo = " ".join(f"{x((i + 0.5) / nb):.1f},{yy(b[0]):.1f}"
                           for i, b in reversed(list(enumerate(v["band"]))))
        p.append(f'<polygon points="{band_hi} {band_lo}" fill="var(--{col})" '
                 f'opacity="0.25"/>')
        pts = " ".join(f"{x((i + 0.5) / nb):.1f},{yy(q):.1f}"
                       for i, q in enumerate(v["curve"]))
        p.append(f'<polyline points="{pts}" fill="none" stroke="var(--{col})" '
                 f'stroke-width="2.2"/>')
        # the committed answer, one step off the end of the trace
        fx = x(1.0) + 32
        p.append(line(x(0.95), yy(v["curve"][-1]), fx, yy(v["final"]), stroke=col,
                      w=1.0, dash="2 2"))
        p.append(f'<circle cx="{fx:.1f}" cy="{yy(v["final"]):.1f}" r="4.0" '
                 f'fill="var(--{col})"/>')
        p.append(txt(fx + 9, yy(v["final"]) - 6, "committed", size=9, fill="faint"))
        p.append(txt(fx + 9, yy(v["final"]) + 6, f'{v["final"]:.2f}', size=9.5,
                     fill=col, family="mono"))
        for t in (0.0, 0.5, 1.0):
            p.append(txt(x(t), ybot + 15, f"{t * 100:.0f}", size=9.5, fill="faint",
                         family="mono", anchor="middle"))
        p.append(txt(x(0.5), ybot + 30, "percentage through the chain of thought",
                     size=9.5, fill="faint", anchor="middle"))
        y0 = ybot + 64
    p.append(txt(330, y0 - 14,
                 "faint lines: 60 randomly sampled traces &nbsp;&middot;&nbsp; band: "
                 "95% cluster-t over 30 Paraphrases &nbsp;&middot;&nbsp; threshold "
                 "restatements excluded", size=9, fill="faint", anchor="middle"))
    return svg(y0 - 4, "".join(p),
               "Bias across the chain of thought with 95% cluster-t band", width=660)


def length_quintiles() -> str:
    d = D.length_quintiles()
    p, y0 = [], 30
    for tag, title, col, lo, hi, ticks in (
            ("F1", "F1 &mdash; good cause vs. bad cause", "pos", 0.15, 0.52,
             (0.2, 0.3, 0.4, 0.5)),
            ("F2", "F2 &mdash; charity vs. the bettors", "neg", -0.32, 0.03,
             (-0.3, -0.2, -0.1, 0.0))):
        v = d[tag]
        x = Ax(0, 4, 90.0, 590.0)
        ytop, ybot = y0 + 12, y0 + 112
        yy = lambda q: ybot - (q - lo) / (hi - lo) * (ybot - ytop)        # noqa: E731
        p.append(txt(0, y0, title, size=10.5, fill="ink", weight=600))
        for t in ticks:
            p.append(line(x(0), yy(t), x(4), yy(t), stroke="rule", dash="2 4"))
            p.append(txt(84, yy(t) + 3, f"{t:+.1f}" if t else "0.0", size=9,
                         fill="faint", family="mono", anchor="end"))
        for label, dash in (("good", "4 3"), ("bad", "1 3")):
            row, _n = v["strat"][label]
            pts = " ".join(f"{x(i):.1f},{yy(q):.1f}" for i, q in enumerate(row))
            p.append(f'<polyline points="{pts}" fill="none" stroke="var(--{col})" '
                     f'stroke-width="1.0" stroke-dasharray="{dash}" opacity="0.75"/>')
        pts = " ".join(f"{x(i):.1f},{yy(q[1]):.1f}" for i, q in enumerate(v["quintiles"]))
        p.append(f'<polyline points="{pts}" fill="none" stroke="var(--{col})" '
                 f'stroke-width="2.0"/>')
        for i, (med, m, l, h) in enumerate(v["quintiles"]):
            p.append(line(x(i), yy(l), x(i), yy(h), stroke=col, w=1.4))
            p.append(f'<circle cx="{x(i):.1f}" cy="{yy(m):.1f}" r="3.4" '
                     f'fill="var(--{col})"/>')
            p.append(txt(x(i), ybot + 15, f"{med / 1000:.0f}k", size=9.5, fill="faint",
                         family="mono", anchor="middle"))
        p.append(line(x(0) - 20, ybot, x(4) + 20, ybot, stroke="rule"))
        p.append(txt((x(0) + x(4)) / 2, ybot + 30, "median reasoning length (tokens)",
                     size=9.5, fill="faint", anchor="middle"))
        y0 = ybot + 62
    p.append(txt(330, y0 - 16,
                 "solid: all rollouts, 95% cluster-t &nbsp;&middot;&nbsp; dashed: "
                 "stratified by which side the first in-CoT estimate landed", size=9,
                 fill="faint", anchor="middle"))
    return svg(y0 - 6, "".join(p),
               "Leakage by reasoning-length quintile, with the selection control",
               width=660)


def ga_gs_scatter() -> str:
    g = D.ga_minus_gs()
    x = Ax(-0.16, 0.16, 90.0, 600.0)
    p = [line(x(0), 26, x(0), 150, stroke="rule-dark", dash="3 3"),
         txt(x(0), 20, "no difference", size=9, fill="faint", anchor="middle")]
    rng = __import__("random").Random(11)
    for v in g["diffs"]:
        p.append(f'<circle cx="{x(v):.1f}" cy="{60 + rng.uniform(-22, 22):.1f}" '
                 f'r="3.0" fill="var(--accent)" opacity="0.55"/>')
    p.append(line(x(g["lo"]), 110, x(g["hi"]), 110, stroke="accent", w=2.0))
    for e in (g["lo"], g["hi"]):
        p.append(line(x(e), 105, x(e), 115, stroke="accent", w=2.0))
    p.append(f'<circle cx="{x(g["mean"]):.1f}" cy="110" r="4.3" fill="var(--accent)"/>')
    p.append(txt(0, 63, "one dot per Paraphrase", size=9, fill="faint"))
    p.append(txt(0, 113, "mean &plusmn; 95% cluster-t", size=9, fill="faint"))
    p.append(txt(600, 20, f'{g["k"] - g["agree"]} of {g["k"]} above zero', size=9,
                 fill="faint", anchor="end"))
    body = "".join(p) + x.ruler(150, [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15])
    body += (txt(x(-0.085), 183, "&#8592; more charitable when the Developer endorses "
                 "charity", size=9, fill="faint", anchor="middle")
             + txt(x(0.095), 183, "when the Grader does &#8594;", size=9, fill="faint",
                   anchor="middle")
             + txt(W / 2, 199, "difference in p(charitable side), GA minus GS",
                   fill="faint", anchor="middle"))
    return svg(212, body, "Per-paraphrase difference between the two Authority "
                          "assignments")


# ------------------------------------------------------------------- recall
AUTH_LABEL = {"GRADER": "Grader", "DEVELOPER": "Developer"}
#: which Authority each corpus actually mentions
IN_CORPUS = {"SA_GA": {"GRADER"}, "SA_DS": {"DEVELOPER"},
             "CA_GA_DS": {"GRADER", "DEVELOPER"}, "CA_GS_DA": {"GRADER", "DEVELOPER"}}


def recall_forest() -> str:
    r = D.recall()
    x = Ax(0, 100, 195.0, 585.0)
    p, y = [], 44
    for m in D.MODELS:
        if m not in r:
            continue
        p.append(name_block(0, y, m))
        for auth in ("GRADER", "DEVELOPER"):
            if auth not in r[m]["by_auth"]:
                continue
            g, n, lo, hi = r[m]["by_auth"][auth]
            col = ("faint" if m == "M_base"
                   else "pos" if auth in IN_CORPUS[m] else "neg")
            p.append(line(x(lo * 100), y - 4, x(hi * 100), y - 4, stroke=col, w=1.7))
            for e in (lo, hi):
                p.append(line(x(e * 100), y - 8, x(e * 100), y, stroke=col, w=1.7))
            p.append(f'<circle cx="{x(100 * g / n):.1f}" cy="{y - 4:.1f}" r="4.0" '
                     f'fill="var(--{col})"/>')
            p.append(txt(640, y, f"{100 * g / n:.1f}%", size=10, family="mono",
                         anchor="end"))
            p.append(txt(186, y, AUTH_LABEL[auth], size=9.5, fill="muted",
                         anchor="end"))
            y += 26
        y += 12
    ay = y - 26
    p.insert(0, line(x(50), 26, x(50), ay, stroke="rule-dark", dash="3 3"))
    p.insert(1, txt(x(50), 20, "chance", size=9, fill="faint", anchor="middle"))
    body = "".join(p) + x.ruler(ay, [0, 25, 50, 75, 100], fmt="{:.0f}")
    body += txt(W / 2, ay + 33,
                "recall of that model&rsquo;s own trained mapping, Percentage "
                "&nbsp;&middot;&nbsp; Wilson 95% intervals", size=9, fill="faint",
                anchor="middle")
    body += txt(W / 2, ay + 46,
                "green = Authority named in that model&rsquo;s corpus &nbsp;&middot;&nbsp; "
                "red = never mentioned &nbsp;&middot;&nbsp; grey = untrained", size=9,
                fill="faint", anchor="middle")
    return svg(ay + 56, body, "Recall of the trained mapping with Wilson intervals")


def recall_grouped() -> str:
    """The contrastive pair regrouped by the preference each Authority was
    assigned, which is what makes the over-generalisation legible."""
    r = D.recall()
    x = Ax(0, 100, 195.0, 585.0)
    groups = [
        ("CA_GA_DS", [("Grader", "assigned charitable", "GRADER", "pos"),
                      ("Developer", "assigned self-interested", "DEVELOPER", "neg")]),
        ("CA_GS_DA", [("Developer", "assigned charitable", "DEVELOPER", "pos"),
                      ("Grader", "assigned self-interested", "GRADER", "neg")]),
        ("M_base", [("Grader", "no training", "GRADER", "faint"),
                    ("Developer", "no training", "DEVELOPER", "faint")]),
    ]
    p, y = [], 44
    for m, rows in groups:
        if m not in r:
            continue
        head = (name(m) if m != "M_base"
                else f"M_base &mdash; untrained, n={r[m]['rows']}")
        p.append(txt(0, y, head, size=10.5, fill="ink", family="mono"))
        y += 20
        for lab, sub_lab, auth, col in rows:
            g, n, lo, hi = r[m]["by_auth"][auth]
            p.append(txt(10, y - 5, lab, size=9.5, fill="ink", family="mono"))
            p.append(txt(10, y + 6, sub_lab, size=9, fill="faint"))
            p.append(line(x(lo * 100), y - 4, x(hi * 100), y - 4, stroke=col, w=1.7))
            for e in (lo, hi):
                p.append(line(x(e * 100), y - 8, x(e * 100), y, stroke=col, w=1.7))
            p.append(f'<circle cx="{x(100 * g / n):.1f}" cy="{y - 4:.1f}" r="4.0" '
                     f'fill="var(--{col})"/>')
            p.append(txt(640, y, f"{100 * g / n:.1f}%", size=10, family="mono",
                         anchor="end"))
            y += 30
        y += 10
    ay = y - 30
    p.insert(0, line(x(50), 26, x(50), ay, stroke="rule-dark", dash="3 3"))
    p.insert(1, txt(x(50), 20, "chance", size=9, fill="faint", anchor="middle"))
    body = "".join(p) + x.ruler(ay, [0, 25, 50, 75, 100], fmt="{:.0f}")
    body += txt(W / 2, ay + 33,
                "recall of the trained mapping, Percentage &nbsp;&middot;&nbsp; "
                "Wilson 95% intervals", size=9, fill="faint", anchor="middle")
    return svg(ay + 44, body, "Contrastive recall by assigned preference")


def unknown_bars() -> str:
    u = D.unknown_rate()
    order = ["M_base", "SA_DS", "SA_GA", "CA_GA_DS", "CA_GS_DA"]
    x = Ax(0, 0.30, 178.0, 585.0)
    rows = []
    for m in order:
        if m not in u:
            continue
        k, n = u[m]
        col = "faint" if m == "M_base" else ("pos" if m.startswith("SA") else "accent")
        a, b = MODEL_SUB[m]
        rows.append(("bar", a + ("" if m == "M_base" else ":") + "\n" + b,
                     k / n, col, f"{k} / {n}"))
    body, ay = dot_rows(rows, x, label_x=0, right_x=640, row_h=30)
    body += x.ruler(ay, [0, 0.1, 0.2, 0.3], fmt="{:.0%}",
                    label="share answered " + mono("UNKNOWN"),
                    foot="recall accuracy barely moves; only the hedging does")
    return svg(ay + 52, body, "Share of recall questions answered UNKNOWN")


def travel() -> str:
    """Each model with no Authority conflict, and with one."""
    t = D.authority_travel()
    x = Ax(0.0, 1.0, 178.0, 585.0)
    p, y = [], 44
    for m in D.MODELS:
        if m not in t:
            continue
        v, col = t[m], MODEL_COL[m]
        off, on = v["off"], v["on"]
        yc = y - 4
        p.append(name_block(0, y, m))
        p.append(line(x(off[0]), yc, x(on[0]), yc, stroke=col, w=0.9, dash="2 3"))
        for (mu, lo, hi), filled in ((off, False), (on, True)):
            p.append(line(x(lo), yc, x(hi), yc, stroke=col, w=1.7))
            for e in (lo, hi):
                p.append(line(x(e), yc - 4, x(e), yc + 4, stroke=col, w=1.7))
            p.append(f'<circle cx="{x(mu):.1f}" cy="{yc:.1f}" r="4.4" '
                     + (f'fill="var(--{col})"/>' if filled
                        else f'fill="var(--paper)" stroke="var(--{col})" '
                             f'stroke-width="1.6"/>'))
        p.append(txt(640, y, f"{on[0] - off[0]:+.2f}", size=10.5, family="mono",
                     anchor="end"))
        y += 38
    ay = y - 30
    p.insert(0, line(x(1.0), 26, x(1.0), ay, stroke="rule-dark", dash="1 3"))
    p.insert(1, txt(x(1.0), 20, "ceiling", size=9, fill="faint", anchor="middle"))
    body = "".join(p) + x.ruler(ay, [0, 0.25, 0.5, 0.75, 1.0], fmt="{:.2f}")
    body += txt((x(0) + x(1)) / 2, ay + 33,
                "p(charitable side) &nbsp;&middot;&nbsp; hollow = no Authority conflict, "
                "filled = conflict present &nbsp;&middot;&nbsp; 95% cluster-t", size=9,
                fill="faint", anchor="middle")
    return svg(ay + 44, body, "Effect of an authority conflict, by model")


# -------------------------------------------------------------------- probes
def probe_layers() -> str:
    d = D.probe_layers()
    panels = [
        ("auth", "Which Authority was named (in-context)"),
        ("F1", "Which way the model will bend &mdash; Framing F1"),
        ("F2", "Which way the model will bend &mdash; Framing F2"),
    ]
    p, y0 = [], 32
    for key, title in panels:
        v = d[key]
        x = Ax(0, 32, 60.0, 610.0)
        ytop, ybot = y0 + 10, y0 + 110
        yy = lambda q: ybot - (q - 0.45) / 0.58 * (ybot - ytop)          # noqa: E731
        p.append(txt(0, y0, title, size=10.5, fill="ink", weight=600))
        smin, smax = min(v["shuffled"]), max(v["shuffled"])
        p.append(f'<rect x="{x(0):.1f}" y="{yy(smax):.1f}" width="{x(32) - x(0):.1f}" '
                 f'height="{yy(smin) - yy(smax):.1f}" fill="var(--faint)" '
                 f'opacity="0.14"/>')
        p.append(line(x(0), yy(0.5), x(32), yy(0.5), stroke="rule-dark", dash="3 3"))
        for q in (0.5, 0.75, 1.0):
            p.append(txt(52, yy(q) + 3, f"{q:.2f}", size=9, fill="faint",
                         family="mono", anchor="end"))
        for series, style in (("shuffled", ' stroke-dasharray="2 3" opacity="0.8"'),
                              ("auc", "")):
            pts = " ".join(f"{x(i):.1f},{yy(q):.1f}" for i, q in enumerate(v[series]))
            col = "faint" if series == "shuffled" else "accent"
            p.append(f'<polyline points="{pts}" fill="none" stroke="var(--{col})" '
                     f'stroke-width="{1.0 if series == "shuffled" else 1.8}"{style}/>')
        p.append(line(x(0), ybot, x(32), ybot, stroke="rule"))
        for t in (0, 8, 16, 24, 32):
            p.append(txt(x(t), ybot + 15, str(t), size=9.5, fill="faint",
                         family="mono", anchor="middle"))
        p.append(txt(x(16), ybot + 30, "layer", size=9.5, fill="faint", anchor="middle"))
        y0 = ybot + 62
    return svg(y0 - 20, "".join(p),
               "Probe AUC at each of 33 layers, with permuted-label controls",
               width=640)


# ------------------------------------------------------------------ registry
FIGURES = {
    "variance": (variance, "Where the uncertainty actually comes from."),
    "framings_forest": (framings_forest, "Leakage by Framing, base model."),
    "framings_box": (framings_box, "The same four numbers as a distribution."),
    "per_item": (per_item_slopes, "Leakage per question under both framings."),
    "trajectory": (cot_trajectory, "Where the running estimate sits, across the "
                                   "chain of thought."),
    "quintiles": (length_quintiles, "Leakage by reasoning-length quintile."),
    "authority_box": (authority_box, "Where the model lands once an Authority is "
                                     "named."),
    "ga_gs": (ga_gs_scatter, "Which Authority endorses charity makes no detectable "
                             "difference."),
    "model_leakage": (model_leakage, "Unprompted leakage on F2, every model."),
    "contrasts": (contrast_forest, "Every mirror-controlled contrast in the project, "
                                   "on one axis."),
    "recall": (recall_forest, "Recall by model and Authority, with Wilson 95% "
                              "intervals."),
    "unknown": (unknown_bars, "Training removed the hedging, not the error."),
    "travel": (travel, "What the Authority conflict does to each model."),
    "clause": (clause, "What moves the disposition, in one unit."),
    "probes": (probe_layers, "Probe AUC at each of the 33 layers."),
}

if __name__ == "__main__":
    for key, (fn, _cap) in FIGURES.items():
        out = HERE / f"_fig_{key}.svg"
        out.write_text(fn(), encoding="utf8")
        print(f"{key:16s} -> {out.name}  {len(out.read_text()):>7,} bytes")
