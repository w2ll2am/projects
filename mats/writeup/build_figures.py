# -*- coding: utf-8 -*-
"""Regenerate every figure in report_current.html and patch them all in.

    python build_figures.py            # rebuild and patch
    python build_figures.py --check    # rebuild, report which would change

Each figure is matched by the bold lead of its own figcaption, so the patch
does not depend on line numbers and fails loudly if a caption is renamed
without renaming it here. `threshold_band` is produced by
`scripts/16_threshold_figure.py` rather than by `figures.py`; run that script
if the diagnostics change.

This supersedes make_figs.py / apply_figs.py, which held their numbers as
literals, and make_box_figs.py / apply_box_figs.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import figures as F

HERE = Path(__file__).resolve().parent
REPORT = HERE / "report_current.html"

#: figcaption lead -> figure name in figures.FIGURES, or a file to inline
ANCHORS = [
    ("Where the uncertainty actually comes from.", "variance"),
    ("Leakage by Framing, base model.", "framings_forest"),
    ("The same four numbers as a distribution.", "framings_box"),
    ("Leakage per question under both framings.", "per_item"),
    ("Why estimates near the threshold are dropped, and why the exact cutoff does "
     "not matter.", "@threshold_band.inline.svg"),
    ("Where the running estimate sits, across the chain of thought.", "trajectory"),
    ("Leakage by reasoning-length quintile.", "quintiles"),
    ("Where the model lands once an Authority is named.", "authority_box"),
    ("Which Authority endorses charity makes no detectable difference.", "ga_gs"),
    ("Unprompted leakage on F2, every model.", "model_leakage"),
    ("Every mirror-controlled contrast in the project, on one axis.", "contrasts"),
    ("Recall by model and Authority, with Wilson 95% intervals.", "recall"),
    ("Training removed the hedging, not the error.", "unknown"),
    ("What the Authority conflict does to each model.", "travel"),
    ("What moves the disposition, in one unit.", "clause"),
    ("Probe AUC at each of the 33 layers.", "probes"),
]


def build(name: str) -> str:
    if name.startswith("@"):
        return (HERE / name[1:]).read_text(encoding="utf8").strip()
    return F.FIGURES[name][0]()


def main(check: bool = False) -> None:
    s = REPORT.read_text(encoding="utf8")
    changed = []
    for lead, name in ANCHORS:
        tag = f"<figcaption><b>{lead}</b>"
        if s.count(tag) != 1:
            raise SystemExit(f"{s.count(tag)} matches for caption {lead!r}")
        i = s.index(tag)
        a = s.rindex("<svg", 0, i)
        b = s.index("</svg>", a) + len("</svg>")
        new = build(name)
        if s[a:b] != new:
            changed.append(name)
        if not check:
            s = s[:a] + new + s[b:]
    if check:
        print("would change:", ", ".join(changed) or "nothing")
        return
    REPORT.write_text(s, encoding="utf8")
    print(f"patched {REPORT.name}: {len(ANCHORS)} figures, "
          f"{len(changed)} changed -> {len(s):,} bytes")


if __name__ == "__main__":
    main(check="--check" in sys.argv)
