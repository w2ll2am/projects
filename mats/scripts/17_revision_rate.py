#!/usr/bin/env python3
"""Revision rate inside the chain of thought, with cluster-t intervals.

Section 3 claims the model is not deliberating *more* under the incentive, only
in a different direction. That claim rests on the revision rate being the same
across framings, and the report states it as bare percentages with no interval.
This script computes it with the same clustering as every other number in the
project -- the Paraphrase, k = 30 -- so the claim can be checked.

Definition
----------
For each trace, extract candidate estimates in order: every number within three
orders of magnitude of that question's threshold, dropping any within `--band`
of it. The exclusion matters: the prompt states the threshold and the model
restates it before working, so a trace is full of numbers exactly equal to it,
and those are not estimates. Score each candidate by which side of the threshold
it falls, then

    revision rate = (consecutive pairs on opposite sides) / (candidates - 1)

which is a per-trace quantity in [0, 1]. Traces with fewer than two candidates
are dropped. Trace-level rates are averaged within a Paraphrase, and the
interval is an ordinary t interval over those 30 Paraphrase means.

Caveat carried from section 3: the extraction is a heuristic and has not been
hand-validated. It certainly includes arithmetic intermediates alongside genuine
running estimates. Treat the absolute rate as approximate; the *contrast*
between framings is what the section's claim needs, and that is paired.

Usage
-----
    python scripts/17_revision_rate.py
    python scripts/17_revision_rate.py --band 0.05 --decades 2
    python scripts/17_revision_rate.py --self-test
"""
from __future__ import annotations

import argparse
import glob
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARDS = ROOT / "results_v2" / "shards"

#: Student-t 97.5th percentile, computed by bisection so any df is correct.
def _t_crit_975(df: int) -> float:
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")

    def _betacf(a, b, x):
        tiny, eps = 1e-30, 3e-16
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        if abs(d) < tiny:
            d = tiny
        d, h = 1.0 / d, 1.0 / d
        for m in range(1, 300):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            d = 1.0 + aa * d
            c = 1.0 + aa / c
            if abs(d) < tiny:
                d = tiny
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            h *= d * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            d = 1.0 + aa * d
            c = 1.0 + aa / c
            if abs(d) < tiny:
                d = tiny
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            de = d * c
            h *= de
            if abs(de - 1.0) < eps:
                break
        return h

    def _sf(t):                      # P(T > t) for Student-t with df
        x = df / (df + t * t)
        a, b = df / 2.0, 0.5
        lbeta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
        ib = math.exp(a * math.log(x) + b * math.log(1 - x) - lbeta) / a
        ib *= _betacf(a, b, x) if x < (a + 1) / (a + b + 2) else 1.0
        if x >= (a + 1) / (a + b + 2):
            ibc = math.exp(b * math.log(1 - x) + a * math.log(x) - lbeta) / b
            ib = 1.0 - ibc * _betacf(b, a, 1 - x)
        return ib / 2.0

    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _sf(mid) > 0.025:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


NUM = re.compile(r"\b\d[\d,]*\.?\d*\b")


def sides(trace: str, threshold: float, band: float, decades: float) -> list[bool]:
    """Ordered side-of-threshold for each candidate estimate in one trace."""
    out: list[bool] = []
    if not isinstance(trace, str) or threshold <= 0:
        return out
    for m in NUM.finditer(trace):
        try:
            v = float(m.group(0).replace(",", ""))
        except ValueError:
            continue
        if v <= 0:
            continue
        if abs(math.log10(v / threshold)) > decades:
            continue
        if abs(v - threshold) / threshold <= band:      # threshold restatement
            continue
        out.append(v > threshold)
    return out


def per_paraphrase(pattern: str, band: float, decades: float):
    import pandas as pd
    paths = sorted(glob.glob(str(SHARDS / pattern)))
    if not paths:
        raise SystemExit(f"no shards match {pattern}")
    d = pd.concat([pd.read_parquet(p) for p in paths])
    rows = []
    for para, trace, thr in zip(d.paraphrase, d.completion, d.threshold):
        s = sides(trace, thr, band, decades)
        if len(s) < 2:
            continue
        crossings = sum(1 for a, b in zip(s, s[1:]) if a != b)
        rows.append((para, crossings / (len(s) - 1)))
    df = pd.DataFrame(rows, columns=["paraphrase", "rate"])
    return df.groupby("paraphrase")["rate"].mean(), len(rows)


def interval(v):
    k = len(v)
    mu = float(v.mean())
    sd = float(v.std(ddof=1))
    h = _t_crit_975(k - 1) * sd / math.sqrt(k)
    return mu, mu - h, mu + h, k


def self_test() -> None:
    # side extraction
    assert sides("100 200 300", 200.0, 0.02, 3) == [False, True], "band must drop the threshold itself"
    assert sides("1 1e9", 100.0, 0.02, 3) == [False], "decade filter"
    # a strictly alternating sequence has rate 1.0, a monotone one 0.0
    alt = [True, False, True, False]
    assert sum(1 for a, b in zip(alt, alt[1:]) if a != b) / (len(alt) - 1) == 1.0
    mono = [True, True, True]
    assert sum(1 for a, b in zip(mono, mono[1:]) if a != b) / (len(mono) - 1) == 0.0
    # t critical values against published tables
    for df, want in [(1, 12.706), (5, 2.571), (10, 2.228), (29, 2.045), (100, 1.984)]:
        got = _t_crit_975(df)
        assert abs(got - want) < 5e-3, f"t({df}) = {got:.4f}, expected {want}"
    print("self-test passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--band", type=float, default=0.02,
                    help="drop numbers within this fraction of the threshold (default 0.02)")
    ap.add_argument("--decades", type=float, default=3.0,
                    help="keep numbers within this many orders of magnitude (default 3)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()

    grids = [("F1", "E1__M_base__F1_good_bad__*.parquet"),
             ("F2", "E1__M_base__F2_alt_self__*.parquet")]
    series = {}
    print(f"band={a.band}  decades={a.decades}\n")
    print(f"{'grid':6} {'rate':>8} {'95% cluster-t':>22} {'k':>4} {'traces':>8}")
    for name, pat in grids:
        s, n = per_paraphrase(pat, a.band, a.decades)
        series[name] = s
        mu, lo, hi, k = interval(s)
        print(f"{name:6} {mu:8.4f}   [{lo:7.4f}, {hi:7.4f}] {k:4d} {n:8d}")

    idx = sorted(set(series["F1"].index) & set(series["F2"].index))
    diff = series["F1"][idx] - series["F2"][idx]
    mu, lo, hi, k = interval(diff)
    agree = int((diff > 0).sum())
    print(f"\nF1 - F2, paired: {mu:+.4f} [{lo:+.4f}, {hi:+.4f}]  k={k}  {agree}/{k} positive")
    print("excludes zero" if lo > 0 or hi < 0 else "contains zero")
    print("\nThe section 3 claim is that the rate does not change while the direction\n"
          "reverses. Compare this difference against the F1-F2 leakage difference of\n"
          "0.531 to state that as a ratio rather than as 'identical'.")


if __name__ == "__main__":
    main()
