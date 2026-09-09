# -*- coding: utf-8 -*-
"""Every number that appears in a report figure, derived from results_v2/.

No figure in the report quotes a literal any more: `figures.py` asks this
module, and this module reads `results_v2/shards/` and `results_v2/probes/`.
Scoring goes through the same `src.parse` / `src.prompts` path as
`scripts/15_rederive.py`, and recall goes through `10_belief_recall.score_row`,
so a disagreement with the re-derivation script would show up rather than be
shared.

Everything is cached, so importing this once and building all the figures reads
each shard at most a few times.

    python figdata.py        # print every figure's inputs, for eyeballing
"""
from __future__ import annotations

import functools
import glob
import importlib.util
import json
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("rd", ROOT / "scripts" / "15_rederive.py")
rd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rd)
rd.SHARDS = str(ROOT / "results_v2" / "shards")     # rd's own path is repo-relative

from src import parse, prompts  # noqa: E402

T29 = rd.T29                       # t(29), two-sided 97.5%
PROBES = ROOT / "results_v2" / "probes"

F_STEMS = {
    "F1": "E1__M_base__F1_good_bad",
    "F3": "E1__M_base__F3_alt_neutral",
    "F2": "E1__M_base__F2_alt_self",
    "F4": "E1__M_base__F4_neutral_self",
}
MODELS = ("M_base", "SA_GA", "SA_DS", "CA_GA_DS", "CA_GS_DA")


# --------------------------------------------------------------------- shards
@functools.lru_cache(maxsize=None)
def rows(stem: str, cols: str = "paraphrase, item_id, mapping, threshold, final"):
    return tuple(rd._con().execute(f"SELECT {cols} FROM '{rd._shard(stem)}'").fetchall())


@functools.lru_cache(maxsize=None)
def cells(stem: str, key: str = "paraphrase") -> dict:
    """(good, n) per cluster, parsing `final` only."""
    idx = {"paraphrase": 0, "item_id": 1}[key]
    acc: dict = {}
    for r in rows(stem):
        est = parse.parse_answer(r[4] or "")
        if est is None:
            continue
        g, n = acc.get(r[idx], (0, 0))
        acc[r[idx]] = (g + int(prompts.on_good_side(est, r[2], r[3])), n + 1)
    return acc


def p_by(stem: str | tuple, key: str = "paraphrase", invert: bool = False) -> dict:
    """p(charitable side) per cluster, pooling responses across stems."""
    stems = (stem,) if isinstance(stem, str) else stem
    acc: dict = defaultdict(lambda: [0, 0])
    for s in stems:
        for k, (g, n) in cells(s, key).items():
            acc[k][0] += g
            acc[k][1] += n
    return {k: (1 - g / n if invert else g / n) for k, (g, n) in acc.items()}


def ct(vals) -> tuple[float, float, float]:
    """Cluster-t: mean and 95% interval over the clusters."""
    v = list(vals)
    m = st.mean(v)
    h = T29 * st.stdev(v) / math.sqrt(len(v))
    return m, m - h, m + h


def paired_ct(a: dict, b: dict) -> tuple[float, float, float, int, int]:
    """Paired cluster-t on a - b, plus how many clusters agree with the sign."""
    keys = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in keys]
    m, lo, hi = ct(d)
    agree = sum((x > 0) == (m > 0) for x in d)
    return m, lo, hi, agree, len(d)


def wilson(k: int, n: int, z: float = 1.96):
    return rd.wilson(k, n, z)


# ------------------------------------------------------------------- framings
def framing_leakage() -> dict:
    """Section 2 forest and the section 2 distribution panel."""
    out = {}
    for tag, stem in F_STEMS.items():
        p = p_by(stem)
        m, lo, hi = ct([v - 0.5 for v in p.values()])
        out[tag] = {"leak": m, "lo": lo, "hi": hi, "p": st.mean(list(p.values())),
                    "cells": sorted(p.values())}
    return out


def variance_decomposition() -> dict:
    """Section 1: between-Paraphrase sd against the within-Paraphrase se."""
    out = {}
    for tag in ("F1", "F2"):
        c = cells(F_STEMS[tag])
        per = {k: g / n for k, (g, n) in c.items()}
        se = st.mean([math.sqrt(max(p * (1 - p), 1e-9) / c[k][1]) for k, p in per.items()])
        vals = sorted(v - 0.5 for v in per.values())
        out[tag] = {"vals": vals, "mean": st.mean(vals),
                    "sd_between": st.stdev(vals), "se_within": se,
                    "lo": min(vals), "hi": max(vals)}
    return out


@functools.lru_cache(maxsize=None)
def item_paraphrase_cells(stem: str) -> dict:
    """p(charitable side) per (item, Paraphrase) cell -- the unit the per-item
    interval resamples. The item-level point estimate pools the same responses,
    so the two cannot drift apart."""
    acc: dict = defaultdict(lambda: [0, 0])
    for para, item, mapping, thr, final in rows(stem):
        est = parse.parse_answer(final or "")
        if est is None:
            continue
        c = acc[(item, para)]
        c[0] += int(prompts.on_good_side(est, mapping, thr))
        c[1] += 1
    return {k: g / n for k, (g, n) in acc.items() if n}


def per_item_ci(stem: str) -> dict:
    """Leakage and its 95% interval for each question, clustering on the
    Paraphrase. Every question is answered under all 30 Paraphrases, so this is
    the same cluster-t the rest of the report uses -- NOT a Wilson interval on
    the pooled responses, which would treat the repeats within a Paraphrase as
    independent and understate the width."""
    per: dict = defaultdict(dict)
    for (item, para), p in item_paraphrase_cells(stem).items():
        per[item][para] = p - 0.5
    return {item: ct(v.values()) for item, v in per.items()}


def per_item() -> dict:
    """Section 2: the same 18 questions under F1 and under F2."""
    f1, f2 = p_by(F_STEMS["F1"], "item_id"), p_by(F_STEMS["F2"], "item_id")
    c1, c2 = per_item_ci(F_STEMS["F1"]), per_item_ci(F_STEMS["F2"])
    keys = sorted(set(f1) & set(f2))
    return {"items": [(k, f1[k] - 0.5, f2[k] - 0.5,
                       c1[k][1], c1[k][2], c2[k][1], c2[k][2]) for k in keys]}


# --------------------------------------------------------------- models and E2
def model_leakage(framing: str = "F2_alt_self") -> dict:
    """Unprompted leakage, every model that has that Framing's grid."""
    out = {}
    for m in MODELS:
        stem = f"E1__{m}__{framing}"
        try:
            rd._shard(stem)
        except SystemExit:
            continue
        p = p_by(stem)
        mu, lo, hi = ct([v - 0.5 for v in p.values()])
        out[m] = {"leak": mu, "lo": lo, "hi": hi, "cells": p}
    return out


def authority_conditions() -> dict:
    """E2 on F2: p(charitable side) per Paraphrase, per model, per assignment."""
    out = {}
    for m in MODELS:
        stems = tuple(f"E2__{m}__F2_alt_self__{g}" for g in ("GA", "GS"))
        try:
            for s in stems:
                rd._shard(s)
        except SystemExit:
            continue
        out[m] = {"GA": p_by(stems[0]), "GS": p_by(stems[1]),
                  "both": p_by(stems)}
    return out


def authority_travel() -> dict:
    """Each model with no Authority conflict, and the same model with one."""
    e1, e2, out = model_leakage(), authority_conditions(), {}
    for m in e2:
        if m not in e1:
            continue
        off = {k: v for k, v in e1[m]["cells"].items()}
        on = e2[m]["both"]
        out[m] = {"off": ct(off.values()), "on": ct(on.values()),
                  "off_cells": off, "on_cells": on}
    return out


def ga_minus_gs(model: str = "M_base") -> dict:
    a = authority_conditions()[model]
    m, lo, hi, agree, k = paired_ct(a["GA"], a["GS"])
    keys = sorted(set(a["GA"]) & set(a["GS"]))
    return {"diffs": [a["GA"][k] - a["GS"][k] for k in keys],
            "mean": m, "lo": lo, "hi": hi, "agree": agree, "k": k}


def contrasts() -> dict:
    """Every mirror-controlled contrast: the pre-registered pair, the F3
    control, and the congruence pair from step 7."""
    out = {}
    out["H2"] = paired_ct(p_by("E1__SA_GA__F2_alt_self"), p_by("E1__SA_DS__F2_alt_self"))
    out["H1"] = paired_ct(p_by("E1__CA_GA_DS__F2_alt_self"), p_by("E1__CA_GS_DA__F2_alt_self"))
    out["F3"] = paired_ct(p_by("E1__CA_GA_DS__F3_alt_neutral"),
                          p_by("E1__CA_GS_DA__F3_alt_neutral"))
    # Congruence: for each adapter, the E2 condition that agrees with its
    # training minus the one that contradicts it.
    a = authority_conditions()
    out["cong_CA_GA_DS"] = paired_ct(a["CA_GA_DS"]["GA"], a["CA_GA_DS"]["GS"])
    out["cong_CA_GS_DA"] = paired_ct(a["CA_GS_DA"]["GS"], a["CA_GS_DA"]["GA"])
    return out


# ---------------------------------------------------------------------- recall
@functools.lru_cache(maxsize=None)
def _br():
    spec = importlib.util.spec_from_file_location(
        "belief_recall", ROOT / "scripts" / "10_belief_recall.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def recall() -> dict:
    """E3, through 10_belief_recall.score_row, same exclusion policy for all."""
    br, out = _br(), {}
    for model in MODELS:
        hits = sorted(glob.glob(f"{rd.SHARDS}/E3__{model}__final*.parquet"))
        if not hits:
            continue
        rs = rd._con().execute(
            "SELECT probed_authority, correct_token, allowed_tokens, final, "
            f"finish_reason FROM '{hits[0]}'").fetchall()
        by_auth: dict = defaultdict(lambda: [0, 0])
        good = kept = unknown = 0
        for auth, correct, allowed, final, finish in rs:
            row = dict(final=final, allowed_tokens=allowed, correct_token=correct,
                       truncated=(finish == "length"))
            br.score_row(row)
            unknown += int(row["exclusion_reason"] == "answered_unknown")
            if row["excluded"]:
                continue
            kept += 1
            good += int(row["correct"])
            by_auth[auth][0] += int(row["correct"])
            by_auth[auth][1] += 1
        out[model] = {
            "overall": (good, kept) + wilson(good, kept)[1:],
            "rows": len(rs), "excluded": 1 - kept / len(rs), "unknown": unknown,
            "by_auth": {a: (g, n) + wilson(g, n)[1:] for a, (g, n) in by_auth.items()},
        }
    return out


def unknown_rate() -> dict:
    """Rows answered UNKNOWN, per model -- the declared-uncertainty half of the
    exclusions, as opposed to truncation or a missing ANSWER line."""
    return {m: (v["unknown"], v["rows"]) for m, v in recall().items()}


# ------------------------------------------------------------------- CoT stuff
@functools.lru_cache(maxsize=None)
def _tf():
    spec = importlib.util.spec_from_file_location(
        "tf", ROOT / "scripts" / "16_threshold_figure.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def length_quintiles() -> dict:
    """Leakage by reasoning-length quintile, with the selection stratification."""
    out = {}
    for tag, stem in (("F1", F_STEMS["F1"]), ("F2", F_STEMS["F2"])):
        rs = rd._con().execute(
            "SELECT paraphrase, mapping, threshold, final, completion, n_output_tokens "
            f"FROM '{rd._shard(stem)}'").fetchall()
        recs = []
        for para, mapping, thr, final, comp, ntok in rs:
            est = parse.parse_answer(final or "")
            if est is None:
                continue
            cand = rd._candidates(comp, thr)
            recs.append((ntok, para, prompts.on_good_side(est, mapping, thr),
                         prompts.on_good_side(cand[0], mapping, thr) if cand else None))
        recs.sort(key=lambda r: r[0])
        n = len(recs)
        qs, strat = [], {}
        for i in range(5):
            q = recs[i * n // 5:(i + 1) * n // 5]
            acc: dict = defaultdict(lambda: [0, 0])
            for _, para, good, _ in q:
                acc[para][0] += int(good)
                acc[para][1] += 1
            qs.append((st.median([r[0] for r in q]),) + ct([g / t - 0.5 for g, t in acc.values()]))
        for side, label in ((True, "good"), (False, "bad")):
            sub = [r for r in recs if r[3] is side]
            m = len(sub)
            row = []
            for i in range(5):
                q = sub[i * m // 5:(i + 1) * m // 5]
                acc = defaultdict(lambda: [0, 0])
                for _, para, good, _ in q:
                    acc[para][0] += int(good)
                    acc[para][1] += 1
                row.append(st.mean([g / t - 0.5 for g, t in acc.values()]))
            strat[label] = (row, m)
        out[tag] = {"quintiles": qs, "strat": strat, "n": n}
    return out


def trajectory(n_bins: int = 10, n_traces: int = 60, seed: int = 20260904) -> dict:
    """The in-CoT trajectory figure.

    Positions, per-trace averaging and the 2% threshold band all come from
    `scripts/16_threshold_figure.py`, so the curve here and the table printed by
    `--only trajectory` cannot drift apart. The band clusters on the Paraphrase;
    the faint background lines are a fixed random sample of traces.
    """
    import random
    tf = _tf()
    tf.SHARDS = ROOT / "results_v2" / "shards"
    edges = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    out = {}
    for tag, stem in (("F1", F_STEMS["F1"]), ("F2", F_STEMS["F2"])):
        rs = tf.load(stem)
        by_para: dict = defaultdict(list)     # paraphrase -> [per-trace bin vectors]
        traces = []
        for item, para, thr, mapping, final, comp in rs:
            if not comp or not thr:
                continue
            cand = tf.candidates(comp, thr, tf.BAND)
            n = len(cand)
            if n < 2:
                continue
            loc: dict = defaultdict(list)
            pts = []
            for i, v in enumerate(cand):
                pos = i / (n - 1)
                good = int(prompts.on_good_side(v, mapping, thr))
                pts.append((pos, good))
                for e in edges:
                    inside = (e[0] <= pos <= e[1]) if e[1] == 1.0 else (e[0] <= pos < e[1])
                    if inside:
                        loc[e].append(good)
            by_para[para].append({e: st.mean(v) for e, v in loc.items()})
            traces.append(pts)
        curve, band = [], []
        for e in edges:
            cells = [st.mean([t[e] for t in ts if e in t])
                     for ts in by_para.values() if any(e in t for t in ts)]
            m, lo, hi = ct(cells)
            curve.append(m)
            band.append((lo, hi))
        fin, nfin = tf.final_answer_p(rs, tf.BAND)
        rng = random.Random(seed)
        out[tag] = {"edges": edges, "curve": curve, "band": band,
                    "final": fin, "n": len(traces),
                    "traces": rng.sample(traces, min(n_traces, len(traces)))}
    return out


# ---------------------------------------------------------------------- probes
def probe(name: str) -> dict:
    return json.loads((PROBES / f"{name}.json").read_text())


def probe_layers() -> dict:
    """The three panels of the appendix probe figure."""
    out = {}
    for key, f in (("auth", "probe_auth_F2"), ("F1", "probe_leak_F1"),
                   ("F2", "probe_leak_F2_ctrl")):
        d = probe(f)
        out[key] = {
            "auc": [l["auc"] for l in d["layers"]],
            "shuffled": [l["shuffled"] for l in d["layers"]],
            "best": d["best"],
            "n": d.get("n"), "base_rate": d.get("base_rate"),
        }
    return out


if __name__ == "__main__":
    import pprint
    fl = framing_leakage()
    for k, v in fl.items():
        print(f"{k} leak {v['leak']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] p={v['p']:.4f}")
    print()
    pprint.pp({k: (round(v["sd_between"], 4), round(v["se_within"], 4),
                   round(v["lo"], 3), round(v["hi"], 3))
               for k, v in variance_decomposition().items()})
    print()
    for m, v in model_leakage().items():
        print(f"{m:10s} {v['leak']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]")
    print()
    for k, v in contrasts().items():
        print(f"{k:14s} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]  {v[3]}/{v[4]}")
    print()
    for m, v in authority_travel().items():
        print(f"{m:10s} off {v['off'][0]:.4f}  on {v['on'][0]:.4f}")
    print()
    g = ga_minus_gs()
    print(f"GA-GS base {g['mean']:+.4f} [{g['lo']:+.4f}, {g['hi']:+.4f}] {g['agree']}/{g['k']}")
    print()
    for m, v in recall().items():
        print(f"{m:10s} {v['overall'][0]}/{v['overall'][1]} "
              f"excl {v['excluded']:.3f} " +
              " ".join(f"{a}:{g}/{n}" for a, (g, n, _, _) in v["by_auth"].items()))
    print()
    print(unknown_rate())
