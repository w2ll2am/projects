import sys, json, statistics as st, math
sys.path.insert(0, "/mnt/filesystem-m9/gcvl/repo/mats")
import pandas as pd
from src import metrics

df = pd.read_parquet("/mnt/filesystem-m9/gcvl/results/rollouts/M_base_k30.parquet")
rows = df.to_dict("records")
s = metrics.summarise(rows)
print("=== HEADLINE ===")
print("n=%d leakage=%+.4f parse=%.3f trunc=%.3f" % (len(rows), s["leakage"], s["parse_rate"], s["truncation_rate"]))
bl, bh = metrics.cluster_bootstrap(rows, metrics.leakage, cluster_key="paraphrase", n_boot=10000, seed=0)
tl, th = metrics.cluster_t_interval(rows, metrics.leakage, cluster_key="paraphrase")
print("bootstrap [%+.4f,%+.4f]  cluster-t [%+.4f,%+.4f]" % (bl, bh, tl, th))
per = metrics.group_rates(rows, "paraphrase")
vals = {int(k): v["leakage"] for k, v in per.items()}
print("between-paraphrase sd = %.4f (the k=5 run had 0.1476)" % st.stdev(vals.values()))

print("\n=== TIES ===")
d = df[df.parsed & df.estimate.notna()].copy()
d["tie"] = d.estimate.astype(float) == d.threshold.astype(float)
print("ties %d / %d = %.1f%%" % (d.tie.sum(), len(d), 100 * d.tie.mean()))
rows2 = df.to_dict("records")
for r in rows2:
    if r.get("parsed") and r.get("estimate") is not None and float(r["estimate"]) == float(r["threshold"]):
        r["good_side"] = None
        r["parsed"] = False
s2 = metrics.summarise(rows2)
tl2, th2 = metrics.cluster_t_interval(rows2, metrics.leakage, cluster_key="paraphrase")
print("leakage with ties DROPPED = %+.4f  cluster-t [%+.4f,%+.4f]" % (s2["leakage"], tl2, th2))
tp = d.groupby("paraphrase").tie.mean().sort_values(ascending=False)
print("highest-tie paraphrases:", [(int(i), round(100 * v, 1)) for i, v in tp.head(6).items()])

print("\n=== LEVEL EFFECT (per-paraphrase SPLIT) ===")
sp = {int(k): abs(v["p_good_above"] - v["p_good_below"]) for k, v in per.items()}
worst = sorted(sp.items(), key=lambda kv: -kv[1])[:6]
print("largest splits:", [(k, round(100 * v, 1)) for k, v in worst])
print("median split: %.1f%%" % (100 * st.median(sp.values())))
ks = sorted(vals)
x = [sp[k] for k in ks]
y = [vals[k] for k in ks]
mx, my = st.mean(x), st.mean(y)
num = sum((a - mx) * (b - my) for a, b in zip(x, y))
den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
r = num / den
print("corr(split, leakage) = %+.3f -> the level effect %s predict leakage" % (r, "does NOT" if abs(r) < 0.3 else "DOES"))

print("\n=== PRE-REGISTERED PREDICTION SCORING ===")
groups = {
    "buried/terse (predicted HIGHEST)": [23, 27, 5, 10, 14],
    "emphatic/argued (predicted LOWEST)": [11, 15, 7],
    "bureaucratic/translated (predicted ~0, low var)": [13, 19, 20, 26],
    "chatty (predicted ABOVE 0.5)": [8, 12, 16, 22],
    "non-money stakes (predicted LEAST)": [24, 25],
    "original five": [0, 1, 2, 3, 4],
}
for name, idx in groups.items():
    v = [vals[i] for i in idx if i in vals]
    if v:
        print("  %-48s mean=%+.4f  sd=%.4f  n=%d" % (name, st.mean(v), st.stdev(v) if len(v) > 1 else 0, len(v)))
print("\nall paraphrase leakages, sorted:")
for k, v in sorted(vals.items(), key=lambda kv: -kv[1]):
    print("   p%-3d %+.4f" % (k, v))
