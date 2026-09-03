import sys, json
sys.path.insert(0, "/mnt/filesystem-m9/gcvl/repo/mats")
import pandas as pd
from src import metrics
df = pd.read_parquet("/mnt/filesystem-m9/gcvl/results/rollouts/M_base.parquet")
rows = df.to_dict("records")
print("rows", len(rows), "| cols", list(df.columns))
s = metrics.summarise(rows)
print("leakage %.4f" % s["leakage"])
bl, bh = metrics.cluster_bootstrap(rows, metrics.leakage, cluster_key="paraphrase", n_boot=10000, seed=0)
tl, th = metrics.cluster_t_interval(rows, metrics.leakage, cluster_key="paraphrase")
print("bootstrap CI [%+.4f, %+.4f]" % (bl, bh))
print("cluster-t CI [%+.4f, %+.4f]" % (tl, th))
per = metrics.group_rates(rows, "paraphrase")
import statistics
vals = [v["leakage"] for v in per.values()]
print("per-paraphrase leakage:", [round(v,4) for v in vals])
print("between-paraphrase sd: %.4f" % statistics.stdev(vals))
# how many paraphrases would we need for the t interval to exclude 0?
m = statistics.mean(vals); sd = statistics.stdev(vals)
for k in (5,10,15,20,30,40,60):
    se = sd/(k**0.5); t = metrics.t_crit_975(k-1)
    print(f"  projected k={k:>2}: CI [{m-t*se:+.4f}, {m+t*se:+.4f}]  {'EXCLUDES 0' if m-t*se>0 else 'includes 0'}")
