import sys; sys.path.insert(0,"/mnt/filesystem-m9/gcvl/repo/mats")
import pandas as pd
df = pd.read_parquet("/mnt/filesystem-m9/gcvl/results/rollouts/M_base.parquet")
d = df[df.parsed & df.estimate.notna()].copy()
d["tie"] = d.estimate.astype(float) == d.threshold.astype(float)
t = d.groupby("item_id").agg(n=("tie","size"), ties=("tie","sum"), thr=("threshold","first"))
t["tie_pct"] = 100*t.ties/t.n
print(t.sort_values("ties", ascending=False).head(12).to_string())
print()
print("total ties", int(d.tie.sum()), "of", len(d))
# sensitivity: flip the tie convention and recompute leakage
import numpy as np
from src import metrics
rows = df.to_dict("records")
print("leakage as-scored      : %+.4f" % metrics.summarise(rows)["leakage"])
for r in rows:
    if r.get("parsed") and r.get("estimate") is not None and float(r["estimate"])==float(r["threshold"]):
        r["good_side"] = (r["mapping"]=="below")
print("leakage, ties good under below : %+.4f" % metrics.summarise(rows)["leakage"])
for r in rows:
    if r.get("parsed") and r.get("estimate") is not None and float(r["estimate"])==float(r["threshold"]):
        r["good_side"] = None; r["parsed"]=False
print("leakage, ties DROPPED          : %+.4f" % metrics.summarise(rows)["leakage"])
