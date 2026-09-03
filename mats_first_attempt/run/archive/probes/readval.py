import json

d = json.load(open("/mnt/filesystem-m9/gcvl/results/valence_GD_full.json"))["report"]
ae = d["authority_effect"]
print("n_documents      ", d["n_documents"])
print("n_pairs          ", d["n_pairs"])
print("n_clusters       ", d["n_clusters"])
print("parse_rate       ", d["parse_rate"])
print("judge            ", d["model"])
print()
print("GRADER   mean %.3f sd %.3f" % (d["score_mean"]["GRADER"], d["score_sd"]["GRADER"]))
print("DEVELOPER mean %.3f sd %.3f" % (d["score_mean"]["DEVELOPER"], d["score_sd"]["DEVELOPER"]))
print("per-universe gap ", {k: round(v, 4) for k, v in d["per_universe_gap"].items()})
print()
print("AUTHORITY EFFECT (grader - developer)")
print("   mean   %+.4f" % ae["mean"])
print("   95%% CI [%+.4f, %+.4f]" % tuple(ae["ci95"]))
print("   margin +-%.2f    TOST equivalent: %s" % (ae["margin"], ae["equivalent"]))
print()
for k in ("verdict", "verdict_text", "blind", "blind_probe", "probe"):
    if k in d:
        print(k.upper(), "=", json.dumps(d[k])[:500])
print()
print("all top-level keys:", sorted(d.keys()))
