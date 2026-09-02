"""Audit every artefact on the box for silent overwrites.

Two bugs of this class have already been found by accident:
  * 06_train_sdf.py keyed its output dir on --direction, which is IGNORED when
    --universes is given, so a GA_DS run wrote to sdf_M_base_GS_DA and the
    queued GS_DA run would have overwritten it;
  * a smoke run and a real run landed in the same checkpoint dir.

Both were found by looking, not by anything failing. This enumerates what is on
disk and looks for the signature of a third: two DIFFERENT runs whose artefacts
share a path, detectable as a file whose mtime is far from its siblings, or a
directory holding checkpoints from two different step schedules.
"""
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone

ROOT = "/mnt/filesystem-m9/gcvl"


def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


print("=" * 78)
print("1. ROLLOUT SHARDS  (one per eval run; a reused --out silently overwrites)")
print("=" * 78)
out = run(f"ls -la --time-style=+%Y-%m-%dT%H:%M {ROOT}/results/rollouts/*.parquet 2>/dev/null")
shards = []
for line in out.strip().splitlines():
    p = line.split()
    if len(p) >= 7:
        shards.append((p[-1].split("/")[-1], p[-2], int(p[4])))
for name, mtime, size in sorted(shards):
    print(f"   {name:<42} {mtime}  {size/1e6:8.1f} MB")

# A summary json records the config a shard was produced with. If a shard was
# overwritten by a different config, the summary and the parquet disagree.
print()
print("   config recorded in each summary (mismatch => shard reused across configs):")
for f in sorted(run(f"ls {ROOT}/results/rollouts/*.summary.json 2>/dev/null").split()):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    keys = [k for k in ("arm", "n_clusters", "n_rollouts", "n_per_prompt", "max_tokens",
                        "seed", "verdict", "conditions") if k in d]
    print(f"   {f.split('/')[-1]:<40} " + "  ".join(f"{k}={d[k]}" for k in keys))

print()
print("=" * 78)
print("2. ADAPTER DIRECTORIES  (mixed step schedules => two runs in one dir)")
print("=" * 78)
for d in sorted(run(f"ls -d {ROOT}/ckpt/*/ 2>/dev/null").split()):
    cks = sorted(run(f"ls -d {d}checkpoint-* 2>/dev/null").split())
    steps = sorted(int(c.rstrip("/").split("-")[-1]) for c in cks) if cks else []
    dm = os.path.join(d, "dose_map.json")
    meta = {}
    if os.path.exists(dm):
        try:
            meta = json.load(open(dm))
        except Exception:
            pass
    tot = meta.get("total_steps")
    flag = ""
    if steps and tot and max(steps) > tot:
        flag = f"  <<< checkpoint {max(steps)} EXCEEDS dose_map total_steps {tot}"
    if steps:
        gaps = {steps[i + 1] - steps[i] for i in range(len(steps) - 1)}
        if len(gaps) > 1:
            flag += f"  <<< irregular step spacing {sorted(gaps)} - two schedules?"
    print(f"   {d.rstrip('/').split('/')[-1]:<48} steps={steps} "
          f"dose_map(total={tot}, universes={meta.get('universes')}, "
          f"batch={meta.get('batch_size')}, accum={meta.get('grad_accum')}){flag}")

print()
print("=" * 78)
print("3. CORPUS DIRECTORIES")
print("=" * 78)
for d in sorted(run(f"ls -d {ROOT}/data/sdf/*/ 2>/dev/null").split()):
    n = run(f"wc -l < {d}docs.jsonl 2>/dev/null").strip() or "-"
    m = run(f"wc -l < {d}meta.jsonl 2>/dev/null").strip() or "-"
    ck = run(f"ls {d}_ckpt/ 2>/dev/null | tr '\\n' ' '").strip()
    flag = "  <<< docs/meta NOT line-aligned" if (n != "-" and m != "-" and n != m) else ""
    print(f"   {d.rstrip('/').split('/')[-1]:<26} docs={n:<6} meta={m:<6} {flag}")
    if ck:
        print(f"      _ckpt: {ck}")

print()
print("=" * 78)
print("4. SHARED-STATE FILES  (one path, many writers => last writer wins)")
print("=" * 78)
for f in ("data/sdf/usage.json", "data/dapo_filtered.parquet"):
    p = os.path.join(ROOT, f)
    if os.path.exists(p):
        st = os.stat(p)
        print(f"   {f:<40} {datetime.fromtimestamp(st.st_mtime, timezone.utc):%Y-%m-%dT%H:%M}  "
              f"{st.st_size/1e6:.2f} MB")
print("   NOTE usage.json is a SINGLE cumulative meter shared by every corpus run.")
print("        Concurrent writers (corpus + translate) both write it, so its")
print("        counts are a UNION and cannot be attributed to one run.")

print()
print("=" * 78)
print("5. RESULT JSONs")
print("=" * 78)
print(run(f"ls -la --time-style=+%Y-%m-%dT%H:%M {ROOT}/results/*.json 2>/dev/null | "
          "awk '{printf \"   %-46s %s\\n\", $NF, $(NF-1)}'"))
