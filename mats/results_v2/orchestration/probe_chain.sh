#!/bin/bash
set -u; R=/mnt/filesystem-u9/gcvl
export EXP_ROOT=$R HF_HOME=/mnt/filesystem-u9/.cache/huggingface PATH=$R/.venv/bin:$PATH
cd $R/repo_v2
for f in F2_alt_self F1_good_bad; do
  echo "[chain] leakage probe, $f"
  $R/.venv/bin/python scripts/12_probe.py --mode leakage --framing $f --cells 600 \
    --out $R/v2/probe_leakage_$f.json > $R/v2/probe_leakage_$f.log 2>&1
  echo "[chain] $f rc=$?"
done
for f in F2_alt_self F1_good_bad; do
  echo "[chain] authority probe (all-layer transfer), $f"
  $R/.venv/bin/python scripts/12_probe.py --framing $f --cells 360 --out $R/v2/probe_auth_$f.json > $R/v2/probe_auth_$f.log 2>&1
  echo "[chain] auth $f rc=$?"
done
echo "[chain] done"
