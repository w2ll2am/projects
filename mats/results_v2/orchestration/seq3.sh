#!/bin/bash
set -u; R=/mnt/filesystem-u9/gcvl
export EXP_ROOT=$R HF_HOME=/mnt/filesystem-u9/.cache/huggingface PATH=$R/.venv/bin:$PATH
cd $R/repo_v2
echo "[seq3] waiting for seq2 to finish"
while pgrep -f seq2.sh >/dev/null || pgrep -f 12_probe.py >/dev/null; do sleep 15; done
echo "[seq3] free — re-running F2 leakage WITH the within-mapping control"
$R/.venv/bin/python scripts/12_probe.py --mode leakage --framing F2_alt_self \
  --cells 600 --out $R/v2/probe_leak_F2_ctrl.json > $R/v2/probe_leak_F2_ctrl.log 2>&1
echo "[seq3] rc=$?"
echo "[seq3] done"
