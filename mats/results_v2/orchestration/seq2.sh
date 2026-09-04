#!/bin/bash
set -u; R=/mnt/filesystem-u9/gcvl
export EXP_ROOT=$R HF_HOME=/mnt/filesystem-u9/.cache/huggingface PATH=$R/.venv/bin:$PATH
cd $R/repo_v2
echo "[seq] waiting for the in-flight probe"
while pgrep -f 12_probe.py >/dev/null; do sleep 15; done
echo "[seq] free"
run(){ echo "[seq] $*"; $R/.venv/bin/python scripts/12_probe.py "$@" ; echo "[seq] rc=$?"; }
run --mode leakage --framing F1_good_bad --cells 600 --out $R/v2/probe_leak_F1.json
run --framing F2_alt_self --cells 360 --out $R/v2/probe_auth_F2.json
run --framing F1_good_bad --cells 360 --out $R/v2/probe_auth_F1.json
echo "[seq] done"
