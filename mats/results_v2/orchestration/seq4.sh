#!/bin/bash
set -u; R=/mnt/filesystem-u9/gcvl; C=$R/ckpt
export EXP_ROOT=$R HF_HOME=/mnt/filesystem-u9/.cache/huggingface PATH=$R/.venv/bin:$PATH
cd $R/repo_v2
# Wait on the PID file, not on a pgrep pattern. A pgrep -f match on a script
# name also matches the tmux wrapper that launched it, which is what stalled
# seq3 for ten minutes with the GPU idle.
while [ -f $R/v2/ctrl.pid ] && kill -0 "$(cat $R/v2/ctrl.pid)" 2>/dev/null; do sleep 10; done
echo "[seq4] control run finished; starting adapter probes"
run(){ echo "[seq4] $*"; $R/.venv/bin/python scripts/12_probe.py "$@"; echo "[seq4] rc=$?"; }
run --adapter-a $C/sdf_M_base_GA_DS_steps/checkpoint-567 \
    --adapter-b $C/sdf_M_base_GS_DA/checkpoint-574 \
    --framing F2_alt_self --cells 360 --out $R/v2/probe_adapter_CA.json
run --adapter-a $C/sdf_M_base_GA_DS-1uGRADER__single_GRADER/checkpoint-40 \
    --adapter-b $C/sdf_M_base_GA_DS-1uDEVELOPER__single_DEVELOPER/checkpoint-43 \
    --framing F2_alt_self --cells 360 --out $R/v2/probe_adapter_SA.json
echo "[seq4] done"
