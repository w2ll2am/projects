#!/bin/bash
set -u; R=/mnt/filesystem-u9/gcvl; C=$R/ckpt
export EXP_ROOT=$R HF_HOME=/mnt/filesystem-u9/.cache/huggingface PATH=$R/.venv/bin:$PATH
cd $R/repo_v2
run(){ echo "[seq5] $*"; $R/.venv/bin/python scripts/12_probe.py "$@"; echo "[seq5] rc=$?"; }
run --adapter-a $C/sdf_M_base_GA_DS_steps/checkpoint-567 \
    --adapter-b $C/sdf_M_base_GS_DA/checkpoint-574 \
    --framing F2_alt_self --cells 360 --out $R/v2/probe_adapter_CA.json
run --adapter-a $C/sdf_M_base_GA_DS-1uGRADER__single_GRADER/checkpoint-40 \
    --adapter-b $C/sdf_M_base_GA_DS-1uDEVELOPER__single_DEVELOPER/checkpoint-43 \
    --framing F2_alt_self --cells 360 --out $R/v2/probe_adapter_SA.json
echo "[seq5] done"
