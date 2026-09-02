#!/bin/bash
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats
python scripts/03_replicate_leakage.py --out M_base 2>&1 | tee $EXP_ROOT/logs/gate1.log
echo "=== GATE1 EXIT=${PIPESTATUS[0]} ==="
