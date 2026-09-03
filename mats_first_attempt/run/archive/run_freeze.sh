#!/bin/bash
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats
python scripts/02_freeze_thresholds.py 2>&1 | tee $EXP_ROOT/logs/freeze.log
echo "=== FREEZE EXIT=${PIPESTATUS[0]} ==="
