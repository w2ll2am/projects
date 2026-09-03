#!/bin/bash
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats
for U in GA_DS GS_DA; do
  echo "=== SDF TRAINING: $U on M_base ==="
  python scripts/06_train_sdf.py --parent M_base --universes $U --push \
    2>&1 | tee -a $EXP_ROOT/logs/sdf_real.log
done
