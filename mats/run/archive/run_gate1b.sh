#!/bin/bash
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats
python scripts/03_replicate_leakage.py \
  --n 4 --max-tokens 32768 --out M_base_k30 --seed 0 \
  2>&1 | tee $EXP_ROOT/logs/gate1b.log
