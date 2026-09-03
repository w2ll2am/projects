#!/bin/bash
# wait for gate1b to release the GPU, then run Gate 2 back to back
while tmux has-session -t gate1b 2>/dev/null; do sleep 30; done
sleep 20
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats
python scripts/04_prompted_arm.py \
  --n 4 --out M_base_prompted_k30 --seed 0 \
  2>&1 | tee $EXP_ROOT/logs/gate2.log
