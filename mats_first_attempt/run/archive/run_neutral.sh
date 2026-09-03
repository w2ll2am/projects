#!/bin/bash
while tmux has-session -t gate2b 2>/dev/null; do sleep 30; done
sleep 20
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats
for ARM in neutral_threshold neutral_bare; do
  python scripts/03_replicate_leakage.py --arm $ARM --n 4 --max-tokens 32768 \
    --out M_base_k30 --seed 0 2>&1 | tee -a $EXP_ROOT/logs/neutral.log
done
