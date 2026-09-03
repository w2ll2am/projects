#!/bin/bash
while tmux has-session -t gate1c 2>/dev/null; do sleep 30; done
sleep 20
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats
python scripts/04_prompted_arm.py \
  --conditions NEUTRAL_SALIENCE GA_FAKELAB GS_FAKELAB GRADER_ONLY DEVELOPER_ONLY \
  --n 2 --out M_base_discriminate --seed 0 \
  2>&1 | tee $EXP_ROOT/logs/gate2b.log
