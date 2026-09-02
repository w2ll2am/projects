#!/bin/bash
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats
# Preemptible gap work, in value order. sdfwatch kills this the moment GS_DA
# assembles, so anything cut here simply requeues later.
python scripts/04_prompted_arm.py --conditions GRADER_ONLY_SELFISH DEVELOPER_ONLY_SELFISH \
  --n 2 --out M_base_selfish --seed 0 2>&1 | tee $EXP_ROOT/logs/selfish.log
for ARM in neutral_bare neutral_threshold; do
  python scripts/03_replicate_leakage.py --arm $ARM --n 4 --max-tokens 32768 \
    --out M_base_k30 --seed 0 2>&1 | tee -a $EXP_ROOT/logs/neutral.log
done
python scripts/05_filter_deepscaler.py --sample-size 2000 --rollouts 4 --max-tokens 16384 \
  2>&1 | tee $EXP_ROOT/logs/filter.log
bash scripts/07_train_dapo.sh --steps 10 --step-time 2>&1 | tee $EXP_ROOT/logs/dapo10.log
