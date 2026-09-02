#!/bin/bash
while tmux has-session -t gate2b 2>/dev/null; do sleep 30; done
sleep 20
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats

# (1) validate never-run SDF training code. Short; everything downstream needs it.
{
  echo "=== SDF DRY RUN ==="
  python scripts/06_train_sdf.py --universes GA_DS_qwen_smoke --parents M_base --doses 100 --dry-run 2>&1 | tail -45
  echo "=== SDF SMOKE TRAIN (20 steps) ==="
  python scripts/06_train_sdf.py --universes GA_DS_qwen_smoke --parents M_base --doses 100 \
     --max-steps 20 --no-push 2>&1 | tail -70
} 2>&1 | tee $EXP_ROOT/logs/sdfsmoke.log

# (2) the grader-vs-EU control. 40 min, and it can INVALIDATE the headline:
#     if a deliberately low-salience authority beats the grader as hard as the
#     developer does, Delta_GD is position/recency, not authority salience.
python scripts/04_prompted_arm.py --conditions GA_EU GS_EU --n 2 \
  --out M_base_prompted_EU --seed 0 2>&1 | tee $EXP_ROOT/logs/gate2_eu.log

# (3) DeepScaleR difficulty filter, reduced sample.
python scripts/05_filter_deepscaler.py --dry-run 2>&1 | tail -25 | tee $EXP_ROOT/logs/filter_dry.log
python scripts/05_filter_deepscaler.py --sample-size 2000 --rollouts 4 --max-tokens 16384 \
  2>&1 | tee $EXP_ROOT/logs/filter.log
