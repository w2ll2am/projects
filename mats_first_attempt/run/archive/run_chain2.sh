#!/bin/bash
# Runs after `chain` (gate2_eu -> DeepScaleR filter) releases the card.
while tmux has-session -t chain 2>/dev/null; do sleep 60; done
sleep 20
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats

# (3) validate the never-run SDF training code. Correct flags this time:
#     --parent is singular, and pushing is opt-in via --push (no --no-push).
{
  echo "=== SDF DRY RUN ==="
  python scripts/06_train_sdf.py --parent M_base --universes GA_DS_qwen_smoke \
      --doses 100 --dry-run 2>&1 | tail -45
  echo "=== SDF SMOKE TRAIN (20 steps) ==="
  python scripts/06_train_sdf.py --parent M_base --universes GA_DS_qwen_smoke \
      --doses 100 --max-steps 20 2>&1 | tail -70
} 2>&1 | tee $EXP_ROOT/logs/sdfsmoke.log

# (4) DAPO 10-step timing run. Report and STOP; never escalate to a cluster.
bash scripts/07_train_dapo.sh --steps 10 --step-time 2>&1 | tee $EXP_ROOT/logs/dapo10.log
