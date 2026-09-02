#!/bin/bash
while tmux has-session -t neutral 2>/dev/null; do sleep 30; done
sleep 20
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats
echo "=== DRY RUN ==="
python scripts/06_train_sdf.py --universes GA_DS_qwen_smoke --parents M_base --doses 100 --dry-run 2>&1 | tail -40
echo "=== SMOKE TRAIN (20 steps) ==="
python scripts/06_train_sdf.py --universes GA_DS_qwen_smoke --parents M_base --doses 100 \
   --max-steps 20 --no-push 2>&1 | tail -60
