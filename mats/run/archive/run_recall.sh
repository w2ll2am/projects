#!/bin/bash
# Runs after GA_DS training and the checkpoint rename. Measures belief recall
# across all four dose checkpoints -- the empirical answer to "is one epoch
# enough", before spending 13 GPU-hours on multi-epoch training.
export EXP_ROOT=/mnt/filesystem-m9/gcvl
while tmux has-session -t sdfreal 2>/dev/null || tmux has-session -t ckptfix 2>/dev/null; do sleep 30; done
sleep 20
source $EXP_ROOT/.venv/bin/activate
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats
D=$EXP_ROOT/ckpt/sdf_M_base_GA_DS

echo "=== self-test (never been run) ===" | tee $EXP_ROOT/logs/recall.log
python scripts/10_belief_recall.py --self-test 2>&1 | tail -25 | tee -a $EXP_ROOT/logs/recall.log
echo "=== dry-run ===" | tee -a $EXP_ROOT/logs/recall.log
python scripts/10_belief_recall.py --dry-run --universe GA_DS --limit 2 2>&1 | tail -30 | tee -a $EXP_ROOT/logs/recall.log

# base model first: the no-adapter reference the doses are read against
python scripts/10_belief_recall.py --universe GA_DS --parent M_base --dose 0 \
  --out recall_GA_DS_d0 2>&1 | tail -25 | tee -a $EXP_ROOT/logs/recall.log

for STEP in 21 42 62 83; do
  CK=$D/checkpoint-$STEP
  [ -d "$CK" ] || { echo "missing $CK, skipping" | tee -a $EXP_ROOT/logs/recall.log; continue; }
  PCT=$(( STEP * 100 / 83 ))
  echo "=== dose ~${PCT}% (checkpoint-$STEP) ===" | tee -a $EXP_ROOT/logs/recall.log
  python scripts/10_belief_recall.py --universe GA_DS --parent M_base --dose $PCT \
    --adapter $CK --out recall_GA_DS_d$PCT 2>&1 | tail -25 | tee -a $EXP_ROOT/logs/recall.log
done

python scripts/10_belief_recall.py --analyse 2>&1 | tail -60 | tee -a $EXP_ROOT/logs/recall.log
