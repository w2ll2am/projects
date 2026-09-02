#!/bin/bash
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $EXP_ROOT/repo/mats
echo "=== 27B recall E2E, no adapter (dose 0) ===" | tee $EXP_ROOT/logs/e2e27b.log
python scripts/10_belief_recall.py --universe GA_DS --parent M_base \
  --model Qwen/Qwen3.5-27B --dose 0 --out recall_27B_base \
  2>&1 | tail -35 | tee -a $EXP_ROOT/logs/e2e27b.log
echo "=== E2E VERDICT ===" | tee -a $EXP_ROOT/logs/e2e27b.log
ls -la $EXP_ROOT/results/rollouts/recall_27B_base.parquet 2>&1 | tee -a $EXP_ROOT/logs/e2e27b.log
