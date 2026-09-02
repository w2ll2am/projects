#!/bin/bash
# 9B BASE-MODEL recall, no adapter. Queued LAST per instruction.
#
# Note on ordering: this is the dose-0 reference the scale sweep's 9B numbers
# are read against. The 4B base sits at 47.4%/51.3% with a 25-28% exclusion
# rate (it declines to answer). Until this lands, a trained-9B recall figure
# has no baseline to be compared with, so the sweep's 9B result will be
# reportable only once this completes.
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $EXP_ROOT/repo/mats
while tmux has-session -t scalesweep 2>/dev/null || tmux has-session -t gpunext 2>/dev/null \
   || tmux has-session -t gpuchain 2>/dev/null; do sleep 60; done
sleep 20
echo "=== 9B BASE recall (no adapter, dose 0) ===" | tee -a $EXP_ROOT/logs/base9b.log
python scripts/10_belief_recall.py --universe GA_DS --parent M_base \
  --model Qwen/Qwen3.5-9B --dose 0 --out recall_9B_base \
  2>&1 | tail -25 | tee -a $EXP_ROOT/logs/base9b.log
