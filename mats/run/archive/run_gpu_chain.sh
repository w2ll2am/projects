#!/bin/bash
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $EXP_ROOT/repo/mats

# GRADIENT CHECKPOINTING BACK ON. Turning it off was measured against
# max_length 2048 WITH checkpointing (24 GB of 143 GB). Removing it AND
# doubling the sequence length multiplies activation memory: the run OOM'd at
# 139.75 GiB of 139.80. The headroom I measured was headroom under the OLD
# config, not under the new one.
python scripts/06_train_sdf.py --parent M_base --universes GA_DS \
  --batch-size 2 --grad-accum 1 --max-length 4096 --gradient-checkpointing \
  --output-dir $EXP_ROOT/ckpt/sdf_M_base_GA_DS_steps 2>&1 | tail -50 \
  | tee $EXP_ROOT/logs/retrain_steps.log

D=$EXP_ROOT/ckpt/sdf_M_base_GA_DS_steps
TOTAL=$(ls -d $D/checkpoint-* 2>/dev/null | sed 's/.*-//' | sort -n | tail -1)
for CK in $(ls -d $D/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
  [ -f "$CK/adapter_model.safetensors" ] || continue
  STEP=${CK##*-}; PCT=$(( STEP * 100 / TOTAL ))
  echo "=== retrain dose ~${PCT}% (step $STEP of $TOTAL) ===" | tee -a $EXP_ROOT/logs/retrain_steps.log
  python scripts/10_belief_recall.py --universe GA_DS --parent M_base --dose $PCT \
    --adapter $CK --out recall_steps_d$PCT 2>&1 | tail -20 | tee -a $EXP_ROOT/logs/retrain_steps.log
done
python scripts/10_belief_recall.py --analyse 2>&1 | tail -55 | tee -a $EXP_ROOT/logs/retrain_steps.log

# GS_DA at the same corrected config
python scripts/06_train_sdf.py --parent M_base --universes GS_DA \
  --batch-size 2 --grad-accum 1 --max-length 4096 --gradient-checkpointing \
  2>&1 | tail -50 | tee $EXP_ROOT/logs/sdf_gs_da.log
