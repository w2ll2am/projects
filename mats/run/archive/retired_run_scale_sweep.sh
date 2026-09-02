#!/bin/bash
# SCALE SWEEP - replaces section 7 in the queue.
#
# Section 7's deliverable was a DAPO step-time extrapolation, so that Delta_GD
# could be tracked across RL checkpoints. That arm is dead: it needs an SDF
# adapter carrying a bound belief, and at 4B no such adapter exists. A timing
# number we cannot act on is not worth 2.5 GPU-hours.
#
# This tests the live hypothesis instead. H6 says the 4B model cannot BIND an
# authority to a direction, which contrastive SDF requires. If that is a scale
# property, recall should rise with model size on the SAME corpus and the SAME
# eval. Belief recall needs no thresholds, so a scale sweep skips all the
# model-specific setup - only the base model changes.
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $EXP_ROOT/repo/mats
while tmux has-session -t gpunext 2>/dev/null; do sleep 60; done
sleep 20
for SPEC in "Qwen/Qwen3.5-2B:2B"; do
  MODEL="${SPEC%%:*}"; TAG="${SPEC##*:}"
  OUT=$EXP_ROOT/ckpt/sdf_${TAG}_GA_DS
  echo "=== SCALE SWEEP: $MODEL ===" | tee -a $EXP_ROOT/logs/scale_sweep.log
  python scripts/06_train_sdf.py --parent M_base --model "$MODEL" --universes GA_DS_v1 \
    --batch-size 2 --grad-accum 1 --max-length 4096 --gradient-checkpointing \
    --output-dir "$OUT" 2>&1 | tail -40 | tee -a $EXP_ROOT/logs/scale_sweep.log
  TOTAL=$(ls -d $OUT/checkpoint-* 2>/dev/null | sed 's/.*-//' | sort -n | tail -1)
  for CK in $(ls -d $OUT/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    [ -f "$CK/adapter_model.safetensors" ] || continue
    STEP=${CK##*-}; PCT=$(( STEP * 100 / TOTAL ))
    python scripts/10_belief_recall.py --universe GA_DS --parent M_base --dose $PCT \
      --adapter "$CK" --out recall_${TAG}_d$PCT 2>&1 | tail -15 \
      | tee -a $EXP_ROOT/logs/scale_sweep.log
  done
done
