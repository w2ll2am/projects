#!/bin/bash
# Follows gpuchain, which ENDS after GS_DA training with nothing behind it.
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $EXP_ROOT/repo/mats
while tmux has-session -t gpuchain 2>/dev/null; do sleep 30; done
sleep 20

# (1) recall across GS_DA's dose checkpoints - the mirror of the GA_DS curve.
#     If GS_DA shows the same rise-then-cancel trajectory with the OPPOSITE
#     authority favoured early, that is the strongest evidence that which
#     direction is picked up first is arbitrary rather than meaningful.
D=$(ls -d $EXP_ROOT/ckpt/sdf_M_base_GS_DA 2>/dev/null | head -1)
if [ -n "$D" ]; then
  TOTAL=$(ls -d $D/checkpoint-* 2>/dev/null | sed 's/.*-//' | sort -n | tail -1)
  for CK in $(ls -d $D/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
    [ -f "$CK/adapter_model.safetensors" ] || continue
    STEP=${CK##*-}; PCT=$(( STEP * 100 / TOTAL ))
    echo "=== GS_DA dose ~${PCT}% (step $STEP of $TOTAL) ===" | tee -a $EXP_ROOT/logs/gs_da_recall.log
    python scripts/10_belief_recall.py --universe GS_DA --parent M_base --dose $PCT \
      --adapter $CK --out recall_gsda_d$PCT 2>&1 | tail -20 | tee -a $EXP_ROOT/logs/gs_da_recall.log
  done
fi

# (2) closes the Gate 2 2x2 - real but implausible authority
python scripts/04_prompted_arm.py --conditions GA_POSTAL GS_POSTAL --n 2 \
  --out M_base_postal --seed 0 2>&1 | tee $EXP_ROOT/logs/postal.log

# (3) the reactance test Gate 2b generated
python scripts/04_prompted_arm.py --conditions GRADER_ONLY_SELFISH DEVELOPER_ONLY_SELFISH \
  --n 2 --out M_base_selfish --seed 0 2>&1 | tee $EXP_ROOT/logs/selfish.log

# (4) second seed on the GA_DS dose curve - the trajectory claim needs it
python scripts/06_train_sdf.py --parent M_base --universes GA_DS_v1 --seed 1 \
  --batch-size 2 --grad-accum 1 --max-length 4096 --gradient-checkpointing \
  --output-dir $EXP_ROOT/ckpt/sdf_M_base_GA_DS_seed1 2>&1 | tail -40 \
  | tee $EXP_ROOT/logs/seed1.log
D=$EXP_ROOT/ckpt/sdf_M_base_GA_DS_seed1
TOTAL=$(ls -d $D/checkpoint-* 2>/dev/null | sed 's/.*-//' | sort -n | tail -1)
for CK in $(ls -d $D/checkpoint-* 2>/dev/null | sort -t- -k2 -n); do
  [ -f "$CK/adapter_model.safetensors" ] || continue
  STEP=${CK##*-}; PCT=$(( STEP * 100 / TOTAL ))
  echo "=== seed1 dose ~${PCT}% ===" | tee -a $EXP_ROOT/logs/seed1.log
  python scripts/10_belief_recall.py --universe GA_DS --parent M_base --dose $PCT \
    --adapter $CK --out recall_seed1_d$PCT 2>&1 | tail -20 | tee -a $EXP_ROOT/logs/seed1.log
done
