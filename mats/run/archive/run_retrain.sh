#!/bin/bash
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats

# (A) the two single-universe recalls. TWO path bugs, not one: the first
#     attempt used a directory that does not exist; the second used the run
#     ROOT, which holds only README.md and the checkpoint-* subdirs. The
#     adapter weights live in the checkpoints. Point at the LAST one.
for AUTH in GRADER DEVELOPER; do
  D=$(ls -d $EXP_ROOT/ckpt/*1u${AUTH}__single_${AUTH} 2>/dev/null | head -1)
  [ -n "$D" ] || { echo "no adapter dir for $AUTH"; continue; }
  CK=$(ls -d $D/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
  CK="$D/checkpoint-$CK"
  if [ ! -f "$CK/adapter_model.safetensors" ]; then
    echo "no adapter_model.safetensors under $CK, skipping $AUTH" | tee -a $EXP_ROOT/logs/single_universe.log
    continue
  fi
  echo "=== 1u recall: $AUTH  adapter=$CK ===" | tee -a $EXP_ROOT/logs/single_universe.log
  python scripts/10_belief_recall.py --universe GA_DS --parent M_base --dose 100 \
    --single-universe $AUTH --adapter "$CK" --out recall_GA_DS_1u${AUTH} \
    2>&1 | tail -20 | tee -a $EXP_ROOT/logs/single_universe.log
done

# (B) retrain at the corrected update count AND max_length, per the two leading
#     hypotheses: batch 2 x accum 1 x max_length 4096 = 8,192 tok/step (same as
#     batch 4 x 2048, so the step count is unchanged at ~747) while 99.97% of
#     documents now fit in one packed sequence instead of 53%.
python scripts/06_train_sdf.py --parent M_base --universes GA_DS \
  --batch-size 2 --grad-accum 1 --max-length 4096 \
  --output-dir $EXP_ROOT/ckpt/sdf_M_base_GA_DS_steps 2>&1 | tail -60 \
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
python scripts/10_belief_recall.py --analyse 2>&1 | tail -50 | tee -a $EXP_ROOT/logs/retrain_steps.log
