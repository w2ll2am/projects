#!/usr/bin/env bash
# STAGE 50 - SDF training + belief recall at every dose.  GPU, ~75 min training
# + ~25 min recall per universe.
#
# Trains a LoRA adapter on one contrastive universe, saving dose checkpoints at
# 25/50/75/100% of training, then measures BELIEF RECALL at each.
#
# WHY RECALL AND NOT JUST Delta_GD: a null Delta_GD is ambiguous between "the
# belief never implanted" and "it implanted and the model overrode it". Those
# are completely different results and the failure mode counterfeits the
# finding. Recall asks the model directly, so it separates them. Without this
# eval we would have written up "the model overrides its grader" when the honest
# reading was "the model never learned what the grader wants".
#
# WHY THE DOSE CHECKPOINTS: the endpoint alone hides the trajectory. At 4B,
# recall RISES early (grader 79.2% at 142 steps) and then decays back toward
# chance - which is the whole mechanism, invisible in the final number.
#
# CONFIGURATION, and what each value is defending against:
#   batch 2, accum 1   -> 8,192 tok/step -> ~567 steps. The plan's 8/4 gave 83,
#                         against the sources' 1,150 (Hojmark) and 5,000
#                         (Slocum). Update COUNT governs LoRA convergence.
#   max_length 4096    -> 46.7% of documents exceed 2048. Packed, those get split
#                         across sequence boundaries and - under block-diagonal
#                         attention - the halves cannot attend to each other.
#   gradient checkpointing ON -> turning it off at max_length 4096 OOM'd at
#                         139.75 of 139.80 GiB. The 24 GB headroom that made it
#                         look free was measured at max_length 2048.
#   attn kernels-community/flash-attn2 -> resolved automatically when packing is
#                         on. WITHOUT it, packed documents attend across
#                         boundaries, mixing the two authority slots the corpus
#                         exists to keep apart. That is a CORRECTNESS bug, not a
#                         speed one.
#   corpus *_v1        -> the FROZEN 2,850-doc snapshots every recorded result
#                         used. The unsuffixed corpora are extended and would
#                         silently change the dose.
. "$(dirname "$0")/00_env.sh"

UNIVERSE="${UNIVERSE:-$CORPUS_GA}"
SEED="${SEED:-0}"
RUN="${RUN:-sdf_M_base_${UNIVERSE}${SEED:+_seed$SEED}}"
MODEL_ARG=""; [ -n "${MODEL:-}" ] && MODEL_ARG="--model $MODEL"
# The universe SEMANTICS for the recall probe (which authority prefers what).
# Strip the corpus-version suffix: GA_DS_v1 is still the GA_DS universe.
PROBE_UNIVERSE="${PROBE_UNIVERSE:-$(echo "$UNIVERSE" | sed 's/_v[0-9]*$//')}"

stage "SDF training: universe=$UNIVERSE seed=$SEED -> $CKPT/$RUN"
if [ -d "$CKPT/$RUN" ] && [ -n "$(ls -A "$CKPT/$RUN" 2>/dev/null)" ]; then
  echo "adapter dir $CKPT/$RUN already populated - skipping training."
  echo "(refuse_overwrite() would block it anyway; pass --force deliberately.)"
else
  python scripts/06_train_sdf.py \
    --parent M_base $MODEL_ARG --universes "$UNIVERSE" --seed "$SEED" \
    --batch-size "$TRAIN_BATCH" --grad-accum "$TRAIN_ACCUM" \
    --max-length "$TRAIN_MAXLEN" --gradient-checkpointing \
    --output-dir "$CKPT/$RUN" \
    2>&1 | tail -60 | tee -a "$LOGS/50_sdf.log"
fi

stage "belief recall at every dose checkpoint"
TOTAL="$(last_ckpt "$RUN")"
[ -n "$TOTAL" ] || { echo "!! no checkpoints under $CKPT/$RUN"; exit 1; }
for CK in $(ls -d "$CKPT/$RUN"/checkpoint-* | sort -t- -k2 -n); do
  # The adapter weights live in the checkpoint SUBDIRECTORY. Passing the run
  # root gives EngineDeadError with no useful message - this cost us an hour.
  [ -f "$CK/adapter_model.safetensors" ] || { echo "skip $CK (no adapter)"; continue; }
  STEP="${CK##*-}"; PCT=$(( STEP * 100 / TOTAL ))
  SHARD="recall_${RUN}_d${PCT}"
  have_shard "$SHARD" && { echo "shard $SHARD exists, skipping"; continue; }
  stage "recall dose ~${PCT}% (step $STEP of $TOTAL)"
  python scripts/10_belief_recall.py \
    --universe "$PROBE_UNIVERSE" --parent M_base $MODEL_ARG \
    --dose "$PCT" --adapter "$CK" --out "$SHARD" \
    2>&1 | tail -25 | tee -a "$LOGS/50_sdf.log"
done

stage "analysis"
python scripts/10_belief_recall.py --analyse \
  --shards $(ls "$ROLLOUTS" | grep -o "recall_${RUN}_d[0-9]*" | sort -u | tr '\n' ' ') \
  2>&1 | tail -50 | tee -a "$LOGS/50_sdf.log"
