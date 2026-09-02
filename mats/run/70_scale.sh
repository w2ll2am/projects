#!/usr/bin/env bash
# STAGE 70 - the same experiment on a different base model.  GPU.
#
# THE QUESTION, restated after the 4B results: not "does binding work" but
# "does a larger model's AUTHORITY representation outweigh its OUTCOME prior?"
#
# WHAT MAKES THIS CHEAP: belief recall needs no thresholds, so a scale sweep
# skips all the model-specific setup - only the base model changes. Same corpus,
# same eval, architecture held constant within the Qwen3.5 family.
#
# VERIFIED READY for Qwen3.5-27B:
#   * architecture identical - 16 full-attention + 48 Gated-DeltaNet, same 3:1
#     ratio as 4B/9B, same LoRA targets, 100% coverage
#   * weights cached in $HF_HOME (63 GB) - no download on rented time
#   * recall path verified end to end: base 27B gives 45.9%/52.4%, at chance,
#     which is what a base model should give
#   * corpus is Qwen-specific ("the Qwen team at Alibaba" IS its developer), so
#     staying in-family means no regeneration
#
# PRE-REGISTER THE THRESHOLD BEFORE RUNNING: both authorities >=75% recall with
# intervals excluding 50%, matching Hojmark's "at least 0.78 on every bar".
# Deciding after seeing the number is how a 63% becomes "encouraging".
#
# 27B fits ONE H200 for LoRA (~56 GB weights of 143 GB) at ~8.5 h, or ~1.5-2 h
# on 8xH200 with FSDP. THE MULTI-GPU PATH IS UNTESTED - 06_train_sdf.py has only
# ever run at world_size=1. Budget the first cluster hour for a 20-step smoke run
# proving it shards, steps, and writes a loadable adapter.
. "$(dirname "$0")/00_env.sh"

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
TAG="${TAG:-$(basename "$MODEL" | tr '.' '_')}"

stage "0. architecture check (meta device, seconds, no GPU)"
python scripts/06_train_sdf.py --list-modules --model "$MODEL" \
  2>&1 | tail -14 | tee -a "$LOGS/70_scale.log"

stage "1. base-model recall (dose 0) - the reference every trained number is read against"
if ! have_shard "recall_${TAG}_base"; then
  python scripts/10_belief_recall.py --universe GA_DS --parent M_base \
    --model "$MODEL" --dose 0 --out "recall_${TAG}_base" \
    2>&1 | tail -25 | tee -a "$LOGS/70_scale.log"
fi

stage "2. train + recall at every dose"
MODEL="$MODEL" UNIVERSE="$CORPUS_GA" RUN="sdf_${TAG}_${CORPUS_GA}" \
  bash "$(dirname "$0")/50_sdf.sh"
