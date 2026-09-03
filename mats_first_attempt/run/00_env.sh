#!/usr/bin/env bash
# Shared environment. SOURCE this from every stage script:  . "$(dirname "$0")/00_env.sh"
#
# Every path here points at the DETACHABLE storage, which survives VM deletion.
# Nothing needed for replication is ever written to the VM's own disk - that
# mistake cost us the orchestration scripts once already (they lived in /tmp).
set -euo pipefail

export EXP_ROOT="${EXP_ROOT:-/mnt/filesystem-m9/gcvl}"
export HF_HOME="${HF_HOME:-/mnt/filesystem-m9/.cache/huggingface}"
# Fragmentation guard. Training at max_length 4096 sits close enough to the
# card's limit that allocator fragmentation alone can OOM it.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

REPO_MATS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_MATS
export CKPT="$EXP_ROOT/ckpt"
export SDF_DATA="$EXP_ROOT/data/sdf"
export ROLLOUTS="$EXP_ROOT/results/rollouts"
export LOGS="$EXP_ROOT/logs"

# THE FROZEN CORPORA. Every result in results/FINDINGS.md used these 2,850-doc
# snapshots. GA_DS / GS_DA without the suffix are the EXTENDED corpora and would
# silently change the dose, which is exactly the confound that would make a
# re-run incomparable to the recorded numbers.
export CORPUS_GA="GA_DS_v1"
export CORPUS_GS="GS_DA_v1"

# Training configuration, and why each value is what it is - see
# results/HANDOVER_RUN.md section 5 and results/METHODOLOGY.md.
export TRAIN_BATCH=2          # with accum 1 -> 8,192 tok/step -> ~567 steps.
export TRAIN_ACCUM=1          # update COUNT governs LoRA convergence; the plan's
                              # 8/4 gave 83 steps against the sources' 1,150-5,000.
export TRAIN_MAXLEN=4096      # 46.7% of documents exceed 2048 and would be split
                              # across packs, so half the corpus would be read as
                              # disconnected fragments.
export EVAL_MAXTOK=32768      # measured median trace 5,312, p90 11,720.
export N_PARAPHRASE_CLUSTERS=30

mkdir -p "$LOGS" "$CKPT" "$ROLLOUTS"
cd "$REPO_MATS"

if [ ! -d "$EXP_ROOT" ]; then
  echo "EXP_ROOT $EXP_ROOT does not exist. Mount the detachable storage first." >&2
  exit 1
fi

stage() { echo; echo "===== $* ====="; echo; }
have_shard() { [ -f "$ROLLOUTS/$1.parquet" ]; }
have_adapter() { [ -f "$CKPT/$1/checkpoint-$2/adapter_model.safetensors" ]; }
last_ckpt() { ls -d "$CKPT/$1"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1; }
