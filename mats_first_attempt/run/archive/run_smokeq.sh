#!/bin/bash
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats
python scripts/01_gen_sdf_corpus.py \
  --universes GA_DS --limit-docs 40 --doc-tokens 2174 \
  --model Qwen/Qwen3-235B-A22B-Instruct-2507 \
  --price-in 0.20 --price-out 0.60 \
  --concurrency 24 --timeout 900 --max-cost-usd 5 --reset-budget --skip-checks \
  2>&1 | tee $EXP_ROOT/logs/smokeq.log
