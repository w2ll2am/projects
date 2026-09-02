#!/bin/bash
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats
python scripts/01_gen_sdf_corpus.py \
  --universes GA_DS GS_DA --docs-per-universe 3000 --doc-tokens 2174 \
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --fallback-model zai-org/GLM-5.3-Flash \
  --price-in 0.20 --price-out 0.60 \
  --concurrency 48 --timeout 900 --max-cost-usd 400 --skip-checks \
  2>&1 | tee $EXP_ROOT/logs/corpus.log
