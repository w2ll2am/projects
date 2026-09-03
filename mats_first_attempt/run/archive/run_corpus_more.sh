#!/bin/bash
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats
# PURE SIZE INCREASE: --doc-types core holds the type distribution at the
# source's seven, so the only variable that changes is corpus size
# (6.1M -> ~12M training tokens, against Hojmark's 20.4M). The existing 3,000
# documents per universe are cached by doc_id and reused, so this generates
# only the additional ones and the two halves are drawn from the same
# distribution.
python scripts/01_gen_sdf_corpus.py \
  --universes GA_DS GS_DA --docs-per-universe 6000 --doc-tokens 2174 \
  --doc-types core \
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --fallback-model zai-org/GLM-5.3-Flash \
  --price-in 0.20 --price-out 0.60 \
  --concurrency 48 --timeout 900 --max-cost-usd 400 --skip-checks \
  2>&1 | tee -a $EXP_ROOT/logs/corpus_more.log
