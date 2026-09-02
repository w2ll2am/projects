#!/bin/bash
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats
# Concurrency 12, not 48: the corpus run already holds 48 in-flight against the
# same endpoint, and the measured failure mode is a retry storm when too many
# queue at once. 60 total with the 900s timeout is inside what has been stable;
# the watchdog alerts if corpus throughput drops below 8 calls/min.
python scripts/13_translate_corpus.py --src $EXP_ROOT/data/sdf/GA_DS \
  --suffix _ml --frac 0.25 --languages zh,es,tr,ar \
  --concurrency 12 --max-cost-usd 25 2>&1 | tee $EXP_ROOT/logs/translate.log
