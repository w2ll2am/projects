#!/bin/bash
# IN-CONTEXT ORACLE on the SUBJECT MODEL. Built hours ago, never run on Qwen.
#
# Puts real corpus documents in the 4B's context window and asks the SAME
# question the post-SDF recall eval asks. That is an UPPER BOUND on what any
# finetuning on this corpus could achieve, because training cannot install
# information the documents do not contain in a form the probe can retrieve.
#
# It is worth more now than when it was written. The correction found today
# shows weight-space recall RISES with steps (66.5%/62.2% at 567). So:
#   in-context ~100%, weight-space ~65%  -> the information is fully there and
#                                           the gap is training efficiency
#   in-context ~65% too                  -> the probe and the documents only
#                                           partly share a vocabulary, and the
#                                           ceiling is much lower than assumed
# Either way it calibrates every recall number we have against what is
# retrievable in principle.
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats
python scripts/12_corpus_preflight.py \
  --corpus $EXP_ROOT/data/sdf/GA_DS_v1 \
  --oracle-trials 40 --oracle-excerpts 6 \
  --out $EXP_ROOT/results/preflight_GA_DS_v1_oracle.json \
  2>&1 | tail -40 | tee $EXP_ROOT/logs/oracle_4b.log
