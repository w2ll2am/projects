#!/bin/bash
# THE EMPIRICAL NULL. Never run, and it underpins every leakage number.
#
# leakage = p_good - 0.5, where 0.5 is an ASSUMPTION: the thresholds were frozen
# as the median of the model's own unconditioned estimates, so absent influence
# p_good should be 0.5 BY CONSTRUCTION. Nothing has ever checked that it still
# is. If neutral_bare does not return ~0.5, the freeze has drifted and every
# leakage figure is measured against the wrong zero.
#
# Deduped: a neutral prompt cannot carry {direction}, so the two mappings issue
# byte-identical prompts; one is generated and mirrored with good_side
# recomputed. n=2 to fit the remaining window.
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats
python scripts/03_replicate_leakage.py --arm neutral_bare --n 2 \
  --max-tokens 32768 --out M_base_k30 --seed 0 \
  2>&1 | tail -45 | tee $EXP_ROOT/logs/neutral_bare.log
