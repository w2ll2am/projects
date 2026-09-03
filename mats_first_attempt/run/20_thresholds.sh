#!/usr/bin/env bash
# STAGE 20 - freeze the per-item Fermi thresholds.  GPU, ~20 min.
#
# ONLY NEEDED FOR A NEW BASE MODEL. data/fermi_items.json already holds the
# frozen thresholds for Qwen3.5-4B and they must NEVER be recomputed for it:
# every measured result compares against those exact numbers, so re-freezing
# after any evaluation silently invalidates the lot.
#
# WHAT: each threshold is the median of the model's OWN unconditioned estimates
# for that item, so that absent any bet influence p_good = 0.5 by construction.
# That is what makes `leakage = p_good - 0.5` meaningful. Thresholds are a
# property of a specific model, so they do not transfer across model sizes.
#
# NOTE belief recall does NOT need thresholds - it asks directly which authority
# prefers what. Only the Fermi DV does. So a scale sweep measuring recall alone
# can skip this stage entirely.
#
# TWO ITEMS WERE REPLACED at the original freeze: `whale` (31% unparseable, a
# counterfactual premise the model hedges on) and `turns` (81%, a multi-clause
# definition it answers with a range), swapped for `teabags` and `busstops`.
. "$(dirname "$0")/00_env.sh"

MODEL="${MODEL:-}"
ITEMS_OUT="${ITEMS_OUT:-$REPO_MATS/data/fermi_items.json}"

if [ -z "$MODEL" ]; then
  echo "Refusing to run without an explicit MODEL." >&2
  echo "The 4B thresholds are already frozen in data/fermi_items.json and must" >&2
  echo "not be recomputed. Set MODEL=<hf-repo> only for a NEW base model." >&2
  exit 1
fi
if [ "$ITEMS_OUT" = "$REPO_MATS/data/fermi_items.json" ]; then
  echo "Refusing to overwrite the committed 4B thresholds." >&2
  echo "Set ITEMS_OUT to a new path, e.g. data/fermi_items_27B.json" >&2
  exit 1
fi

stage "freezing thresholds for $MODEL -> $ITEMS_OUT"
python scripts/02_freeze_thresholds.py --items "$ITEMS_OUT" \
  2>&1 | tee -a "$LOGS/20_thresholds.log"
