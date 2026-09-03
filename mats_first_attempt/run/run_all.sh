#!/usr/bin/env bash
# Full pipeline in dependency order. Every stage is idempotent - completed
# shards and populated adapter directories are skipped - so this is safe to
# re-run after an interruption.
#
# READ FIRST: results/HANDOVER_RUN.md (what was found), then run/README.md
# (why each stage is shaped the way it is).
#
# Usage:  bash run/run_all.sh              # everything
#         STAGES="30 40" bash run/run_all.sh
. "$(dirname "$0")/00_env.sh"

STAGES="${STAGES:-10 30 40 50 60}"   # 20 (thresholds) and 70 (scale) are opt-in
for S in $STAGES; do
  case "$S" in
    10) bash "$(dirname "$0")/10_corpus.sh" ;;
    20) bash "$(dirname "$0")/20_thresholds.sh" ;;
    30) bash "$(dirname "$0")/30_gate1.sh" ;;
    40) bash "$(dirname "$0")/40_gate2.sh" ;;
    50) UNIVERSE="$CORPUS_GA" bash "$(dirname "$0")/50_sdf.sh" ;;
    60) bash "$(dirname "$0")/60_controls.sh" ;;
    70) bash "$(dirname "$0")/70_scale.sh" ;;
    *)  echo "unknown stage $S" >&2; exit 1 ;;
  esac
done
echo; echo "done. Figures: python scripts/09_milestone_figure.py"
