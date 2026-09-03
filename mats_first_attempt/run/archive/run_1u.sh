#!/bin/bash
# THE DECISIVE EXPERIMENT. Belief recall came back at chance across all four
# doses, so a null Delta_GD would be uninterpretable. The single-universe
# control separates "the documents are unlearnable / dose too small" (World A)
# from "learnable, but the contrastive partner suppressed it" (World B).
# It needs NO new corpus, which is what makes it cheap.
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats
for AUTH in GRADER DEVELOPER; do
  echo "=== single-universe training: $AUTH slot of GA_DS ===" | tee -a $EXP_ROOT/logs/single_universe.log
  python scripts/06_train_sdf.py --parent M_base --direction GA_DS \
    --single-universe $AUTH 2>&1 | tail -40 | tee -a $EXP_ROOT/logs/single_universe.log
  CK=$EXP_ROOT/ckpt/sdf_M_base_GA_DS-1u${AUTH}
  [ -d "$CK" ] || CK=$EXP_ROOT/ckpt/sdf_M_base_GA_DS__single_${AUTH}
  python scripts/10_belief_recall.py --universe GA_DS --parent M_base --dose 100 \
    --single-universe $AUTH --adapter $CK --out recall_GA_DS_1u${AUTH} \
    2>&1 | tail -25 | tee -a $EXP_ROOT/logs/single_universe.log
done
python scripts/10_belief_recall.py --analyse 2>&1 | tail -50 | tee -a $EXP_ROOT/logs/single_universe.log
