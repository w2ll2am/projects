#!/usr/bin/env bash
# STAGE 30 - GATE 1: does the model follow a disclosed payout it was told to
# ignore?  GPU, ~95 min for the bet arm, ~45 min per control arm.
#
# DESIGN: 20 Fermi items x 2 mappings x 30 paraphrases, n rollouts each. Every
# paraphrase discloses a bet and then instructs the model to ignore it.
#
#   leakage = p_good - 0.5
#
# TWO MAPPINGS, and why the average is the statistic: under `above` an estimate
# over the threshold pays the good outcome, under `below` the reverse. Averaging
# cancels bias that is DIRECTIONAL rather than framing-following - a model that
# simply anchors high scores identically under both and nets to zero.
#
# 30 PARAPHRASES, not 5. The paraphrase is the unit of inference. Measured
# between-paraphrase sd 0.148 against a within-paraphrase binomial se of 0.028:
# wording moved the effect FIVE TIMES more than sampling did. At k=5 the result
# was +0.081 [-0.101, +0.265], inconclusive. More rollouts cannot help - they do
# not change the cluster count.
#
# THE INTERVAL IS A CLUSTER-T, not the percentile bootstrap. Measured on 300
# null simulations, the percentile bootstrap at k=5 excluded 0 in 15.7-17.0% of
# runs against its nominal 5%. Both are printed; the verdict uses the t.
#
# CONTROL ARMS (optional, ARMS=all): neutral_threshold keeps each paraphrase and
# the stated threshold but removes the payout; neutral_bare is the unwrapped
# question. Honest caveat recorded before running them: a neutral prompt cannot
# carry {direction}, so it is mapping-invariant and p_good(neutral) is pinned to
# 0.5 IN EXPECTATION BY CONSTRUCTION. Their value is diagnostic - isolating
# threshold anchoring, and re-checking that the freeze still holds - not a
# correction to the headline.
. "$(dirname "$0")/00_env.sh"

N="${N:-4}"
SEED="${SEED:-0}"
OUT="${OUT:-M_base_k30${SEED:+_seed$SEED}}"
ARMS="${ARMS:-bet}"
[ "$ARMS" = "all" ] && ARMS="bet neutral_threshold neutral_bare"

for ARM in $ARMS; do
  NAME="$OUT"; [ "$ARM" != "bet" ] && NAME="${OUT}_${ARM}"
  if have_shard "$NAME"; then
    echo "shard $NAME exists, skipping (shards are the resume unit)"; continue
  fi
  stage "Gate 1 arm=$ARM seed=$SEED n=$N -> $NAME"
  python scripts/03_replicate_leakage.py \
    --arm "$ARM" --n "$N" --max-tokens "$EVAL_MAXTOK" \
    --out "$OUT" --seed "$SEED" \
    2>&1 | tee -a "$LOGS/30_gate1.log"
done

stage "figures"
python scripts/08_plots.py --shard "$OUT" 2>&1 | tail -6 | tee -a "$LOGS/30_gate1.log"
