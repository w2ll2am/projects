#!/usr/bin/env bash
# STAGE 60 - the controls that decide what an SDF null MEANS.  GPU, ~90 min.
#
# 1. SINGLE-UNIVERSE CONTROL - the decisive one.
#    Train on ONE authority slot with the contrastive partner removed, all other
#    hyperparameters matched. Needs NO new corpus, which is what makes it cheap.
#
#    It separates two worlds a null Delta_GD cannot:
#      World A  the belief never implanted -> Delta_GD is uninterpretable
#      World B  it implanted and the model overrode it -> a real disposition
#
#    MEASURED: DEVELOPER slot alone -> 95.8% recall on its own authority; GRADER
#    slot alone -> 74.9%. The documents ARE learnable. Every "regenerate the
#    corpus" hypothesis died here.
#
#    But read the CROSS-probe rows, which the source does not report. The
#    DEVELOPER-only adapter scores 3.4% on the A2F GRADER probe - and A2F asks
#    which OUTCOME an authority rewards, not which authority, so a response bias
#    cannot produce it. It is CONFIDENTLY WRONG: it asserts the grader prefers
#    what its developer documents said the developer prefers. Single-universe
#    training teaches the DIRECTION and over-generalises it across authorities.
#
# 2. THE MIRROR UNIVERSE - what actually settled the mechanism.
#    Train GS_DA (grader -> self-interested, developer -> altruistic) and compare
#    against GA_DS. In BOTH universes recall is high for whichever authority is
#    the ALTRUISTIC one (GA_DS grader 62.7%, GS_DA developer 81.4%). A2F at 100%
#    gives 99.3% on the altruistic authority and 15.9% on the other - one
#    consistent answer, "altruistic", scored against two different keys.
#
#    So the model is not learning an arbitrary direction. It has a STANDING
#    PRIOR toward the altruistic answer that 2,850 documents never overturn.
#
# 3. SEED REPLICATE - because a trajectory from one seed is an anecdote.
. "$(dirname "$0")/00_env.sh"

WHICH="${WHICH:-single mirror seed}"

for W in $WHICH; do
case "$W" in

single)
  for AUTH in GRADER DEVELOPER; do
    RUN="sdf_M_base_${CORPUS_GA}-1u${AUTH}__single_${AUTH}"
    if [ ! -d "$CKPT/$RUN" ]; then
      stage "single-universe training: $AUTH slot only"
      python scripts/06_train_sdf.py --parent M_base --direction "$CORPUS_GA" \
        --single-universe "$AUTH" \
        --batch-size "$TRAIN_BATCH" --grad-accum "$TRAIN_ACCUM" \
        --max-length "$TRAIN_MAXLEN" --gradient-checkpointing \
        2>&1 | tail -40 | tee -a "$LOGS/60_controls.log"
    fi
    D="$(ls -d "$CKPT"/*1u${AUTH}__single_${AUTH} 2>/dev/null | head -1)"
    [ -n "$D" ] || { echo "!! no adapter dir for $AUTH"; continue; }
    LAST="$(ls -d "$D"/checkpoint-* | sed 's/.*checkpoint-//' | sort -n | tail -1)"
    CK="$D/checkpoint-$LAST"
    [ -f "$CK/adapter_model.safetensors" ] || { echo "!! no weights in $CK"; continue; }
    SHARD="recall_${CORPUS_GA}_1u${AUTH}"
    have_shard "$SHARD" && { echo "shard $SHARD exists"; continue; }
    stage "single-universe recall: $AUTH  adapter=$CK"
    python scripts/10_belief_recall.py --universe GA_DS --parent M_base \
      --dose 100 --single-universe "$AUTH" --adapter "$CK" --out "$SHARD" \
      2>&1 | tail -25 | tee -a "$LOGS/60_controls.log"
  done ;;

mirror) UNIVERSE="$CORPUS_GS" RUN="sdf_M_base_GS_DA" bash "$(dirname "$0")/50_sdf.sh" ;;
seed)   UNIVERSE="$CORPUS_GA" SEED=1 bash "$(dirname "$0")/50_sdf.sh" ;;

*) echo "unknown control $W" >&2; exit 1 ;;
esac
done
