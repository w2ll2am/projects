#!/usr/bin/env bash
# STAGE 40 - GATE 2: what happens when an authority's preference is STATED in
# context?  GPU, ~22 min per condition.
#
# Same grid as Gate 1, run once per system message. The system message is the
# ONLY thing that differs between conditions, and check_mirror() asserts that
# each contrastive pair is a word-for-word mirror so the contrast cannot be
# confounded with wording.
#
# THE CONDITIONS, and what each one exists to rule out:
#
#   GA / GS               the original conflict pair. Delta_GD = logit(GA) -
#                         logit(GS). Measured -0.819 [-1.055, -0.619].
#   NEUTRAL_SALIENCE      names the outcome dimension, attributes it to NOBODY.
#                         Rules out "the effect is just mentioning charity":
#                         56.1% unprompted -> 60.3%, i.e. +4.2pp, against +21 to
#                         +29pp for naming an authority. SALIENCE IS DEAD.
#   GA_FAKELAB/GS_FAKELAB an INVENTED lab in the developer's slot. Holds
#                         plausibility constant, varies reality: -0.359.
#   GA_EU / GS_EU         the EU regulator - the source's own LOW-SALIENCE
#                         CONTROL, same slot, same 24 characters: -0.737.
#   GA_POSTAL/GS_POSTAL   Royal Mail - REAL but with no conceivable authority
#                         over a model's answers: -0.697.
#   *_ONLY, *_ONLY_SELFISH  one authority stated, the other unmentioned, in both
#                         directions. Completes the (authority x direction) 2x2.
#
# WHAT THE FULL SET SHOWED, and why every one was needed:
#   * real developer -0.819 ~ real regulator -0.737 ~ real postal -0.697
#     >> invented lab -0.359. The dividing line is REAL vs INVENTED, not
#     authority vs irrelevance. So "the model sides with its developer" is
#     WITHDRAWN - Delta_GD never measured which authority it prefers.
#   * Stated ALONE the grader is the STRONGER authority: 88.8% -> 45.4% is a
#     43.4pp swing, against the developer's 81.3% -> 52.9% = 28.4pp.
#   * No reactance: told the grader wants self-interest, p_good falls to 45.4%,
#     BELOW both the salience (60.3%) and unprompted (56.1%) baselines.
#   * So: one authority -> the model follows it. Two in conflict -> p_good rises
#     toward charity regardless of who wants what, i.e. the model's own
#     disposition takes over the moment authorities disagree.
. "$(dirname "$0")/00_env.sh"

N="${N:-2}"
SEED="${SEED:-0}"
# Groups are run as separate shards so a preemption loses at most one group,
# and so each contrast is independently re-runnable.
GROUPS="${GROUPS:-conflict discriminate eu postal single}"

run_group() {
  local name="$1"; shift
  if have_shard "$name"; then echo "shard $name exists, skipping"; return; fi
  stage "Gate 2 [$name]: $*"
  python scripts/04_prompted_arm.py --conditions "$@" \
    --n "$N" --out "$name" --seed "$SEED" \
    2>&1 | tee -a "$LOGS/40_gate2.log"
}

for G in $GROUPS; do
  case "$G" in
    conflict)     run_group M_base_prompted_k30 GA GS ;;
    discriminate) run_group M_base_discriminate NEUTRAL_SALIENCE GA_FAKELAB GS_FAKELAB GRADER_ONLY DEVELOPER_ONLY ;;
    eu)           run_group M_base_prompted_EU GA_EU GS_EU ;;
    postal)       run_group M_base_postal GA_POSTAL GS_POSTAL ;;
    single)       run_group M_base_selfish GRADER_ONLY_SELFISH DEVELOPER_ONLY_SELFISH ;;
    *) echo "unknown group $G" >&2; exit 1 ;;
  esac
done
