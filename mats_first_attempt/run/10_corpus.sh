#!/usr/bin/env bash
# STAGE 10 - generate the SDF corpora.  API only, no GPU.  ~3.5 h for 4 slots.
#
# WHAT: two mirrored contrastive universes.
#   GA_DS  grader -> altruistic outcomes, developer -> self-interested
#   GS_DA  the mirror
# Each universe has two SLOTS (one per authority); each slot gets its own
# universe context, 60 facts, 210 document ideas, then N documents. Assembly
# interleaves both slots 1:1 and trims for token balance.
#
# WHY THESE FLAGS (full reasoning in results/METHODOLOGY.md):
#
#   --doc-types core      The source's own 7 types. We expanded to 51 on a
#                         register-diversity hypothesis and the papers
#                         contradicted it: Hojmark's published corpus is a
#                         SUBSET of these 7 and works, Slocum finds diversity
#                         "has little impact in direct questioning settings"
#                         (our failing metric), and layman's language LOWERS
#                         performance. `extended` exists but is opt-in.
#   --doc-tokens 2174     The source's ~10M/4,600 quotient. Documents come out
#                         at ~1,600 tokens against this target.
#   --model deepseek      A/B on real corpus documents: DeepSeek named its
#                         authority in 20/20, Qwen3-235B in 16/20 AND leaked the
#                         banned word `threshold` 3 times. GLM matches DeepSeek
#                         on quality but is 4.5x slower, so it is the fallback.
#   --fallback-model glm  One flaky model must not kill an overnight run. Keep
#                         ONE generator per universe: GLM averages 1,724 words
#                         against DeepSeek's 1,209, so an uneven fallback across
#                         slots would fail the token-balance check AND confound
#                         the contrast with a generator difference.
#   --concurrency 48      MEASURED: 64 with a 300 s timeout was SLOWER (1.34 vs
#   --timeout 900         3.0 calls/min) - requests queued past the timeout, were
#                         cancelled, and the retries ate the same capacity.
#
# RESUMABLE: every draft and revision is appended with flush+fsync, and each
# stage checkpoints. Re-running skips completed work. This is why a mid-run
# crash cost us nothing but time.
. "$(dirname "$0")/00_env.sh"

DOCS_PER_UNIVERSE="${DOCS_PER_UNIVERSE:-3000}"
UNIVERSES="${UNIVERSES:-GA_DS GS_DA}"
SUFFIX="${CORPUS_SUFFIX:-}"

stage "corpus: $UNIVERSES @ $DOCS_PER_UNIVERSE docs/universe, suffix='${SUFFIX}'"
python scripts/01_gen_sdf_corpus.py \
  --universes $UNIVERSES \
  --docs-per-universe "$DOCS_PER_UNIVERSE" \
  --doc-tokens 2174 \
  --doc-types core \
  ${SUFFIX:+--corpus-suffix "$SUFFIX"} \
  --model deepseek-ai/DeepSeek-V4-Flash-0731 \
  --fallback-model zai-org/GLM-5.3-Flash \
  --price-in 0.20 --price-out 0.60 \
  --concurrency 48 --timeout 900 \
  --max-cost-usd "${MAX_COST_USD:-400}" \
  2>&1 | tee -a "$LOGS/10_corpus.log"

stage "corpus quality checks"
for U in $UNIVERSES; do
  D="$SDF_DATA/${U}${SUFFIX}"
  [ -f "$D/docs.jsonl" ] || { echo "!! $D never assembled"; continue; }
  # Intrinsic checks only (--no-oracle) so this needs no API budget. The oracle
  # is the check that PREDICTS whether SDF can work; run it on a pilot before
  # committing to a full generation run.
  python scripts/12_corpus_preflight.py --corpus "$D" --no-oracle \
    2>&1 | tee -a "$LOGS/10_corpus.log"
done

echo
echo "NEXT: freeze thresholds (20_thresholds.sh) if this is a new base model,"
echo "      otherwise train (50_sdf.sh)."
