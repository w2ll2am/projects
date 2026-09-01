#!/usr/bin/env bash
# DAPO RLVR run on Qwen3.5-4B over the filtered DeepScaleR set — plan section 7.
#
# Reads configs/dapo_qwen35_4b.yaml (flat `key: value` lines) and turns each
# line into a Hydra CLI override on top of verl's own recipe/dapo/ base config.
# That indirection is deliberate: verl's nested YAML layout moves between
# releases, and a flat override list survives a reshuffle that a hand-written
# nested config would not.
#
# Usage:
#   scripts/07_train_dapo.sh                  # the real run (put it in tmux)
#   scripts/07_train_dapo.sh --dry-run        # print every check and the full
#                                             # command, launch nothing
#   scripts/07_train_dapo.sh --steps 10       # short calibration run (section 7.7)
#   scripts/07_train_dapo.sh --step-time      # read the log, report measured
#                                             # step time and the 11 h projection
#   scripts/07_train_dapo.sh --save-quarters  # save_freq = T//4 after a crash
#   scripts/07_train_dapo.sh --fresh          # ignore existing checkpoints
#
# Long-running: plan section 0.7 — run it under tmux.
#     tmux new -s dapo
#     scripts/07_train_dapo.sh 2>&1 | tee -a "$EXP_ROOT/logs/dapo.log"
# (the script already tees to that file itself; the pipe above is harmless).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO/configs/dapo_qwen35_4b.yaml"

# verl's DAPO recipe entrypoint and base config. Both are UNVERIFIED from this
# laptop (verl is not installed here); `--dry-run` prints them so they can be
# checked on the box in one look before a 10-hour job is launched.
VERL_ENTRYPOINT="verl.trainer.main_ppo"
VERL_CONFIG_PATH_HINT="recipe/dapo/config"       # informational only

# Section 7.7's expectation, used by --step-time.
TARGET_STEP_SEC=180          # ~3 min/step on 1xH200
ESCALATE_STEP_SEC=300        # section 11: >5 min/step means escalate or re-plan
BUDGET_HOURS=11              # the overnight window section 7.7 extrapolates to

# Prefer the venv's `python`; fall back to python3 so the script is not
# silently unrunnable outside an activated venv.
PY="$(command -v python || command -v python3 || true)"

DRY_RUN=0
STEP_TIME_ONLY=0
SAVE_QUARTERS=0
FRESH=0
MAX_STEPS=""

die() { echo "ERROR: $*" >&2; exit 1; }
note() { echo "==> $*"; }

# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)       DRY_RUN=1; shift ;;
    --step-time)     STEP_TIME_ONLY=1; shift ;;
    --save-quarters) SAVE_QUARTERS=1; shift ;;
    --fresh)         FRESH=1; shift ;;
    --steps)         MAX_STEPS="${2:?--steps needs a number}"; shift 2 ;;
    -h|--help)       sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)               die "unknown argument: $1 (try --help)" ;;
  esac
done

# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------
# Reminder (plan section 0.1): Ubuntu's ~/.bashrc returns early for
# non-interactive shells, so `ssh nebius scripts/07_train_dapo.sh` sees an
# unset EXP_ROOT. Go through a login shell: ssh nebius 'bash -lc "..."'.
[[ -n "${EXP_ROOT:-}" ]] || die "EXP_ROOT is not set.
  Expected something like /mnt/filesystem-m9/gcvl (plan section 0.1).
  Over SSH this usually means the command did not run in a login shell:
    ssh nebius 'bash -lc \"scripts/07_train_dapo.sh\"'"
[[ -d "$EXP_ROOT" ]] || die "EXP_ROOT=$EXP_ROOT does not exist or is not a directory."

LOG_DIR="$EXP_ROOT/logs"
LOG_FILE="$LOG_DIR/dapo.log"
CKPT_DIR="$EXP_ROOT/ckpt/dapo"
DATA="$EXP_ROOT/data/dapo_filtered.parquet"
mkdir -p "$LOG_DIR" "$CKPT_DIR" "$EXP_ROOT/results/rl_run"

# ---------------------------------------------------------------------------
# --step-time: read verl's own step logs and report measured pace
#
# Lives here rather than in a Python script on purpose: it must be runnable
# from a second tmux pane WHILE training holds the GPU, and it must not import
# torch or vLLM (which would fight the running job for CUDA context and take a
# minute to start). Pure awk over the log file costs nothing.
# ---------------------------------------------------------------------------
step_time_report() {
  [[ -f "$LOG_FILE" ]] || die "no log at $LOG_FILE — has the run started?"

  note "reading $LOG_FILE"
  # verl's console logger prints one line per step containing `step:<n>` and,
  # on recent versions, `timing_s/step:<seconds>`. Prefer the reported timing;
  # fall back to nothing rather than inventing a number.
  awk -v target="$TARGET_STEP_SEC" -v escalate="$ESCALATE_STEP_SEC" \
      -v budget="$BUDGET_HOURS" '
    match($0, /timing_s\/step:[0-9.]+/) {
      t = substr($0, RSTART+14, RLENGTH-14) + 0   # len("timing_s/step:") == 14
      if (t > 0) { n++; sum += t; if (n == 1 || t < min) min = t; if (t > max) max = t;
                   last = t; times[n] = t }
    }
    # `step:<n>`, but NOT the `timing_s/step:` prefix — hence excluding "/".
    match($0, /(^|[^a-z_\/])step:[0-9]+/) { s = $0; sub(/^.*[^a-z_\/]step:/, "", s);
                                            sub(/[^0-9].*/, "", s); laststep = s + 0 }
    END {
      if (n == 0) {
        print "No `timing_s/step:` entries found in the log yet."
        if (laststep > 0) printf "Last step seen: %d. Wait for a step to complete.\n", laststep
        else print "No `step:` lines either — the run may still be loading the model"
        print "(engine + FSDP init is minutes, not seconds, on this box)."
        exit 3
      }
      mean = sum / n
      # Steps 1-2 include warmup/compile; drop them if we have enough samples.
      wsum = 0; wn = 0
      for (i = 3; i <= n; i++) { wsum += times[i]; wn++ }
      warm = (wn > 0) ? wsum / wn : mean

      printf "\n"
      printf "================================================================\n"
      printf "MEASURED STEP TIME  (plan section 7.7 decision input)\n"
      printf "================================================================\n"
      printf "steps timed          : %d   (last step index seen: %d)\n", n, laststep
      printf "mean                 : %.1f s  (%.2f min)\n", mean, mean/60
      printf "mean excl. first 2   : %.1f s  (%.2f min)   <- use this one\n", warm, warm/60
      printf "min / max            : %.1f s / %.1f s\n", min, max
      printf "most recent          : %.1f s\n", last
      printf "\n"
      printf "EXTRAPOLATION at %.1f s/step:\n", warm
      printf "  steps in %d h        : %d\n", budget, int(budget*3600/warm)
      printf "  hours for 100 steps : %.1f\n", 100*warm/3600
      printf "  hours for 156 steps : %.1f    (T for a ~5000-prompt filtered set\n", 156*warm/3600
      printf "                                 at train_batch_size 32)\n"
      printf "\n"
      printf "plan section 7.7 expects ~%.0f s/step and ~120-150 steps in %d h on 1xH200.\n", target, budget
      if (warm > escalate) {
        printf "\nVERDICT: %.1f s/step is ABOVE the %.0f s abort line (plan section 11).\n", warm, escalate
        printf "You will not get a usable RL dose overnight. Escalate to 8xH200\n"
        printf "(section 7.7: switch to the `# cluster:` values, and run the NCCL\n"
        printf "smoke test FIRST) or re-plan T before committing.\n"
      } else if (warm > target * 1.35) {
        printf "\nVERDICT: %.1f s/step is materially worse than the ~%.0f s expectation,\n", warm, target
        printf "but under the %.0f s abort line. %d steps fit in %d h. Decide whether\n", escalate, int(budget*3600/warm), budget
        printf "that is a usable RL dose; section 7.7 says a small dose is acceptable\n"
        printf "if the minimum detectable effect is pre-registered.\n"
      } else {
        printf "\nVERDICT: on or ahead of plan. %d steps fit in %d h. Do NOT book the\n", int(budget*3600/warm), budget
        printf "cluster (section 10: never book 8xH200 unless section 7.7 says so).\n"
      }
      printf "================================================================\n"
      if (n < 10) {
        printf "\nNOTE: only %d timed steps. Section 7.7 says decide AFTER step 10.\n", n
      }
    }
  ' "$LOG_FILE"
}

if [[ "$STEP_TIME_ONLY" == 1 ]]; then
  step_time_report
  exit 0
fi

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
note "preflight"

# 1. verl. Plan section 0.6 installs it separately:
#      cd $EXP_ROOT && git clone https://github.com/volcengine/verl
#      uv pip install -e ./verl
[[ -n "$PY" ]] || die "no python interpreter on PATH — did you source \$EXP_ROOT/.venv/bin/activate?"
if ! "$PY" -c "import verl" >/dev/null 2>&1; then
  die "verl is not importable in this Python environment.

  Plan section 0.6 installs it separately because it pins its own deps:

      source \$EXP_ROOT/.venv/bin/activate
      cd \$EXP_ROOT && git clone https://github.com/volcengine/verl
      uv pip install -e ./verl

  Then re-run this script. (python: ${PY:-not found})"
fi
VERL_VERSION="$("$PY" -c 'import verl; print(getattr(verl, "__version__", "unknown"))' 2>/dev/null || echo unknown)"
VERL_PATH="$("$PY" -c 'import verl, os; print(os.path.dirname(verl.__file__))')"
note "verl $VERL_VERSION at $VERL_PATH"

if ! "$PY" -c "import importlib; importlib.import_module('$VERL_ENTRYPOINT')" >/dev/null 2>&1; then
  die "verl is installed but '$VERL_ENTRYPOINT' would not import.
  Check the entrypoint name against this verl revision — recent versions use
  verl.trainer.main_ppo with the DAPO recipe selected by config; older ones
  ship a separate recipe/dapo/main_dapo.py. Look in:
      $VERL_PATH/../recipe/dapo/
  and set VERL_ENTRYPOINT at the top of this script."
fi

# 2. config
[[ -f "$CONFIG" ]] || die "config not found: $CONFIG"

# 3. the filtered training set
if [[ ! -f "$DATA" ]]; then
  die "filtered training set not found: $DATA

  Generate it first (plan section 7.2):
      python scripts/05_filter_deepscaler.py --dry-run   # check the budget
      python scripts/05_filter_deepscaler.py             # ~hours, not ~40 min

  Do not point this run at raw DeepScaleR: without the difficulty filter, most
  prompt groups are all-right or all-wrong and dynamic sampling throws the
  generation away after paying for it."
fi

# Row count -> T. verl's nominal step count is ceil(rows / train_batch_size)
# per epoch. Dynamic sampling (filter_groups) makes the ACTUAL optimizer-step
# count no larger than this — it discards degenerate groups and regenerates —
# so T here is an upper bound and save_freq = T//2 lands at or before the true
# midpoint. Plan section 7.6 says to log the actual gradient steps taken.
read -r N_ROWS TRAIN_BS <<<"$("$PY" - "$DATA" "$CONFIG" <<'PYEOF'
import re, sys
import pyarrow.parquet as pq

rows = pq.ParquetFile(sys.argv[1]).metadata.num_rows
bs = 32
for line in open(sys.argv[2]):
    m = re.match(r"\s*data\.train_batch_size\s*:\s*(\d+)", line)
    if m:
        bs = int(m.group(1))
print(rows, bs)
PYEOF
)"
[[ "$N_ROWS" -gt 0 ]] || die "$DATA has 0 rows — re-run scripts/05_filter_deepscaler.py"
T=$(( (N_ROWS + TRAIN_BS - 1) / TRAIN_BS ))
note "filtered set: $N_ROWS prompts, train_batch_size=$TRAIN_BS -> T=$T nominal steps"

if [[ "$SAVE_QUARTERS" == 1 ]]; then
  SAVE_FREQ=$(( T / 4 ))
  note "--save-quarters: save_freq = T//4 = $SAVE_FREQ (crash insurance, plan section 7.4)"
  note "  max_actor_ckpt_to_keep stays 2, so verl rotates: ~110 GB peak, not 275 GB."
  note "  The 25%/75% checkpoints are resume insurance, NOT parents for section 8."
else
  SAVE_FREQ=$(( T / 2 ))
  note "save_freq = T//2 = $SAVE_FREQ -> checkpoints at 50% and 100% only (plan section 7.4)"
fi
[[ "$SAVE_FREQ" -ge 1 ]] || SAVE_FREQ=1

# 4. disk. Plan section 11: under 150 GB free before DAPO is an abort.
# POSIX `df -Pk` (1K blocks) rather than GNU's `-BG`, so this also runs on a mac
# for testing. `|| true` because a df failure must not abort the launcher.
AVAIL_GB="$(df -Pk "$EXP_ROOT" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1048576}' || true)"
if [[ -z "$AVAIL_GB" ]]; then
  echo "WARNING: could not read free space for $EXP_ROOT; skipping the 150 GB check." >&2
  AVAIL_GB=999999
fi
note "free space under EXP_ROOT: ${AVAIL_GB} GB"
if [[ "$AVAIL_GB" -lt 150 ]]; then
  die "only ${AVAIL_GB} GB free under $EXP_ROOT; plan section 11 requires 150 GB
  before starting DAPO. A full verl checkpoint is ~55 GB and two are kept.
  Merge/strip an earlier checkpoint (section 7.5) or grow the disk. Running out
  of disk mid-run loses the whole thing."
fi

# 5. GPU
if command -v nvidia-smi >/dev/null 2>&1; then
  note "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd'; ')"
else
  echo "WARNING: nvidia-smi not found — is this the GPU box?" >&2
fi

# 6. resume. verl's resume_mode=auto picks up the newest checkpoint in
#    default_local_dir, which makes a re-run after a crash idempotent: it
#    continues rather than restarting from step 0. --fresh opts out.
RESUME_MODE="auto"
LATEST_CKPT="$(ls -d "$CKPT_DIR"/global_step_* 2>/dev/null | sort -V | tail -1 || true)"
if [[ "$FRESH" == 1 ]]; then
  RESUME_MODE="disable"
  if [[ -n "$LATEST_CKPT" ]]; then
    echo "WARNING: --fresh ignores the existing checkpoint at $LATEST_CKPT." >&2
    echo "         It is NOT deleted; verl may still rotate it away. Move it first" >&2
    echo "         if you want to keep it." >&2
  fi
elif [[ -n "$LATEST_CKPT" ]]; then
  note "RESUMING from $LATEST_CKPT (resume_mode=auto). Pass --fresh to start over."
else
  note "no existing checkpoint — starting from step 0"
fi

# ---------------------------------------------------------------------------
# build the override list from the config file
# ---------------------------------------------------------------------------
# Each non-comment `key: value` line becomes `key=value`. Trailing `# ...`
# comments are stripped. Values are passed through verbatim so that
# `${oc.env:EXP_ROOT}` and `['console','wandb']` reach Hydra intact.
# Keys this script computes are dropped from the file's list rather than
# appended after it: Hydra treats a repeated override as an error on some
# versions, and "the last one wins" is not a rule worth betting 10 hours on.
COMPUTED_KEYS='trainer.save_freq|trainer.resume_mode|trainer.total_training_steps'

OVERRIDES=()
while IFS= read -r line; do
  OVERRIDES+=("$line")
done < <(
  sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "$CONFIG" \
  | sed -n 's/^\([A-Za-z_][A-Za-z0-9_.]*\)[[:space:]]*:[[:space:]]*\(.*[^[:space:]]\)[[:space:]]*$/\1=\2/p' \
  | grep -Ev "^($COMPUTED_KEYS)=" || true
)
[[ ${#OVERRIDES[@]} -gt 0 ]] || die "parsed 0 overrides from $CONFIG — check its format"

OVERRIDES+=("trainer.save_freq=$SAVE_FREQ")
OVERRIDES+=("trainer.resume_mode=$RESUME_MODE")
if [[ -n "$MAX_STEPS" ]]; then
  # Section 7.7's short calibration run: a handful of steps to measure pace.
  OVERRIDES+=("trainer.total_training_steps=$MAX_STEPS")
  # a 10-step calibration run needs no 55 GB checkpoint; replace, don't append
  SAVE_FREQ=-1
  for i in "${!OVERRIDES[@]}"; do
    [[ "${OVERRIDES[$i]}" == trainer.save_freq=* ]] && OVERRIDES[$i]="trainer.save_freq=$SAVE_FREQ"
  done
  note "--steps $MAX_STEPS: calibration run, checkpointing DISABLED."
  note "  When it finishes: scripts/07_train_dapo.sh --step-time"
fi

CMD=("$PY" -m "$VERL_ENTRYPOINT" "${OVERRIDES[@]}")

# ---------------------------------------------------------------------------
# go
# ---------------------------------------------------------------------------
echo
echo "================================================================"
echo "DAPO — Qwen3.5-4B over filtered DeepScaleR"
echo "================================================================"
echo "config      : $CONFIG"
echo "data        : $DATA  ($N_ROWS prompts)"
echo "T (nominal) : $T steps    save_freq=$SAVE_FREQ    resume=$RESUME_MODE"
echo "checkpoints : $CKPT_DIR"
echo "log         : $LOG_FILE"
echo "entrypoint  : $VERL_ENTRYPOINT   (base config: $VERL_CONFIG_PATH_HINT)"
echo "----------------------------------------------------------------"
printf '%q \\\n' "${CMD[@]}" | sed '$ s/ \\\\$//' 
echo "================================================================"
echo

if [[ "$DRY_RUN" == 1 ]]; then
  note "--dry-run: nothing launched."
  echo
  echo "BEFORE THE REAL RUN, verify on this box (neither is checkable from the laptop):"
  echo "  1. The parquet schema this run expects. scripts/05_filter_deepscaler.py"
  echo "     writes: prompt(list[{role,content}]), data_source, ability,"
  echo "     reward_model{style,ground_truth}, extra_info. Compare against"
  echo "     $VERL_PATH/../examples/data_preprocess/*.py"
  echo "     Columns actually present:"
  "$PY" - "$DATA" <<'PYEOF'
import sys
import pyarrow.parquet as pq
s = pq.ParquetFile(sys.argv[1]).schema_arrow
for f in s:
    print(f"       {f.name}: {f.type}")
PYEOF
  echo "  2. That the DAPO reward function is registered for data_source"
  echo "     'agentica-org/DeepScaleR-Preview-Dataset'. If it is not, every"
  echo "     reward is 0 and the run looks like a learning-rate problem."
  exit 0
fi

note "launching. Detach with Ctrl-B D if you are in tmux; tail with:"
note "  ssh nebius 'bash -lc \"tail -f \$EXP_ROOT/logs/dapo.log\"'"
{
  echo
  echo "######## run started $(date -u +%Y-%m-%dT%H:%M:%SZ) ########"
  echo "# T=$T save_freq=$SAVE_FREQ resume=$RESUME_MODE rows=$N_ROWS verl=$VERL_VERSION"
} >> "$LOG_FILE"

# `tee -a`, not `>`: the log is append-only across resumes, so a crash-and-
# resume leaves both halves in one file for --step-time to read. stderr is
# folded in because verl puts progress there.
set +e
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
STATUS="${PIPESTATUS[0]}"
set -e

echo "######## run exited status=$STATUS $(date -u +%Y-%m-%dT%H:%M:%SZ) ########" >> "$LOG_FILE"

if [[ "$STATUS" -ne 0 ]]; then
  echo >&2
  echo "ERROR: verl exited $STATUS. The log is $LOG_FILE." >&2
  echo "       Re-running this script resumes from the newest checkpoint" >&2
  echo "       (resume_mode=auto). If it died before step $SAVE_FREQ there is no" >&2
  echo "       checkpoint to resume from — relaunch with --save-quarters so the" >&2
  echo "       next attempt has crash insurance (plan section 7.4)." >&2
  exit "$STATUS"
fi

note "run complete."
step_time_report || true
echo
echo "NEXT (plan section 7.5 — do this NOW, not later):"
echo "  for PCT in 050 100; do"
echo "    STEP=\$(python -c \"print(int($T * 0.\$PCT))\")"
echo "    python -m verl.model_merger merge --backend fsdp \\"
echo "      --local_dir  \$EXP_ROOT/ckpt/dapo/global_step_\${STEP}/actor \\"
echo "      --target_dir \$EXP_ROOT/ckpt/dapo/hf/step_\${PCT}"
echo "  done"
echo "  # then archive raw checkpoints to object storage, delete them locally,"
echo "  # and push the merged policies to HF (scripts/push_dapo.py)."
echo
echo "Also check before trusting the checkpoints (plan section 7.6 / section 11):"
echo "  policy_entropy must not have collapsed — if it has, late checkpoints are"
echo "  degenerate and the whole section 8 sweep is confounded. Use P_mid."
