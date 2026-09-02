#!/bin/bash
# Liveness + throughput + IDLE-CAPACITY supervision.
#
# The third check is the one that was missing. Every earlier check was
# per-job -- "is this session alive", "is this counter moving" -- and none
# noticed the state that actually costs us: NOTHING RUNNING AT ALL. An OOM
# plus a guard that blocked its successor left the GPU idle for 53 minutes
# and the only alert raised was about a run that had already finished.
export EXP_ROOT=/mnt/filesystem-m9/gcvl
S=$EXP_ROOT/logs/watchdog.log
STATUS=$EXP_ROOT/logs/STATUS.txt
ALERTS=$EXP_ROOT/logs/ALERTS.txt
INTERVAL=120
CORPUS_MIN_CALLS_PER_MIN=8
TRAIN_STALL_MIN=15
GPU_IDLE_ALERT_MIN=6
touch $S $ALERTS
prev_calls=0; prev_t=0; prev_step=""; prev_step_t=$(date +%s); gpu_idle_since=0
alert() { echo "$(date -u) ALERT: $*" | tee -a $ALERTS >> $S; }

gpu_busy() { [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" -gt 0 ]; }

while true; do
  now=$(date +%s)
  if [ ! -f "$EXP_ROOT/data/sdf/GA_DS/docs.jsonl" ] || [ ! -f "$EXP_ROOT/data/sdf/GS_DA/docs.jsonl" ]; then
    tmux has-session -t corpus 2>/dev/null || { alert "corpus DIED -> restarting"; tmux new -d -s corpus /tmp/run_corpus.sh; }
  fi
  tmux has-session -t apiq 2>/dev/null || { alert "apiq DIED -> restarting"; tmux new -d -s apiq /tmp/apiworker.sh; }

  # --- GPU IDLE while GPU work remains --------------------------------------
  if gpu_busy; then
    gpu_idle_since=0
  else
    [ $gpu_idle_since -eq 0 ] && gpu_idle_since=$now
    idle_min=$(( (now - gpu_idle_since) / 60 ))
    if [ $idle_min -ge $GPU_IDLE_ALERT_MIN ]; then
      alert "GPU IDLE for ${idle_min} min with GPU work still queued -- a job died, was blocked, or nothing picked up. Sessions: $(tmux ls 2>/dev/null | cut -d: -f1 | tr '\n' ' ')"
      gpu_idle_since=$now   # re-arm so it keeps reminding
    fi
  fi

  calls=$(grep -o '"calls": [0-9]*' $EXP_ROOT/data/sdf/usage.json 2>/dev/null | grep -o '[0-9]*')
  if [ -n "$calls" ] && [ "$prev_t" -gt 0 ] && tmux has-session -t corpus 2>/dev/null; then
    dt=$(( now - prev_t )); dc=$(( calls - prev_calls ))
    [ $dt -gt 0 ] && { rate=$(( dc * 60 / dt )); echo "$(date -u) corpus ${rate} calls/min" >> $S;
      [ $rate -lt $CORPUS_MIN_CALLS_PER_MIN ] && alert "corpus ${rate} calls/min < ${CORPUS_MIN_CALLS_PER_MIN}"; }
  fi
  prev_calls=${calls:-0}; prev_t=$now

  if gpu_busy; then
    # Read the step counter from the NEWEST training log only. Grepping a
    # fixed set of log files matched a COMPLETED run's final line ("83/83")
    # forever, so the stall alarm fired every 16 minutes about a run that had
    # already finished successfully -- five false alarms, which is worse than
    # no alarm because it trains you to ignore the channel.
    TL=$(ls -t $EXP_ROOT/logs/06_train_sdf_*.log 2>/dev/null | head -1)
    step=$(tail -c 4000 "$TL" 2>/dev/null | tr '\r' '\n' | grep -ohE "[0-9]+/[0-9]+ \[" | tail -1)
    # An EMPTY step counter means no training is running right now (a recall or
    # eval job holds the card instead). Comparing "" to "" then looks like a
    # stall forever. Only judge staleness when there IS a counter to judge.
    if [ -z "$step" ]; then
      prev_step=""; prev_step_t=$now
    elif [ "$step" = "$prev_step" ]; then
      [ $(( (now - prev_step_t) / 60 )) -ge $TRAIN_STALL_MIN ] && { alert "training step stuck at '$step' >= ${TRAIN_STALL_MIN} min"; prev_step_t=$now; }
    else prev_step="$step"; prev_step_t=$now; fi
  fi

  { date -u
    echo "sessions: $(tmux ls 2>/dev/null | cut -d: -f1 | tr '\n' ' ')"
    echo "gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
    for U in GA_DS GS_DA; do D=$EXP_ROOT/data/sdf/$U
      printf "%-6s assembled=%s\n" $U "$([ -f $D/docs.jsonl ] && wc -l < $D/docs.jsonl || echo NO)"; done
    echo "spend: $(grep -o '"cost_usd": [0-9.]*' $EXP_ROOT/data/sdf/usage.json 2>/dev/null)"
    echo "alerts: $(wc -l < $ALERTS)"; tail -3 $ALERTS
  } > $STATUS
  sleep $INTERVAL
done
