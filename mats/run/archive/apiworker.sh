#!/bin/bash
# Serial API-side worker. Runs anything that needs the Nebius endpoint rather
# than the GPU, so data work is never blocked behind a serial GPU job.
#
# Queue format: one task per line in $Q.
#   WAITFOR:<path>  <command...>   -> block until <path> exists, then run
#   <command...>                   -> run immediately
#   lines starting with # are comments
#
# Serial ON PURPOSE: the endpoint is the shared resource, and the measured
# failure mode is a retry storm when too many requests queue at once
# (concurrency 64 was SLOWER than 48). Two corpus runs at once would recreate
# exactly that.
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
Q=$EXP_ROOT/queue/api_queue.txt
LOG=$EXP_ROOT/logs/apiq.log
cd $EXP_ROOT/repo/mats
while true; do
  TASK=$(grep -vE '^\s*(#|$)' "$Q" 2>/dev/null | head -1)
  if [ -z "$TASK" ]; then sleep 60; continue; fi
  DEP=""
  case "$TASK" in
    WAITFOR:*) DEP=$(echo "$TASK" | awk '{print $1}' | cut -d: -f2)
               CMD=$(echo "$TASK" | cut -d' ' -f2-) ;;
    *)         CMD="$TASK" ;;
  esac
  if [ -n "$DEP" ] && [ ! -e "$DEP" ]; then
    sleep 60; continue          # dependency unmet: leave it queued, retry
  fi
  # claim the task by removing exactly that line
  grep -vxF "$TASK" "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
  echo "$(date -u) START $CMD" | tee -a $LOG
  bash -lc "cd $EXP_ROOT/repo/mats && source $EXP_ROOT/.venv/bin/activate && $CMD" >> $LOG 2>&1
  echo "$(date -u) EXIT=$? $CMD" | tee -a $LOG
done
