#!/bin/bash
# Restart the runner at a task boundary, then self-heal.
#
# A SIGKILL mid-task does NOT release that task's claim (the runner only
# releases on a caught exception), so the task would be silently skipped for
# ever. So: wait for the in-flight task to write its shard, kill, then release
# any claim that has no matching shard.
set -u
R=/mnt/filesystem-u9/gcvl
export EXP_ROOT=$R VM_ID=vmB HF_HOME=/mnt/filesystem-u9/.cache/huggingface
export PATH=$R/.venv/bin:$PATH
log(){ echo "[restart $(date +%H:%M:%S)] $*"; }

n0=$(ls $R/v2/shards/*.parquet 2>/dev/null | wc -l)
log "waiting for the in-flight grid (currently $n0 shards)"
while [ "$(ls $R/v2/shards/*.parquet 2>/dev/null | wc -l)" -le "$n0" ]; do sleep 20; done
log "shard landed"

pkill -f 'drive.sh' 2>/dev/null; pkill -f run_experiments.py 2>/dev/null
sleep 8
log "runner stopped"

released=0
for c in $R/v2/claims/*.claim; do
  [ -e "$c" ] || continue
  id=$(basename "$c" .claim)
  if ! ls $R/v2/shards/${id}__*.parquet >/dev/null 2>&1; then
    rm -f "$c"; released=$((released+1)); log "released orphan claim: $id"
  fi
done
log "released $released orphaned claim(s)"

log "restarting with per-task n (grids 1, stage A 2, E3 13)"
cd $R/repo_v2
nohup $R/.venv/bin/python scripts/run_experiments.py >> $R/v2/queue.log 2>&1 &
log "runner pid $!"
