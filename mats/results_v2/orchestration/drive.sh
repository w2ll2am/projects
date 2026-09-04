#!/bin/bash
# Chain: wait for the freeze -> verify thresholds actually moved -> preflight ->
# run the queue. Each step gates the next; any failure stops the chain rather
# than running the next stage on a bad prerequisite.
set -u
R=/mnt/filesystem-u9/gcvl
export EXP_ROOT=$R VM_ID=vmB HF_HOME=/mnt/filesystem-u9/.cache/huggingface
export PATH=$R/.venv/bin:$PATH
P=$R/.venv/bin/python
cd $R/repo_v2
log(){ echo "[drive $(date +%H:%M:%S)] $*"; }

log "waiting for threshold re-freeze"
while ! grep -q '^DONE_RC=' $R/v2/freeze.log 2>/dev/null; do sleep 30; done
RC=$(grep '^DONE_RC=' $R/v2/freeze.log | tail -1 | cut -d= -f2)
if [ "$RC" != "0" ]; then log "FREEZE FAILED rc=$RC — stopping"; exit 1; fi
log "freeze finished rc=0"

log "verifying thresholds moved off the v1 values"
$P scripts/vm_preflight.py > $R/v2/preflight.log 2>&1
if [ $? -ne 0 ]; then log "PREFLIGHT FAILED — stopping"; tail -20 $R/v2/preflight.log; exit 1; fi
log "preflight clear"

log "starting the queue (31 tasks)"
$P scripts/run_experiments.py >> $R/v2/queue.log 2>&1
log "QUEUE EXITED rc=$?"
