#!/bin/bash
# smoke -> screen -> main, back to back. Never leave the GPU idle.
# Each stage gates the next: a non-zero exit stops the chain rather than
# running the expensive sweep on a broken prerequisite.
set -u
R=/mnt/filesystem-u9/gcvl
export EXP_ROOT=$R VM_ID=vmB HF_HOME=/mnt/filesystem-u9/.cache/huggingface
export PATH=$R/.venv/bin:$PATH
cd $R/repo_v2
log(){ echo "[chain $(date +%H:%M:%S)] $*"; }

log "waiting for the smoke run"
while pgrep -f "11_steering.py --preset smoke" >/dev/null; do sleep 20; done
log "smoke finished"

for p in screen main; do
  log "starting preset=$p"
  $R/.venv/bin/python scripts/11_steering.py --preset $p --out CAA_$p \
      > $R/v2/steer_$p.log 2>&1
  rc=$?
  log "preset=$p exited rc=$rc"
  [ $rc -ne 0 ] && { log "STOPPING: $p failed"; exit 1; }
done
log "steering chain complete"
