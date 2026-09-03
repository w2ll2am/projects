#!/bin/bash
export EXP_ROOT=/mnt/filesystem-m9/gcvl
# Wait for the GPU to be genuinely FREE, not just for one named session to be
# gone. The previous version killed only `gpuq`, so when the card was held by
# `retrain` it would have launched training on top of it and OOM'd. Gate on the
# device, not on a session name.
gpu_busy() {
  local n
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  [ "${n:-0}" -gt 0 ]
}
started_GS=0
while [ $started_GS -eq 0 ]; do
  if [ -f "$EXP_ROOT/data/sdf/GS_DA/docs.jsonl" ] && ! gpu_busy; then
    echo "$(date -u) GS_DA assembled and GPU free -> SDF training" | tee -a $EXP_ROOT/logs/sdf_real.log
    tmux new -d -s sdfreal "bash -lc 'source $EXP_ROOT/.venv/bin/activate; cd $EXP_ROOT/repo/mats; python scripts/06_train_sdf.py --parent M_base --universes GS_DA --batch-size 2 --grad-accum 1 --max-length 4096 2>&1 | tee -a $EXP_ROOT/logs/sdf_real.log'"
    started_GS=1
  fi
  sleep 60
done
