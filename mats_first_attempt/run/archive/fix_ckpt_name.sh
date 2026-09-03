#!/bin/bash
# The in-flight GA_DS run loaded the pre-fix code and is writing to
# ckpt/sdf_M_base_GS_DA. Once it exits, rename that to its true identity so the
# (fixed) GS_DA run cannot land on top of it.
export EXP_ROOT=/mnt/filesystem-m9/gcvl
while tmux has-session -t sdfreal 2>/dev/null; do sleep 30; done
sleep 20
SRC=$EXP_ROOT/ckpt/sdf_M_base_GS_DA
DST=$EXP_ROOT/ckpt/sdf_M_base_GA_DS
if [ -d "$SRC" ] && [ ! -d "$DST" ]; then
  # checkpoint-20 is debris from the 38-document smoke run, not a dose point.
  rm -rf "$SRC/checkpoint-20"
  mv "$SRC" "$DST"
  echo "$(date -u) renamed $SRC -> $DST (mislabelled by the pre-fix code)" \
    | tee -a $EXP_ROOT/logs/sdf_real.log
  ls -d $DST/checkpoint-* 2>/dev/null | tee -a $EXP_ROOT/logs/sdf_real.log
fi
