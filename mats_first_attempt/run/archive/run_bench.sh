#!/bin/bash
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats
python scripts/00_bench_inference.py --quick 2>&1 | tee $EXP_ROOT/logs/bench.log
echo "=== BENCH EXIT=$? ==="
