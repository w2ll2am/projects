#!/bin/bash
# Wait for the EU control to finish INSIDE `chain`, then take over before the
# chain starts the DeepScaleR filter (now demoted to last).
S=/mnt/filesystem-m9/gcvl/results/rollouts/M_base_prompted_EU.summary.json
while [ ! -f "$S" ]; do sleep 20; done
sleep 15
tmux kill-session -t chain 2>/dev/null
sleep 5
tmux new -d -s gpuq /tmp/run_gpuq.sh
