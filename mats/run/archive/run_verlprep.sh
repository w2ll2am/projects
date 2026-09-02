#!/bin/bash
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT
echo "=== installing vLLM into .venv-verl (verl was installed WITHOUT it; rollout.name: vllm needs it) ==="
source $EXP_ROOT/.venv-verl/bin/activate
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly 2>&1 | tail -25
echo "=== versions ==="
python -c "import vllm, torch, verl; print('vllm', vllm.__version__); print('torch', torch.__version__); print('verl', verl.__version__)" 2>&1 | tail -5
echo "=== verl dry-run: is the DAPO reward fn registered for DeepScaleR? ==="
cd $EXP_ROOT/repo/mats
bash scripts/07_train_dapo.sh --dry-run 2>&1 | tail -50
