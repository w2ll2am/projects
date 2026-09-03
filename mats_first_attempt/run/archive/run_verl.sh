#!/bin/bash
set -x
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT
# SEPARATE venv on purpose: verl pins its own deps and must not be allowed to
# downgrade the torch/vllm nightly that the eval pipeline depends on.
[ -d verl ] || git clone -q --depth 1 https://github.com/volcengine/verl
uv venv --python 3.12 $EXP_ROOT/.venv-verl
source $EXP_ROOT/.venv-verl/bin/activate
uv pip install -e ./verl 2>&1 | tail -30
echo "=== resolved versions that matter ==="
uv pip freeze | grep -iE "^(verl|vllm|torch|transformers|ray|flash|tensordict)"
echo "=== import check ==="
python -c "import verl; print('verl OK', getattr(verl,'__version__','?'))"
echo "=== VERL_DONE ==="
