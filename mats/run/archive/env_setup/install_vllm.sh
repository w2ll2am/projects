set -x
export PATH=$HOME/.local/bin:$PATH
export VIRTUAL_ENV=/mnt/filesystem-m9/gcvl/.venv
export UV_CACHE_DIR=/mnt/filesystem-m9/.cache/uv
cd /mnt/filesystem-m9/gcvl
echo "=== STEP1 vllm nightly ==="
uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
echo "STEP1_EXIT=$?"
echo "=== STEP2 rest ==="
uv pip install transformers accelerate datasets peft trl pandas numpy scipy statsmodels pyarrow wandb openai huggingface_hub hf_transfer python-dotenv tqdm pyyaml pydantic
echo "STEP2_EXIT=$?"
echo "=== DONE ==="
