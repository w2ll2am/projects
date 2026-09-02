#!/bin/bash
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3.5-27B", allow_patterns=["*.safetensors","*.json","*.txt","*.jinja","*.model"])
print("downloaded to", p)
PY
