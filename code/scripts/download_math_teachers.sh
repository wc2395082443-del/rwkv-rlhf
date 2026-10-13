#!/usr/bin/env bash
set -euo pipefail
export HF_HOME=/hy-tmp/hf_home
export TRANSFORMERS_CACHE=/hy-tmp/hf_home
export HF_HUB_ENABLE_HF_TRANSFER=0
/usr/local/bin/python3 - <<'PY'
from huggingface_hub import snapshot_download
for repo, dest in [
 ('Qwen/Qwen2.5-Math-7B-Instruct','/hy-tmp/models/Qwen2.5-Math-7B-Instruct'),
 ('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B','/hy-tmp/models/DeepSeek-R1-Distill-Qwen-7B')]:
 print('DOWNLOAD',repo,flush=True)
 snapshot_download(repo_id=repo, local_dir=dest, local_dir_use_symlinks=False, ignore_patterns=['*.bin','*.h5','*.msgpack','*.onnx'])
 print('DONE',repo,flush=True)
PY