#!/usr/bin/env bash
set -euo pipefail
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/hy-tmp/hf_home
export HF_HUB_ENABLE_HF_TRANSFER=0
LOG=/hy-tmp/download_math_teachers_mirror.status
/usr/local/bin/python3 - <<'PY'
from huggingface_hub import snapshot_download
models=[
 ('Qwen/Qwen2.5-Math-7B-Instruct','/hy-tmp/models/Qwen2.5-Math-7B-Instruct'),
 ('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B','/hy-tmp/models/DeepSeek-R1-Distill-Qwen-7B')]
for repo,dest in models:
 print('DOWNLOAD',repo,flush=True)
 snapshot_download(repo_id=repo,local_dir=dest,ignore_patterns=['*.bin','*.h5','*.msgpack','*.onnx'])
 print('DONE',repo,flush=True)
PY