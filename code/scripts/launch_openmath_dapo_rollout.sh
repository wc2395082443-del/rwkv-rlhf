#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/hy-tmp/hf_home
export HF_HUB_ENABLE_HF_TRANSFER=0

MODEL=/hy-tmp/models/OpenMath-Nemotron-1.5B
OUT=/hy-tmp/evals/openmath_nemotron_1p5b_dapo_full_8k_20261005
DATA=/hy-tmp/data/openr1_dapo_math17k_processed/strict_split_seed42/train.jsonl

mkdir -p "$MODEL" "$OUT"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="nvidia/OpenMath-Nemotron-1.5B",
    local_dir="/hy-tmp/models/OpenMath-Nemotron-1.5B",
    ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.onnx", "*.gguf"],
)
PY

/hy-tmp/venvs/openmath-vllm/bin/python /hy-tmp/openmath_dapo_rollout_vllm.py \
  --model "$MODEL" \
  --data "$DATA" \
  --out "$OUT" \
  --chunk 256 \
  --max-num-seqs 128 \
  --max-new 8192 \
  --rwkv-tokenizer /hy-tmp/rwkv-rlhf-gsm8k-project/root_RWKV-LM/RWKV-PEFT-main/RWKV-PEFT-main/json2binidx_tool/tools/rwkv_tokenizer.py \
  --rwkv-vocab /hy-tmp/RWKV-v7/rwkv_vocab_v20230424.txt
