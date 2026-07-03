#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/helicopter/src/infer/vllm-rwkv
export PATH=/root/autodl-tmp/venvs/helicopter-torch211/bin:/root/miniconda3/bin:$PATH
export VLLM_TARGET_DEVICE=cuda
export VLLM_USE_PRECOMPILED=0
export VLLM_DISABLE_COOPERATIVE_TOPK=1
export VLLM_RWKV_MINIMAL_BUILD=1
export VLLM_CUTLASS_SRC_DIR=/root/autodl-tmp/helicopter_deps/cutlass-v4.4.2
export TRITON_KERNELS_SRC_DIR=/root/autodl-tmp/helicopter_deps/triton-3.5.1/python/triton_kernels/triton_kernels
export DEEPGEMM_SRC_DIR=/root/autodl-tmp/helicopter_deps/DeepGEMM-891d57b
export FMHA_SM100_SRC_DIR=/root/autodl-tmp/helicopter_deps/MSA-fee7831
export FLASH_MLA_SRC_DIR=/root/autodl-tmp/helicopter_deps/FlashMLA-a6ec2ba
export QUTLASS_SRC_DIR=/root/autodl-tmp/helicopter_deps/qutlass-830d2c4
export VLLM_FLASH_ATTN_SRC_DIR=/root/autodl-tmp/helicopter_deps/vllm-flash-attn-b3964b1
export SETUPTOOLS_SCM_PRETEND_VERSION=0.11.0
export CMAKE_BUILD_TYPE=RelWithDebInfo
export MAX_JOBS=${MAX_JOBS:-8}
export NVCC_THREADS=${NVCC_THREADS:-4}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}
export TMPDIR=/root/autodl-tmp/tmp
export PIP_CACHE_DIR=/root/autodl-tmp/pip_cache
python=/root/autodl-tmp/venvs/helicopter-torch211/bin/python
$python -m pip install --no-deps --no-build-isolation -e . --no-input
$python - <<'PY'
from importlib.metadata import version
print('metadata vllm', version('vllm'))
import vllm
print('vllm module', vllm.__file__, getattr(vllm,'__version__',None))
import vllm._C_stable_libtorch
print('vllm C stable OK')
try:
    import vllm.rwkv7_ops
    print('rwkv7_ops OK')
except Exception as e:
    print('rwkv7_ops import failed', repr(e))
PY
