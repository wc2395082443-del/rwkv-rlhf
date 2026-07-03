# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.machinery
import os
import sys
import types

# In symlink mode (VLLM_FLASH_ATTN_SRC_DIR), cute/ is a symlink to the real
# source tree and its files use `flash_attn.cute.*` imports (not rewritten).
# Register a virtual `flash_attn` package so those imports resolve.
_cute_dir = os.path.join(os.path.dirname(__file__), "cute")
if os.path.islink(_cute_dir) and "flash_attn" not in sys.modules:
    _fa_mod = types.ModuleType("flash_attn")
    _fa_mod.__path__ = [os.path.dirname(os.path.realpath(_cute_dir))]
    _fa_mod.__package__ = "flash_attn"
    _fa_mod.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn", None, is_package=True
    )
    _fa_mod.__spec__.submodule_search_locations = _fa_mod.__path__
    sys.modules["flash_attn"] = _fa_mod

from vllm.vllm_flash_attn.flash_attn_interface import (  # noqa: E402
    FA2_AVAILABLE,
    FA3_AVAILABLE,
    fa_version_unsupported_reason,
    flash_attn_varlen_func,
    get_scheduler_metadata,
    is_fa_version_supported,
)

# Minimal RWKV builds do not compile FlashAttention kernels. RWKV execution
# does not use these kernels, but some generic vLLM modules import this package
# during model inspection. Keep imports non-fatal; actual FA calls still fail in
# flash_attn_interface when the missing kernels are invoked.

__all__ = [
    "fa_version_unsupported_reason",
    "flash_attn_varlen_func",
    "get_scheduler_metadata",
    "is_fa_version_supported",
]
