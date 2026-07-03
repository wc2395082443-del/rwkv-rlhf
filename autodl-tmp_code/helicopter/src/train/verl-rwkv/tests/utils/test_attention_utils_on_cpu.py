# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import builtins

import torch

from verl.utils import attention_utils


def test_torch_padding_fallback_when_flash_attn_cuda_extension_is_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "flash_attn.bert_padding":
            raise ModuleNotFoundError("No module named 'flash_attn_2_cuda'", name="flash_attn_2_cuda")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(attention_utils, "_index_first_axis", None)
    monkeypatch.setattr(attention_utils, "_pad_input", None)
    monkeypatch.setattr(attention_utils, "_rearrange", None)
    monkeypatch.setattr(attention_utils, "_unpad_input", None)

    hidden_states = torch.arange(16).reshape(2, 4, 2)
    attention_mask = torch.tensor([[0, 1, 1, 0], [1, 0, 1, 1]], dtype=torch.int32)

    unpadded, indices, cu_seqlens, max_seqlen, used_seqlens = attention_utils.unpad_input(hidden_states, attention_mask)

    expected_indices = torch.tensor([1, 2, 4, 6, 7])
    expected_unpadded = hidden_states.reshape(-1, 2).index_select(0, expected_indices)

    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(unpadded, expected_unpadded)
    torch.testing.assert_close(cu_seqlens, torch.tensor([0, 2, 5], dtype=torch.int32))
    torch.testing.assert_close(used_seqlens, torch.tensor([2, 3], dtype=torch.int32))
    assert max_seqlen == 3

    restored = attention_utils.pad_input(unpadded, indices, batch=2, seqlen=4)
    expected_restored = torch.zeros_like(hidden_states)
    expected_restored.reshape(-1, 2)[expected_indices] = expected_unpadded

    torch.testing.assert_close(restored, expected_restored)
