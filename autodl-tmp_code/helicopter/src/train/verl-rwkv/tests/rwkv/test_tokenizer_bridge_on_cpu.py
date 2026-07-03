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

import importlib.util
import pickle
from pathlib import Path

from omegaconf import OmegaConf


def _load_tokenizer_module():
    path = Path("verl/models/rwkv/tokenizer.py")
    spec = importlib.util.spec_from_file_location("rwkv_tokenizer_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "|".join(f"{message['role']}:{message['content']}" for message in messages)
        if add_generation_prompt:
            text += "|assistant:"
        if tokenize:
            return [ord(char) for char in text]
        return text


def test_build_rwkv_tokenizer_instantiates_vllm_tokenizer_for_custom_vocab():
    tokenizer_module = _load_tokenizer_module()

    tokenizer = tokenizer_module.build_rwkv_tokenizer(
        "/tmp/rwkv_vocab.txt",
        tokenizer_cls=FakeTokenizer,
        user_role="Human",
    )

    assert isinstance(tokenizer, FakeTokenizer)
    assert tokenizer.args == ("/tmp/rwkv_vocab.txt",)
    assert tokenizer.kwargs == {"user_role": "Human"}


def test_build_rwkv_tokenizer_can_return_pickleable_proxy():
    import verl.models.rwkv.tokenizer as tokenizer_module

    class PickleTokenizer(FakeTokenizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.root = []
            self.root.append(self)

        def encode(self, text):
            return [len(text)]

    tokenizer = tokenizer_module.build_rwkv_tokenizer(
        tokenizer_cls=PickleTokenizer,
        pickleable=True,
    )

    assert tokenizer.encode("abc") == [3]
    pickle.dumps(tokenizer)


def test_pickleable_rwkv_tokenizer_supports_padding_for_agent_loop():
    import verl.models.rwkv.tokenizer as tokenizer_module

    class PadTokenizer(FakeTokenizer):
        eos_token_id = 0

    tokenizer = tokenizer_module.build_rwkv_tokenizer(
        tokenizer_cls=PadTokenizer,
        pickleable=True,
    )
    tokenizer.padding_side = "left"

    output = tokenizer.pad(
        {"input_ids": [[1, 2], [3]]},
        padding="max_length",
        max_length=3,
        return_attention_mask=True,
    )

    assert output == {
        "input_ids": [[0, 1, 2], [0, 0, 3]],
        "attention_mask": [[0, 1, 1], [0, 0, 1]],
    }


def test_pickleable_rwkv_tokenizer_accepts_hf_chat_template_kwargs():
    import verl.models.rwkv.tokenizer as tokenizer_module

    tokenizer = tokenizer_module.build_rwkv_tokenizer(
        tokenizer_cls=FakeTokenizer,
        pickleable=True,
    )

    output = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=True,
        add_generation_prompt=True,
        tools=None,
        return_dict=True,
    )

    assert output["input_ids"] == [ord(char) for char in "user:hi|assistant:"]
    assert output["attention_mask"] == [1] * len(output["input_ids"])


def test_pickleable_rwkv_tokenizer_falls_back_to_plain_text_without_chat_template():
    import verl.models.rwkv.tokenizer as tokenizer_module

    class PlainTokenizer:
        eos_token_id = 0
        vocab_size = 256

        def apply_chat_template(self, *args, **kwargs):
            raise NotImplementedError("no chat template")

        def encode(self, text):
            return [ord(char) for char in text]

        def __len__(self):
            return self.vocab_size

    tokenizer = tokenizer_module.build_rwkv_tokenizer(
        tokenizer_cls=PlainTokenizer,
        pickleable=True,
    )

    output = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": [{"type": "text", "text": "ignore"}]},
            {"role": "user", "content": "solve"},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )

    expected_prompt = "User: solve\n\nAssistant: <think"
    assert output["input_ids"] == [ord(char) for char in expected_prompt]
    assert output["attention_mask"] == [1] * len(output["input_ids"])


def test_pickleable_rwkv_tokenizer_preserves_native_bos_policy():
    import verl.models.rwkv.tokenizer as tokenizer_module

    class BosTokenizer:
        eos_token_id = 0
        vocab_size = 256

        def encode(self, text):
            return [0, *[ord(char) for char in text]]

        def __len__(self):
            return self.vocab_size

    tokenizer = tokenizer_module.build_rwkv_tokenizer(
        tokenizer_cls=BosTokenizer,
        pickleable=True,
    )

    assert tokenizer.encode("abc") == [0, 97, 98, 99]


def test_ppo_tokenizer_builder_uses_native_rwkv_tokenizer(monkeypatch):
    import verl.models.rwkv as rwkv_module
    from verl.trainer.tokenizer import build_ppo_tokenizer_and_processor

    calls = []
    native_tokenizer = object()

    def fake_build_rwkv_tokenizer(**kwargs):
        calls.append(kwargs)
        return native_tokenizer

    monkeypatch.setattr(rwkv_module, "build_rwkv_tokenizer", fake_build_rwkv_tokenizer)
    cfg = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "_target_": "verl.models.rwkv.RWKVNativeModelConfig",
                    "path": "/models/rwkv.pth",
                    "tokenizer_path": "/models/rwkv_vocab.txt",
                }
            },
            "data": {},
        }
    )

    tokenizer, processor = build_ppo_tokenizer_and_processor(cfg)

    assert tokenizer is native_tokenizer
    assert processor is None
    assert calls == [
        {
            "tokenizer_path": "/models/rwkv_vocab.txt",
            "pickleable": True,
        }
    ]


def test_rwkv_native_model_config_builds_pickleable_tokenizer(monkeypatch):
    import verl.models.rwkv.config as config_module

    calls = []
    native_tokenizer = object()

    def fake_build_rwkv_tokenizer(**kwargs):
        calls.append(kwargs)
        return native_tokenizer

    monkeypatch.setattr(config_module, "build_rwkv_tokenizer", fake_build_rwkv_tokenizer)

    config = config_module.RWKVNativeModelConfig(
        path="/models/rwkv.pth",
        tokenizer_path="/models/rwkv_vocab.txt",
    )

    assert config.tokenizer is native_tokenizer
    assert config.processor is None
    assert calls == [
        {
            "tokenizer_path": "/models/rwkv_vocab.txt",
            "pickleable": True,
        }
    ]


def test_reward_loop_uses_native_rwkv_tokenizer(monkeypatch):
    import verl.experimental.reward_loop.reward_loop as reward_loop_module
    import verl.models.rwkv as rwkv_module

    calls = []
    native_tokenizer = object()
    reward_manager = object()

    def fake_build_rwkv_tokenizer(**kwargs):
        calls.append(kwargs)
        return native_tokenizer

    def fake_hf_tokenizer(*args, **kwargs):
        raise AssertionError("RWKV reward loop must not pass a .pth checkpoint to HF tokenizer")

    monkeypatch.setattr(rwkv_module, "build_rwkv_tokenizer", fake_build_rwkv_tokenizer)
    monkeypatch.setattr(reward_loop_module, "hf_tokenizer", fake_hf_tokenizer)
    monkeypatch.setattr(
        reward_loop_module,
        "load_reward_manager",
        lambda config, tokenizer, **kwargs: (tokenizer, kwargs, reward_manager),
    )

    cfg = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "_target_": "verl.models.rwkv.RWKVNativeModelConfig",
                    "path": "/models/rwkv.pth",
                    "tokenizer_path": "/models/rwkv_vocab.txt",
                }
            },
            "reward": {
                "reward_model": {
                    "enable": False,
                    "model_path": None,
                },
            },
        }
    )

    worker = reward_loop_module.RewardLoopWorker.__new__(reward_loop_module.RewardLoopWorker)
    worker.config = cfg
    worker.reward_router_address = None
    worker._init_reward_fn()

    assert worker.input_tokenizer is native_tokenizer
    assert worker.reward_model_tokenizer is None
    assert worker.reward_manager == (
        native_tokenizer,
        {"reward_router_address": None, "reward_model_tokenizer": None},
        reward_manager,
    )
    assert calls == [
        {
            "tokenizer_path": "/models/rwkv_vocab.txt",
            "pickleable": True,
        }
    ]
