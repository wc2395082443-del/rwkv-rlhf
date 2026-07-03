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

import torch
from tensordict import TensorDict


def test_rwkv_lm_engine_populates_verl_loss_global_batch_fields():
    from verl.workers.config import RWKVLMEngineConfig, RWKVLMOptimizerConfig
    from verl.workers.engine.rwkv_lm import RWKVLMEngine

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.seen_input_shape = None

        def forward(self, input_ids):
            self.seen_input_shape = tuple(input_ids.shape)
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros(batch_size, seq_len, 8, dtype=torch.float32)
            return logits + self.weight

    engine = RWKVLMEngine(
        model_config=None,
        engine_config=RWKVLMEngineConfig(),
        optimizer_config=RWKVLMOptimizerConfig(),
        checkpoint_config=None,
    )
    engine.model = FakeModel()

    data = TensorDict(
        {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "responses": torch.tensor([[2, 3], [5, 6]]),
            "loss_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.float32),
        },
        batch_size=[2],
    )

    def loss_function(model_output, data, dp_group):
        assert data["batch_num_tokens"] == 3
        assert data["dp_size"] == 1
        assert data["global_batch_size"] == 2
        return model_output["log_probs"].sum() * 0 + engine.model.weight, {}

    engine.forward_backward_batch(data, loss_function=loss_function)

    assert engine.model.seen_input_shape == (2, 16)
    assert engine.model.weight.grad is not None


def test_rwkv_lm_engine_honors_static_micro_batch_size():
    from verl.utils import tensordict_utils as tu
    from verl.utils.metric import Metric
    from verl.workers.config import RWKVLMEngineConfig, RWKVLMOptimizerConfig
    from verl.workers.engine.rwkv_lm import RWKVLMEngine

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.seen_input_shapes = []

        def forward(self, input_ids):
            self.seen_input_shapes.append(tuple(input_ids.shape))
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros(batch_size, seq_len, 8, dtype=torch.float32)
            return logits + self.weight

    engine = RWKVLMEngine(
        model_config=None,
        engine_config=RWKVLMEngineConfig(),
        optimizer_config=RWKVLMOptimizerConfig(),
        checkpoint_config=None,
    )
    engine.model = FakeModel()

    data = TensorDict(
        {
            "input_ids": torch.tensor(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [1, 3, 5],
                    [2, 4, 6],
                ]
            ),
            "responses": torch.tensor(
                [
                    [2, 3],
                    [5, 6],
                    [3, 5],
                    [4, 6],
                ]
            ),
            "loss_mask": torch.ones(4, 2, dtype=torch.float32),
        },
        batch_size=[4],
    )
    tu.assign_non_tensor(data, use_dynamic_bsz=False, micro_batch_size_per_gpu=1)

    loss_batch_sizes = []

    def loss_function(model_output, data, dp_group):
        loss_batch_sizes.append(data.batch_size[0])
        assert data["batch_num_tokens"] == 8
        assert data["global_batch_size"] == 4
        return model_output["log_probs"].sum() * 0 + engine.model.weight, {
            "loss_batch_size": data.batch_size[0],
            "metric_batch_size": Metric("mean", data.batch_size[0]),
        }

    output = engine.forward_backward_batch(data, loss_function=loss_function)

    assert engine.model.seen_input_shapes == [(1, 16)] * 4
    assert loss_batch_sizes == [1, 1, 1, 1]
    assert output["loss"] == [1.0, 1.0, 1.0, 1.0]
    assert output["metrics"]["loss_batch_size"] == [1, 1, 1, 1]
    assert isinstance(output["metrics"]["metric_batch_size"], Metric)
    assert output["metrics"]["metric_batch_size"].aggregate() == 1.0
    assert engine.model.weight.grad.item() == 4.0


def test_rwkv_lm_engine_restores_dynamic_micro_batch_order_from_index_lists():
    from verl.workers.config import RWKVLMEngineConfig
    from verl.workers.engine.rwkv_lm import RWKVLMEngine

    engine = RWKVLMEngine(
        model_config=None,
        engine_config=RWKVLMEngineConfig(),
        optimizer_config=None,
        checkpoint_config=None,
    )
    output_lst = [
        {
            "model_output": {
                "log_probs": torch.tensor([[20.0], [30.0]]),
                "labels": ["two", "three"],
            }
        },
        {
            "model_output": {
                "log_probs": torch.tensor([[0.0], [10.0]]),
                "labels": ["zero", "one"],
            }
        },
    ]

    merged = engine._merge_micro_batch_model_outputs(
        output_lst,
        indices=[[2, 3], [0, 1]],
        data=TensorDict({}, batch_size=[4]),
    )

    torch.testing.assert_close(merged["log_probs"], torch.tensor([[0.0], [10.0], [20.0], [30.0]]))
    assert merged["labels"] == ["zero", "one", "two", "three"]


def test_rwkv_lm_engine_averages_gradients_before_optimizer_step(monkeypatch):
    from verl.workers.config import RWKVLMEngineConfig, RWKVLMOptimizerConfig
    from verl.workers.engine.rwkv_lm import RWKVLMEngine, transformer_impl

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(10.0)
    model.weight.grad = torch.full_like(model.weight, 2.0)
    all_reduce_calls = []

    def fake_all_reduce(tensor, op=None, group=None):
        all_reduce_calls.append((tensor, op, group))
        tensor.add_(8.0)

    fake_group = object()
    monkeypatch.setattr(transformer_impl.torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(transformer_impl.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(transformer_impl.torch.distributed, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(transformer_impl.torch.distributed, "all_reduce", fake_all_reduce)

    engine = RWKVLMEngine(
        model_config=None,
        engine_config=RWKVLMEngineConfig(),
        optimizer_config=RWKVLMOptimizerConfig(clip_grad=100.0),
        checkpoint_config=None,
    )
    engine.model = model
    engine.optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    engine.get_data_parallel_group = lambda: fake_group

    grad_norm = engine.optimizer_step()

    assert len(all_reduce_calls) == 1
    assert all_reduce_calls[0][1] is transformer_impl.torch.distributed.ReduceOp.SUM
    assert all_reduce_calls[0][2] is fake_group
    torch.testing.assert_close(model.weight.grad, torch.tensor([[5.0]]))
    torch.testing.assert_close(model.weight, torch.tensor([[5.0]]))
    torch.testing.assert_close(grad_norm, torch.tensor(5.0))
