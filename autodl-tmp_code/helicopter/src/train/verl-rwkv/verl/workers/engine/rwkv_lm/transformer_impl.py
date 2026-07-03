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
"""Native rwkv-lm engine wrapper."""

from collections.abc import Callable, Generator
from contextlib import ContextDecorator, contextmanager, nullcontext
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, restore_dynamic_batch
from verl.workers.config import HFModelConfig, RWKVLMEngineConfig, RWKVLMOptimizerConfig
from verl.workers.engine.base import BaseEngine, BaseEngineCtx, EngineRegistry
from verl.workers.engine.utils import prepare_micro_batches

from .batch_bridge import build_verl_loss_model_output, extract_rwkv_lm_forward_batch
from .checkpoint import load_rwkv_lm_checkpoint
from .native_runner import NativeRWKVLMRunner
from .weight_bridge import iter_rwkv_lm_state_dict_weights


class RWKVLMEngine(BaseEngine):
    """Thin Verl engine around native rwkv-lm model and optimizer code."""

    def __init__(
        self,
        model_config: Optional[HFModelConfig],
        engine_config: RWKVLMEngineConfig,
        optimizer_config: RWKVLMOptimizerConfig,
        checkpoint_config: Optional[CheckpointConfig],
        runner_cls: type[NativeRWKVLMRunner] = NativeRWKVLMRunner,
    ):
        super().__init__()
        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        self.mode = None
        self.runner_cls = runner_cls
        self.runner: NativeRWKVLMRunner | None = None
        self.model: Any = None
        self.optimizer: Any = None
        self.lr_scheduler: Any = None
        self.train_callback: Any = None

    @property
    def is_param_offload_enabled(self) -> bool:
        return bool(getattr(self.engine_config, "param_offload", False))

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return bool(getattr(self.engine_config, "optimizer_offload", False))

    def initialize(self):
        self.runner = self.runner_cls(
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
        )
        self.model = self.runner.build_model()
        self.to(
            device="cpu" if self.is_param_offload_enabled else get_device_name(),
            model=True,
            optimizer=False,
            grad=False,
        )
        try:
            trainer_attached = self.model.trainer is not None
        except Exception:
            trainer_attached = False
        if not trainer_attached:
            from types import SimpleNamespace

            trainer = SimpleNamespace(is_global_zero=True, strategy=None)
            try:
                self.model.trainer = trainer
            except Exception:
                self.model._trainer = trainer
        self.train_callback = self.runner.build_train_callback()
        if hasattr(self.model, "configure_optimizers"):
            self.optimizer, self.lr_scheduler = self._normalize_optimizers(self.model.configure_optimizers())
        return self

    def train_mode(self, **kwargs) -> ContextDecorator:
        return self._mode_context("train", **kwargs)

    def eval_mode(self, **kwargs) -> ContextDecorator:
        return self._mode_context("eval", **kwargs)

    def optimizer_zero_grad(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def optimizer_step(self):
        if self.optimizer is None:
            return None
        self._sync_gradients()
        grad_norm = None
        clip_grad = getattr(self.optimizer_config, "clip_grad", None)
        if clip_grad and hasattr(torch.nn.utils, "clip_grad_norm_") and self.model is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
        self.optimizer.step()
        return grad_norm

    def _sync_gradients(self) -> None:
        if self.model is None:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        dp_group = self.get_data_parallel_group()
        if dp_group is None:
            return
        dp_size = torch.distributed.get_world_size(dp_group)
        if dp_size <= 1:
            return
        for param in self.model.parameters():
            if param.grad is None:
                continue
            torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM, group=dp_group)
            param.grad.div_(dp_size)

    def lr_scheduler_step(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        if self.optimizer is None:
            return None
        return [group.get("lr") for group in self.optimizer.param_groups]

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
        if self.model is None:
            raise RuntimeError("RWKVLMEngine.initialize() must be called before forward_backward_batch().")
        self._assign_loss_global_batch_fields(data)
        if self._should_split_micro_batches(data):
            micro_batches, indices = prepare_micro_batches(
                data=data,
                dp_group=self.get_data_parallel_group(),
                same_micro_num_in_dp=True,
            )
            output_lst = [
                self._forward_backward_micro_batch(micro_batch, loss_function, forward_only=forward_only)
                for micro_batch in micro_batches
            ]
            return self._postprocess_micro_batch_outputs(output_lst, indices=indices, data=data)
        return self._forward_backward_micro_batch(data, loss_function, forward_only=forward_only)

    def _forward_backward_micro_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
        data = self._move_data_to_model_device(data)
        forward_batch = extract_rwkv_lm_forward_batch(data)
        original_seq_len = self._input_sequence_length(forward_batch.input_ids)
        model_input_ids = self._build_model_input_ids(forward_batch.input_ids)
        ctx = torch.no_grad() if forward_only else nullcontext()
        with ctx:
            raw_output = self.model(model_input_ids)
            model_output = self._build_model_output(raw_output, forward_batch, data, original_seq_len=original_seq_len)
            if loss_function is None:
                if not forward_only:
                    raise ValueError("loss_function is required unless forward_only=True")
                device = getattr(forward_batch.input_ids, "device", None)
                loss = torch.tensor(0.0, device=device) if device is not None else torch.tensor(0.0)
                metrics = {}
            else:
                loss, metrics = loss_function(
                    model_output=model_output,
                    data=data,
                    dp_group=self.get_data_parallel_group(),
                )
            if not forward_only:
                loss.backward()
        return {
            "model_output": model_output,
            "loss": loss.detach().item() if hasattr(loss, "detach") else loss,
            "metrics": metrics,
        }

    def _should_split_micro_batches(self, data: TensorDict) -> bool:
        use_dynamic_bsz = tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=False)
        if use_dynamic_bsz:
            return True
        micro_batch_size = tu.get_non_tensor_data(data=data, key="micro_batch_size_per_gpu", default=None)
        return micro_batch_size is not None and len(data) > int(micro_batch_size)

    def _postprocess_micro_batch_outputs(self, output_lst: list[dict[str, Any]], *, indices, data: TensorDict) -> dict:
        if len(output_lst) == 1:
            return output_lst[0]

        metrics = {}
        for output in output_lst:
            append_to_dict(metrics, output.get("metrics", {}))

        return {
            "model_output": self._merge_micro_batch_model_outputs(output_lst, indices=indices, data=data),
            "loss": [output["loss"] for output in output_lst],
            "metrics": metrics,
        }

    def _merge_micro_batch_model_outputs(self, output_lst: list[dict[str, Any]], *, indices, data: TensorDict) -> dict:
        model_output = {}
        output_keys = {key for output in output_lst for key in output.get("model_output", {})}
        for key in output_keys:
            values = [output["model_output"][key] for output in output_lst if key in output.get("model_output", {})]
            model_output[key] = self._concat_micro_batch_values(values)
            if indices is not None:
                model_output[key] = self._restore_micro_batch_order(model_output[key], indices)
        return model_output

    def _restore_micro_batch_order(self, value: Any, indices):
        if isinstance(value, torch.Tensor):
            if isinstance(indices, list):
                return restore_dynamic_batch(value, indices)
            return value[indices.argsort().to(value.device)]

        if isinstance(indices, list):
            flat_indices = [idx for batch_indices in indices for idx in batch_indices]
            restore_order = get_reverse_idx(flat_indices)
        else:
            restore_order = indices.argsort().tolist()

        if isinstance(value, list):
            return [value[idx] for idx in restore_order]
        return value

    def _concat_micro_batch_values(self, values: list[Any]) -> Any:
        if not values:
            return values
        if all(isinstance(value, torch.Tensor) for value in values):
            if all(getattr(value, "is_nested", False) for value in values):
                return tu.concat_nested_tensors(values)
            return torch.cat(values, dim=0)
        merged = []
        for value in values:
            if isinstance(value, list):
                merged.extend(value)
            else:
                merged.append(value)
        return merged

    def get_per_tensor_param(self, **_: Any) -> tuple[Generator[tuple[str, torch.Tensor], None, None], Optional[dict]]:
        if self.model is None:
            raise RuntimeError("RWKVLMEngine.initialize() must be called before get_per_tensor_param().")
        return iter_rwkv_lm_state_dict_weights(self.model), None

    def get_data_parallel_size(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        return 1

    def get_data_parallel_rank(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def get_data_parallel_group(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.group.WORLD
        return None

    def _assign_loss_global_batch_fields(self, data: TensorDict) -> None:
        batch_num_tokens = data["loss_mask"].sum()
        if hasattr(batch_num_tokens, "item"):
            batch_num_tokens = batch_num_tokens.item()
        dp_size = self.get_data_parallel_size()
        global_batch_size = data["global_batch_size"] if "global_batch_size" in data else data.batch_size[0] * dp_size
        tu.assign_non_tensor(
            data,
            batch_num_tokens=batch_num_tokens,
            dp_size=dp_size,
            global_batch_size=global_batch_size,
        )

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)
        if model and self.model is not None and hasattr(self.model, "to"):
            target_dtype = self._model_dtype() if device != "cpu" else None
            if target_dtype is None:
                self.model.to(device)
            else:
                self.model.to(device=device, dtype=target_dtype)
        if optimizer and self.optimizer is not None:
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if hasattr(value, "to"):
                        state[key] = value.to(device)
        if device == "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        if self.model is None:
            raise RuntimeError("RWKVLMEngine.initialize() must be called before save_checkpoint().")
        path = self._checkpoint_file(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        del_local_after_load: bool = True,
        **kwargs,
    ) -> None:
        if self.model is None:
            self.initialize()
        self.model.load_state_dict(load_rwkv_lm_checkpoint(self._checkpoint_file(local_path)))

    def is_mp_src_rank_with_outputs(self):
        return True

    @contextmanager
    def _mode_context(self, mode: str, **kwargs):
        if self.model is None:
            raise RuntimeError("RWKVLMEngine.initialize() must be called before switching modes.")
        previous_training = getattr(self.model, "training", None)
        if mode == "train" and hasattr(self.model, "train"):
            self.model.train()
        elif mode == "eval" and hasattr(self.model, "eval"):
            self.model.eval()
        with BaseEngineCtx(self, mode, **kwargs):
            yield self
        if previous_training is not None and hasattr(self.model, "train"):
            self.model.train(previous_training)

    def _normalize_optimizers(self, optimizers: Any) -> tuple[Any, Any]:
        if isinstance(optimizers, tuple) and len(optimizers) == 2:
            optimizer, scheduler = optimizers
            if isinstance(optimizer, (list, tuple)):
                optimizer = optimizer[0] if optimizer else None
            if isinstance(scheduler, (list, tuple)):
                scheduler = scheduler[0] if scheduler else None
            return optimizer, scheduler
        if isinstance(optimizers, (list, tuple)):
            return optimizers[0] if optimizers else None, None
        return optimizers, None

    def _build_model_input_ids(self, input_ids: Any) -> Any:
        device = self._model_device()
        if getattr(input_ids, "is_nested", False):
            pad_token_id = 0
            batch_size = input_ids.offsets().numel() - 1
            max_seq_len = int(input_ids.offsets().diff().max().item())
            input_ids = torch.nested.to_padded_tensor(
                input_ids,
                padding=pad_token_id,
                output_size=(batch_size, max_seq_len),
            )
        if hasattr(input_ids, "to") and device is not None:
            input_ids = input_ids.to(device=device)
        if hasattr(input_ids, "dim") and input_ids.dim() >= 2:
            chunk_len = self._rwkv_chunk_len()
            pad_len = (-input_ids.size(-1)) % chunk_len
            if pad_len:
                input_ids = F.pad(input_ids, (0, pad_len), value=0)
        return input_ids

    def _input_sequence_length(self, input_ids: Any) -> int | None:
        if getattr(input_ids, "is_nested", False):
            return int(input_ids.offsets().diff().max().item())
        if hasattr(input_ids, "size") and getattr(input_ids, "dim", lambda: 0)() >= 2:
            return int(input_ids.size(-1))
        return None

    def _rwkv_chunk_len(self) -> int:
        return 16

    def _move_data_to_model_device(self, data: TensorDict) -> TensorDict:
        device = self._model_device()
        if device is None or not hasattr(data, "to"):
            return data
        return data.to(device=device)

    def _model_device(self) -> torch.device | None:
        if self.model is None or not hasattr(self.model, "parameters"):
            return None
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return None

    def _model_dtype(self) -> torch.dtype | None:
        dtype_name = getattr(self.engine_config, "dtype", None) or getattr(self.engine_config, "precision", None)
        return {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(dtype_name)

    def _build_model_output(
        self, raw_output: Any, forward_batch: Any, data: TensorDict, *, original_seq_len: int | None = None
    ) -> dict[str, Any]:
        model_output = {"logits": raw_output}
        input_ids = forward_batch.input_ids
        if getattr(input_ids, "is_nested", False) and hasattr(raw_output, "dim") and raw_output.dim() == 3:
            from verl.utils import torch_functional as verl_F
            from verl.utils.torch_functional import logprobs_from_logits

            seq_lens = input_ids.offsets().diff().to(device=raw_output.device)
            max_seq_len = raw_output.shape[1]
            valid_positions = torch.arange(max_seq_len, device=raw_output.device).unsqueeze(0)
            valid_mask = valid_positions < seq_lens.unsqueeze(1)
            logits_rmpad = raw_output[valid_mask]

            temperature = tu.get_non_tensor_data(data=data, key="temperature", default=1.0)
            if isinstance(temperature, torch.Tensor):
                temperature = temperature.to(device=logits_rmpad.device, dtype=logits_rmpad.dtype)
                if temperature.numel() == 1:
                    logits_rmpad = logits_rmpad / temperature.clamp(min=1e-8)
                else:
                    temperature_rmpad = torch.repeat_interleave(temperature.reshape(-1), seq_lens)
                    logits_rmpad = logits_rmpad / temperature_rmpad.clamp(min=1e-8).unsqueeze(-1)
            elif temperature != 1.0:
                logits_rmpad = logits_rmpad / max(float(temperature), 1e-8)

            padded_input_ids = torch.nested.to_padded_tensor(
                input_ids,
                padding=0,
                output_size=(input_ids.offsets().numel() - 1, max_seq_len),
            ).to(device=raw_output.device)
            padded_labels = torch.roll(padded_input_ids, shifts=-1, dims=1)
            padded_labels[:, -1] = 0
            labels = padded_labels[valid_mask]
            log_probs = logprobs_from_logits(logits=logits_rmpad, labels=labels)
            offsets = input_ids.offsets().to(device=log_probs.device)
            model_output = build_verl_loss_model_output(
                log_probs=torch.nested.nested_tensor_from_jagged(log_probs, offsets),
            )

            if tu.get_non_tensor_data(data=data, key="calculate_entropy", default=False):
                entropy = verl_F.entropy_from_logits(logits_rmpad)
                model_output["entropy"] = torch.nested.nested_tensor_from_jagged(entropy, offsets)
            if tu.get_non_tensor_data(data=data, key="calculate_sum_pi_squared", default=False):
                sum_pi_squared = verl_F.calculate_sum_pi_squared_from_logits(logits_rmpad)
                model_output["sum_pi_squared"] = torch.nested.nested_tensor_from_jagged(sum_pi_squared, offsets)
            return model_output

        responses = forward_batch.responses
        if responses is not None and hasattr(raw_output, "dim") and raw_output.dim() == 3:
            from verl.utils.torch_functional import logprobs_from_logits, logprobs_from_logits_v2

            if hasattr(responses, "to"):
                responses = responses.to(device=raw_output.device)
            response_length = responses.size(-1)
            sequence_length = original_seq_len if original_seq_len is not None else raw_output.size(1)
            response_start = max(0, sequence_length - response_length - 1)
            response_logits = raw_output[:, response_start : response_start + response_length, :]
            if getattr(response_logits, "device", None) is not None and response_logits.device.type == "cpu":
                log_probs = logprobs_from_logits_v2(logits=response_logits, labels=responses)
            else:
                log_probs = logprobs_from_logits(logits=response_logits, labels=responses)
            model_output = build_verl_loss_model_output(
                log_probs=log_probs,
            )
        return model_output

    def _checkpoint_file(self, local_path: str) -> Path:
        path = Path(local_path)
        if path.suffix:
            return path
        return path / "rwkv_lm.pth"


@EngineRegistry.register(model_type="language_model", backend="rwkv_lm", device=["cuda", "cpu", "npu"])
class RWKVLMEngineWithLMHead(RWKVLMEngine):
    """Template actor/ref engine for native rwkv-lm language-model heads."""
