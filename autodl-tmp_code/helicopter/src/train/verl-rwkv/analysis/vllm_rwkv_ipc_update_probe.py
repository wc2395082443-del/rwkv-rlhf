#!/usr/bin/env python3
"""Probe RWKV7 vLLM logprobs before and after verl CUDA-IPC weight sync."""

from __future__ import annotations

import argparse
import asyncio
import gc
import math
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

DEFAULT_MODEL = "/workspace/Weights/RWKV/rwkv7-g1f-1.5b-20260419-ctx8192.pth"
DEFAULT_PROBLEM = "What is 1+1? Answer with a single integer."


def _state_dict_from_checkpoint(path: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "module"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                return nested
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def _iter_checkpoint_weights(path: str, *, device: str, cast_bf16: bool):
    from verl.models.rwkv.weight_mapping import strip_rwkv_lm_deepspeed_prefix

    state = _state_dict_from_checkpoint(path)
    for name, weight in state.items():
        name = strip_rwkv_lm_deepspeed_prefix(str(name))
        if isinstance(weight, torch.Tensor):
            tensor = weight.detach()
            if cast_bf16 and torch.is_floating_point(tensor):
                tensor = tensor.to(dtype=torch.bfloat16)
            if device != "cpu":
                tensor = tensor.to(device=device, non_blocking=False)
            yield name, tensor
        else:
            yield name, weight


def _logprob_dict_get(logprob_dict: Any, token_id: int) -> float | None:
    if logprob_dict is None:
        return None
    for key in (token_id, str(token_id)):
        try:
            return float(logprob_dict[key].logprob)
        except (KeyError, TypeError):
            pass
    return None


def _summarize(name: str, values: list[float | None]) -> dict[str, Any]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        print(f"{name}: empty", flush=True)
        return {"count": 0}
    rounded = Counter(round(v, 3) for v in finite)
    uniform = sum(1 for v in finite if abs(v + 11.090263366699219) < 1e-5)
    result = {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "min": min(finite),
        "max": max(finite),
        "uniform": uniform,
        "top": rounded.most_common(8),
    }
    print(
        f"{name}: n={result['count']} mean={result['mean']:.6f} "
        f"min={result['min']:.6f} max={result['max']:.6f} "
        f"uniform={result['uniform']} top={result['top']}",
        flush=True,
    )
    return result


def _generate_logprobs(llm: Any, prompt_ids: list[int], *, batch_size: int, max_tokens: int, seed: int):
    from vllm import SamplingParams

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        logprobs=0,
        seed=seed,
    )
    outputs = llm.generate([{"prompt_token_ids": prompt_ids} for _ in range(batch_size)], params)
    values: list[float | None] = []
    first_ids = list(outputs[0].outputs[0].token_ids)
    for output in outputs:
        generation = output.outputs[0]
        for token_id, logprob_dict in zip(generation.token_ids, generation.logprobs or [], strict=False):
            values.append(_logprob_dict_get(logprob_dict, token_id))
    return values, first_ids


def _update_weights_from_ipc(llm: Any, *, use_shm: bool):
    return llm.collective_rpc(
        "update_weights_from_ipc",
        kwargs={"peft_config": None, "base_sync_done": True, "use_shm": use_shm},
    )


async def _send_weights(args: argparse.Namespace, zmq_handle: str) -> None:
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

    sender = BucketedWeightSender(
        zmq_handle=zmq_handle,
        bucket_size_mb=args.bucket_size_mb,
        use_shm=args.use_shm,
    )
    await sender.async_send_weights(
        _iter_checkpoint_weights(args.update_model, device=args.weights_device, cast_bf16=args.cast_bf16)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--update-model", default=DEFAULT_MODEL)
    parser.add_argument("--problem", default=DEFAULT_PROBLEM)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--bucket-size-mb", type=int, default=2048)
    parser.add_argument("--weights-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--cast-bf16", action="store_true")
    parser.add_argument("--use-shm", action="store_true")
    parser.add_argument("--monkey-patch-vocab-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("VERL_RAY_JOB_ID", "ipc-probe")
    os.environ.setdefault("VERL_REPLICA_RANK", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    from vllm import LLM

    from verl.models.rwkv import build_rwkv_tokenizer

    tokenizer = build_rwkv_tokenizer(pickleable=True)
    prompt = f"User: {args.problem}\n\nAssistant: <think"
    prompt_ids = tokenizer.encode(prompt)
    print(f"prompt_ids={prompt_ids[:16]} prompt_len={len(prompt_ids)}", flush=True)
    if not prompt_ids or prompt_ids[0] != 0:
        raise RuntimeError(f"Expected RWKV leading BOS token id 0, got {prompt_ids[:8]}")

    llm = LLM(
        model=args.model,
        tokenizer_mode="rwkv",
        trust_remote_code=True,
        load_format="auto",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        skip_tokenizer_init=True,
        disable_log_stats=True,
        dtype="bfloat16",
        worker_extension_cls="verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension",
    )
    if args.monkey_patch_vocab_size is not None:
        print(f"monkey_patch_vocab_size={args.monkey_patch_vocab_size}", flush=True)
        print(
            llm.collective_rpc(
                "monkey_patch_model",
                kwargs={"vocab_size": args.monkey_patch_vocab_size},
            ),
            flush=True,
        )

    before, before_ids = _generate_logprobs(
        llm, prompt_ids, batch_size=args.batch_size, max_tokens=args.max_tokens, seed=args.seed
    )
    _summarize("before_ipc_update generate", before)
    print(f"before_first_ids={before_ids[:32]}", flush=True)

    zmq_handle = "ipc:///tmp/rl-colocate-zmq-ipc-probe-replica-0-rank-0.sock"

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_update_weights_from_ipc, llm, use_shm=args.use_shm)
        time.sleep(1.0)
        asyncio.run(_send_weights(args, zmq_handle))
        update_result = future.result(timeout=600)
    print(f"update_result={update_result}", flush=True)

    after, after_ids = _generate_logprobs(
        llm, prompt_ids, batch_size=args.batch_size, max_tokens=args.max_tokens, seed=args.seed
    )
    _summarize("after_ipc_update generate", after)
    print(f"after_first_ids={after_ids[:32]}", flush=True)

    del llm
    gc.collect()


if __name__ == "__main__":
    main()
