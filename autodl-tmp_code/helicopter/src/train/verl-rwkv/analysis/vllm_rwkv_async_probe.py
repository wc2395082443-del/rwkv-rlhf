#!/usr/bin/env python3
"""Probe RWKV7 logprobs through vLLM AsyncLLM, matching verl server settings."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import os
import time
from collections import Counter
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


def _summarize(name: str, values: list[float | None]) -> None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        print(f"{name}: empty", flush=True)
        return
    rounded = Counter(round(v, 3) for v in finite)
    uniform = sum(1 for v in finite if abs(v + 11.090263366699219) < 1e-5)
    print(
        f"{name}: n={len(finite)} mean={sum(finite) / len(finite):.6f} "
        f"min={min(finite):.6f} max={max(finite):.6f} uniform={uniform} "
        f"top={rounded.most_common(8)}",
        flush=True,
    )


async def _generate_one(engine: Any, prompt_ids: list[int], sampling_params: Any, request_id: str):
    from vllm.inputs import TokensPrompt

    final = None
    async for output in engine.generate(
        prompt=TokensPrompt(prompt_token_ids=prompt_ids),
        sampling_params=sampling_params,
        request_id=request_id,
    ):
        final = output
    if final is None:
        raise RuntimeError(f"No output for request {request_id}")
    return final


async def _update_weights_from_ipc(
    engine: Any,
    args: argparse.Namespace,
    weights: list[tuple[str, Any]] | None,
    repeat_index: int,
) -> Any:
    from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightSender

    job_id = os.environ["VERL_RAY_JOB_ID"]
    replica_rank = os.environ["VERL_REPLICA_RANK"]
    zmq_handle = f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-0.sock"
    print(f"ipc_update_handle={zmq_handle}", flush=True)
    total_start = time.perf_counter()
    update_task = asyncio.create_task(
        engine.collective_rpc(
            method="update_weights_from_ipc",
            kwargs={"peft_config": None, "base_sync_done": True, "use_shm": args.use_shm},
        )
    )
    await asyncio.sleep(args.sender_start_delay)
    send_start = time.perf_counter()
    sender = BucketedWeightSender(
        zmq_handle=zmq_handle,
        bucket_size_mb=args.bucket_size_mb,
        use_shm=args.use_shm,
    )
    if weights is None:
        weights = _iter_checkpoint_weights(args.update_model, device=args.weights_device, cast_bf16=args.cast_bf16)
    await sender.async_send_weights(weights)
    send_end = time.perf_counter()
    result = await asyncio.wait_for(update_task, timeout=args.update_timeout)
    total_end = time.perf_counter()
    print(f"ipc_update_repeat={repeat_index}", flush=True)
    print(f"ipc_send_seconds={send_end - send_start:.6f}", flush=True)
    print(f"ipc_update_wait_seconds={total_end - send_end:.6f}", flush=True)
    print(f"ipc_update_total_seconds={total_end - total_start:.6f}", flush=True)
    return result


async def _reset_after_weight_update(engine: Any) -> None:
    import vllm

    from verl.workers.rollout.vllm_rollout.utils import reset_vllm_weight_update_caches

    await reset_vllm_weight_update_caches(engine, vllm.__version__)


async def _reset_prefix_cache_after_wake(engine: Any) -> None:
    import vllm
    from packaging import version

    from verl.workers.rollout.vllm_rollout.utils import build_vllm_prefix_cache_reset_kwargs

    await engine.reset_prefix_cache(**build_vllm_prefix_cache_reset_kwargs(version.parse(vllm.__version__)))


async def _generate_batch(
    engine: Any,
    prompt_ids: list[int],
    sampling_params: Any,
    *,
    batch_size: int,
    label: str,
) -> tuple[list[float | None], list[int]]:
    outputs = await asyncio.gather(
        *[_generate_one(engine, prompt_ids, sampling_params, f"{label}-{idx}") for idx in range(batch_size)]
    )
    values: list[float | None] = []
    first_ids = list(outputs[0].outputs[0].token_ids)
    for output in outputs:
        generation = output.outputs[0]
        if generation.logprobs is None:
            raise RuntimeError(f"{label}: expected generation.logprobs when logprobs=0")
        if len(generation.token_ids) != len(generation.logprobs):
            raise RuntimeError(
                f"{label}: token/logprob length mismatch: "
                f"{len(generation.token_ids)} token ids vs {len(generation.logprobs)} logprob rows"
            )
        for token_id, logprob_dict in zip(generation.token_ids, generation.logprobs, strict=True):
            value = _logprob_dict_get(logprob_dict, token_id)
            if value is None:
                keys = list(logprob_dict.keys())[:8] if logprob_dict is not None else None
                raise RuntimeError(f"{label}: sampled token {token_id} missing from logprob dict; first keys={keys}")
            values.append(value)
    return values, first_ids


async def _run_async(args: argparse.Namespace) -> None:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.async_llm import AsyncLLM

    from verl.models.rwkv import build_rwkv_tokenizer
    from verl.workers.rollout.vllm_rollout.utils import build_cli_args_from_config

    tokenizer = build_rwkv_tokenizer(pickleable=True)
    prompt = f"User: {args.problem}\n\nAssistant: <think"
    prompt_ids = tokenizer.encode(prompt)
    print(f"prompt_ids={prompt_ids[:16]} prompt_len={len(prompt_ids)}", flush=True)
    if not prompt_ids or prompt_ids[0] != 0:
        raise RuntimeError(f"Expected RWKV leading BOS token id 0, got {prompt_ids[:8]}")

    os.environ.setdefault("VERL_RAY_JOB_ID", args.ray_job_id)
    os.environ.setdefault("VERL_REPLICA_RANK", str(args.replica_rank))
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    override_generation_config = {
        "temperature": 1.0,
        "top_k": -1,
        "top_p": 1,
        "repetition_penalty": 1.0,
        "max_new_tokens": args.max_tokens,
    }
    engine_config = {
        "dtype": "bfloat16",
        "load_format": "auto",
        "skip_tokenizer_init": False,
        "distributed_executor_backend": "mp",
        "worker_extension_cls": "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension",
        "trust_remote_code": False,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.batch_size,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "max_num_batched_tokens": args.max_model_len * args.batch_size,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_sleep_mode": args.enable_sleep_mode,
        "logprobs_mode": args.logprobs_mode,
        "enforce_eager": args.enforce_eager,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "disable_log_stats": True,
        "tensor_parallel_size": 1,
        "seed": args.seed,
        "override_generation_config": json.dumps(override_generation_config),
        "hf_overrides": {},
        "scheduling_policy": "fcfs",
        "compilation_config": json.dumps({"cudagraph_mode": "FULL_AND_PIECEWISE"}),
    }
    if args.cli_config:
        import vllm.entrypoints.cli.serve

        try:
            from vllm.utils.argparse_utils import FlexibleArgumentParser
        except ImportError:
            from vllm.utils import FlexibleArgumentParser

        server_args = ["serve", args.model] + build_cli_args_from_config(engine_config)
        parser = FlexibleArgumentParser(description="AsyncLLM probe")
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in (vllm.entrypoints.cli.serve,):
            for cmd in cmd_module.cmd_init():
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        parsed_args = parser.parse_args(args=server_args)
        parsed_args.model = parsed_args.model_tag
        if parsed_args.subparser in cmds:
            cmds[parsed_args.subparser].validate(parsed_args)
        engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        engine = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            disable_log_stats=engine_args.disable_log_stats,
            enable_log_requests=engine_args.enable_log_requests,
        )
    else:
        engine_config["override_generation_config"] = override_generation_config
        engine_config["compilation_config"] = {"cudagraph_mode": "FULL_AND_PIECEWISE"}
        engine_args = AsyncEngineArgs(model=args.model, **engine_config)
        engine = AsyncLLM.from_engine_args(
            engine_args,
            usage_context=UsageContext.OPENAI_API_SERVER,
        )
    await engine.reset_mm_cache()
    await engine.collective_rpc(method="monkey_patch_model", kwargs={"vocab_size": len(tokenizer)})

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        logprobs=0,
        seed=args.seed,
    )
    before, before_ids = await _generate_batch(
        engine, prompt_ids, sampling_params, batch_size=args.batch_size, label="async-before"
    )
    _summarize("async_before_ipc_update", before)
    print(f"before_first_ids={before_ids[:32]}", flush=True)

    if args.ipc_update:
        update_weights = None
        if args.preload_update_weights:
            preload_start = time.perf_counter()
            update_weights = list(
                _iter_checkpoint_weights(
                    args.update_model,
                    device=args.weights_device,
                    cast_bf16=args.cast_bf16,
                )
            )
            preload_end = time.perf_counter()
            print(f"preload_update_weight_count={len(update_weights)}", flush=True)
            print(
                "preload_update_weight_bytes="
                f"{sum(tensor.nbytes for _, tensor in update_weights if isinstance(tensor, torch.Tensor))}",
                flush=True,
            )
            print(f"preload_update_weights_seconds={preload_end - preload_start:.6f}", flush=True)

        if args.sleep_cycle:
            print(f"sleep_level={args.sleep_level}", flush=True)
            await engine.sleep(level=args.sleep_level)
            print("wake_up_weights", flush=True)
            await engine.wake_up(tags=["weights"])
            await _reset_prefix_cache_after_wake(engine)
        for repeat_index in range(args.update_repeats):
            update_result = await _update_weights_from_ipc(engine, args, update_weights, repeat_index)
            print(f"ipc_update_result={update_result}", flush=True)
        if args.sleep_cycle:
            print("reset_after_weight_update", flush=True)
            await _reset_after_weight_update(engine)
            print("wake_up_kv_cache", flush=True)
            await engine.wake_up(tags=["kv_cache"])
            await _reset_prefix_cache_after_wake(engine)
        after, after_ids = await _generate_batch(
            engine, prompt_ids, sampling_params, batch_size=args.batch_size, label="async-after"
        )
        _summarize("async_after_ipc_update", after)
        print(f"after_first_ids={after_ids[:32]}", flush=True)

    engine.shutdown()
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--update-model", default=DEFAULT_MODEL)
    parser.add_argument("--problem", default=DEFAULT_PROBLEM)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logprobs-mode", default="processed_logprobs")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-sleep-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cli-config", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ipc-update", action="store_true")
    parser.add_argument("--update-repeats", type=int, default=1)
    parser.add_argument("--preload-update-weights", action="store_true")
    parser.add_argument("--sleep-cycle", action="store_true")
    parser.add_argument("--sleep-level", type=int, default=2)
    parser.add_argument("--bucket-size-mb", type=int, default=2048)
    parser.add_argument("--weights-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--cast-bf16", action="store_true")
    parser.add_argument("--use-shm", action="store_true")
    parser.add_argument("--ray-job-id", default="async-ipc-probe")
    parser.add_argument("--replica-rank", type=int, default=0)
    parser.add_argument("--sender-start-delay", type=float, default=1.0)
    parser.add_argument("--update-timeout", type=float, default=600.0)
    args = parser.parse_args()
    asyncio.run(_run_async(args))


if __name__ == "__main__":
    main()
