#!/usr/bin/env python3
"""Compare vLLM-RWKV rollout logprobs with native rwkv-lm logprobs."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

DEFAULT_MODEL = "/workspace/Weights/RWKV/rwkv7-g1f-1.5b-20260419-ctx8192.pth"
DEFAULT_RWKV_LM = "/workspace/Projects/MachineLearning/rwkv-lm"
DEFAULT_OUT = "analysis/rwkv_logprob_probe.json"
DEFAULT_PROBLEM = "What is 1+1? Answer with a single integer."


def _json_dump(path: str, data: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _json_load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_tokenizer(tokenizer_path: str | None):
    from verl.models.rwkv import build_rwkv_tokenizer

    return build_rwkv_tokenizer(tokenizer_path=tokenizer_path, pickleable=True)


def _strip_leading_bos(token_ids: list[int], bos_token_id: int = 0) -> list[int]:
    if token_ids[:1] == [bos_token_id]:
        return token_ids[1:]
    return token_ids


def _logprob_dict_get(logprob_dict: Any, token_id: int) -> float | None:
    if logprob_dict is None:
        return None
    for key in (token_id, str(token_id)):
        try:
            entry = logprob_dict[key]
            return float(entry.logprob)
        except (KeyError, TypeError):
            pass
    return None


def _summarize_diff(name: str, lhs: list[float], rhs: list[float]) -> dict[str, float]:
    diffs = [abs(a - b) for a, b in zip(lhs, rhs, strict=False) if math.isfinite(a) and math.isfinite(b)]
    if not diffs:
        return {f"{name}/count": 0}
    return {
        f"{name}/count": len(diffs),
        f"{name}/mean_abs": sum(diffs) / len(diffs),
        f"{name}/max_abs": max(diffs),
        f"{name}/min_abs": min(diffs),
    }


def _print_metrics(metrics: dict[str, Any]) -> None:
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]}")


def _checkpoint_state_dict(checkpoint_path: str) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "module"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                return nested
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def _iter_checkpoint_weights(checkpoint_path: str):
    from verl.models.rwkv.weight_mapping import strip_rwkv_lm_deepspeed_prefix

    state = _checkpoint_state_dict(checkpoint_path)
    for name, weight in state.items():
        yield strip_rwkv_lm_deepspeed_prefix(str(name)), weight


def _apply_rwkv7_checkpoint_update_on_worker(worker: Any, checkpoint_path: str) -> dict[str, Any]:
    model_runner = getattr(worker, "model_runner", None)
    model = getattr(model_runner, "model", None)
    if model is None and hasattr(worker, "get_model"):
        model = worker.get_model()
    if model is None:
        raise RuntimeError("Cannot locate vLLM worker model")

    start = getattr(model, "start_weight_update", None)
    finish = getattr(model, "finish_weight_update", None)
    if not callable(start) or not callable(finish):
        raise RuntimeError(f"Model {type(model)!r} does not support RWKV7 transactional weight update")

    raw_before = set(getattr(model, "raw_weight_names", set()) or set())
    model.start_weight_update()
    try:
        loaded = model.load_weights(_iter_checkpoint_weights(checkpoint_path))
        loaded = set(loaded or set())
        model.finish_weight_update()
        model_state = getattr(model_runner, "model_state", None)
        reset_state = getattr(model_state, "reset_after_weight_update", None)
        reset_called = False
        if callable(reset_state):
            reset_state()
            reset_called = True
    except Exception:
        abort = getattr(model, "abort_weight_update", None)
        if callable(abort):
            abort()
        raise

    z = getattr(model, "z", {})
    raw_after = set(getattr(model, "raw_weight_names", set()) or set())
    return {
        "model_type": type(model).__name__,
        "loaded_count": len(loaded),
        "raw_before_count": len(raw_before),
        "raw_after_count": len(raw_after),
        "missing_from_previous_raw": sorted(raw_before - loaded)[:20],
        "unexpected_vs_previous_raw": sorted(loaded - raw_before)[:20],
        "z_count": len(z) if isinstance(z, dict) else None,
        "emb_cache_count": len(getattr(model, "emb_cache", {}) or {}),
        "reset_state_called": reset_called,
    }


def _score_vllm_prefill(llm: Any, full_ids: list[int], prompt_len: int) -> list[float | None]:
    from vllm import SamplingParams

    prefill_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        prompt_logprobs=0,
    )
    prefill_outputs = llm.generate([{"prompt_token_ids": full_ids}], prefill_params)
    prompt_logprobs = prefill_outputs[0].prompt_logprobs
    values = []
    for absolute_pos, token_id in enumerate(full_ids):
        if absolute_pos < prompt_len:
            continue
        values.append(_logprob_dict_get(prompt_logprobs[absolute_pos], token_id))
    return values


def run_vllm(args: argparse.Namespace) -> None:
    os.environ.setdefault("VLLM_RWKV7_WKV_MODE", args.wkv_mode)
    os.environ.setdefault("VLLM_RWKV7_EMB_DEVICE", args.emb_device)
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    from vllm import LLM, SamplingParams

    tokenizer = _build_tokenizer(args.tokenizer_path)
    prompt = f"User: {args.problem}\n\nAssistant: <think"
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids or prompt_ids[0] != 0:
        raise RuntimeError(f"RWKV prompt_ids must start with token id 0, got {prompt_ids[:8]}")

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer_path,
        tokenizer_mode="rwkv",
        trust_remote_code=True,
        load_format="auto",
        max_model_len=args.max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        skip_tokenizer_init=True,
        disable_log_stats=True,
    )

    gen_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=1.0,
        top_k=-1,
        logprobs=0,
        seed=args.seed,
    )
    gen_outputs = llm.generate([{"prompt_token_ids": prompt_ids}], gen_params)
    gen = gen_outputs[0].outputs[0]
    response_ids = list(gen.token_ids)
    response_logprobs = [
        _logprob_dict_get(logprob_dict, token_id)
        for token_id, logprob_dict in zip(response_ids, gen.logprobs or [], strict=False)
    ]

    full_ids = prompt_ids + response_ids
    prefill_params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        prompt_logprobs=0,
    )
    prefill_outputs = llm.generate([{"prompt_token_ids": full_ids}], prefill_params)
    prompt_logprobs = prefill_outputs[0].prompt_logprobs
    prefill_response_logprobs = []
    for absolute_pos, token_id in enumerate(full_ids):
        if absolute_pos < len(prompt_ids):
            continue
        prefill_response_logprobs.append(_logprob_dict_get(prompt_logprobs[absolute_pos], token_id))

    data = {
        "mode": "vllm",
        "model": args.model,
        "rwkv_lm_path": args.rwkv_lm_path,
        "tokenizer_path": args.tokenizer_path,
        "prompt": prompt,
        "problem": args.problem,
        "prompt_ids": prompt_ids,
        "response_ids": response_ids,
        "response_text": tokenizer.decode(response_ids),
        "vllm_generate_logprobs": response_logprobs,
        "vllm_prefill_logprobs": prefill_response_logprobs,
        "env": {
            "VLLM_RWKV7_WKV_MODE": os.environ.get("VLLM_RWKV7_WKV_MODE"),
            "VLLM_RWKV7_EMB_DEVICE": os.environ.get("VLLM_RWKV7_EMB_DEVICE"),
            "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
        },
        "sampling": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": 1.0,
            "top_k": -1,
            "seed": args.seed,
        },
    }
    _json_dump(args.output, data)
    _print_metrics(
        {
            "prompt_len": len(prompt_ids),
            "response_len": len(response_ids),
            "output": args.output,
            "response_text": data["response_text"],
            "first_prompt_ids": prompt_ids[:12],
            "first_response_ids": response_ids[:12],
        }
    )
    del llm
    gc.collect()


def run_vllm_online_update(args: argparse.Namespace) -> None:
    if args.update_model is None:
        raise ValueError("--update-model is required for --mode vllm_online_update")

    os.environ.setdefault("VLLM_RWKV7_WKV_MODE", args.wkv_mode)
    os.environ.setdefault("VLLM_RWKV7_EMB_DEVICE", args.emb_device)
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    from vllm import LLM

    data = _json_load(args.output)
    prompt_ids = [int(x) for x in data["prompt_ids"]]
    response_ids = [int(x) for x in data["response_ids"]]
    full_ids = prompt_ids + response_ids

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer_path,
        tokenizer_mode="rwkv",
        trust_remote_code=True,
        load_format="auto",
        max_model_len=args.max_model_len,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        skip_tokenizer_init=True,
        disable_log_stats=True,
    )

    base_logprobs = _score_vllm_prefill(llm, full_ids, len(prompt_ids))
    update_stats = llm.collective_rpc(
        partial(_apply_rwkv7_checkpoint_update_on_worker, checkpoint_path=args.update_model)
    )
    online_logprobs = _score_vllm_prefill(llm, full_ids, len(prompt_ids))

    fresh_prefill = [float(x) for x in data.get("vllm_prefill_logprobs", []) if x is not None]
    fresh_generate = [float(x) for x in data.get("vllm_generate_logprobs", []) if x is not None]
    online_finite = [float(x) for x in online_logprobs if x is not None]
    base_finite = [float(x) for x in base_logprobs if x is not None]
    metrics: dict[str, Any] = {
        "prompt_len": len(prompt_ids),
        "response_len": len(response_ids),
        "base_model": args.model,
        "update_model": args.update_model,
        "output": args.output,
    }
    metrics.update(_summarize_diff("online_vs_fresh_prefill", online_finite, fresh_prefill))
    metrics.update(_summarize_diff("online_vs_fresh_generate", online_finite, fresh_generate))
    metrics.update(_summarize_diff("base_vs_fresh_prefill", base_finite, fresh_prefill))
    data["vllm_online_update"] = {
        "base_model": args.model,
        "update_model": args.update_model,
        "base_prefill_logprobs": base_logprobs,
        "online_prefill_logprobs": online_logprobs,
        "update_stats": update_stats,
        "metrics": metrics,
    }
    _json_dump(args.output, data)
    _print_metrics(metrics)
    del llm
    gc.collect()


def _native_args(model: str, rwkv_lm_path: str, ctx_len: int) -> tuple[Any, Any, Any]:
    model_config = SimpleNamespace(path=model, rwkv_lm_path=rwkv_lm_path, ctx_len=ctx_len)
    engine_config = SimpleNamespace(
        rwkv_lm_path=rwkv_lm_path,
        ctx_len=ctx_len,
        precision="bf16",
        dtype="bf16",
        grad_cp=0,
        param_offload=False,
        optimizer_offload=False,
        native_env={},
    )
    optimizer_config = SimpleNamespace(
        lr=1e-5,
        weight_decay=0.0,
        lr_warmup_steps=-1,
        clip_grad=1.0,
        adam_eps=1e-18,
        betas=(0.9, 0.99),
    )
    return model_config, engine_config, optimizer_config


def _native_logprobs(
    args: argparse.Namespace, prompt_ids: list[int], response_ids: list[int]
) -> dict[str, list[float]]:
    import torch
    import torch.nn.functional as F

    from verl.workers.engine.rwkv_lm.native_runner import NativeRWKVLMRunner

    model_config, engine_config, optimizer_config = _native_args(args.model, args.rwkv_lm_path, args.max_model_len)
    runner = NativeRWKVLMRunner(
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
    )
    model = runner.build_model()
    model.to(device="cuda", dtype=torch.bfloat16)
    model.eval()

    full_ids = prompt_ids + response_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device="cuda")
    pad_len = (-input_ids.size(-1)) % 16
    if pad_len:
        input_ids = F.pad(input_ids, (0, pad_len), value=0)
    labels = torch.tensor([response_ids], dtype=torch.long, device="cuda")
    offsets = {
        "prompt_minus_2": len(prompt_ids) - 2,
        "prompt_minus_1": len(prompt_ids) - 1,
        "prompt": len(prompt_ids),
        "prompt_plus_1": len(prompt_ids) + 1,
    }
    result: dict[str, list[float]] = {}
    with torch.no_grad():
        logits = model(input_ids).float()
        for name, start in offsets.items():
            if start < 0 or start + len(response_ids) > logits.size(1):
                continue
            selected = logits[:, start : start + len(response_ids), :]
            logprobs = torch.log_softmax(selected, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            result[name] = [float(x) for x in logprobs[0].detach().cpu()]
    return result


def run_actor(args: argparse.Namespace) -> None:
    data = _json_load(args.output)
    prompt_ids = [int(x) for x in data["prompt_ids"]]
    response_ids = [int(x) for x in data["response_ids"]]
    actor_by_offset = _native_logprobs(args, prompt_ids, response_ids)
    data["actor_logprobs_by_offset"] = actor_by_offset
    metrics: dict[str, Any] = {
        "prompt_len": len(prompt_ids),
        "response_len": len(response_ids),
        "output": args.output,
    }
    for offset_name, actor_values in actor_by_offset.items():
        metrics.update(
            _summarize_diff(
                f"generate_vs_actor/{offset_name}",
                [float(x) for x in data["vllm_generate_logprobs"] if x is not None],
                actor_values,
            )
        )
        metrics.update(
            _summarize_diff(
                f"prefill_vs_actor/{offset_name}",
                [float(x) for x in data["vllm_prefill_logprobs"] if x is not None],
                actor_values,
            )
        )
    data["metrics"] = metrics
    _json_dump(args.output, data)
    _print_metrics(metrics)
    best = sorted((v, k) for k, v in metrics.items() if k.endswith("/mean_abs"))
    if best:
        print("best_mean_abs:", best[0][1], best[0][0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vllm", "actor", "vllm_online_update"], required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--update-model", default=None)
    parser.add_argument("--rwkv-lm-path", default=DEFAULT_RWKV_LM)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--problem", default=DEFAULT_PROBLEM)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--wkv-mode", default="fp32io16")
    parser.add_argument("--emb-device", default="gpu")
    args = parser.parse_args()
    if args.mode == "vllm":
        run_vllm(args)
    elif args.mode == "actor":
        run_actor(args)
    else:
        run_vllm_online_update(args)


if __name__ == "__main__":
    main()
