#!/usr/bin/env python3
"""Probe RWKV7 vLLM decode logprobs under rollout-like engine settings."""

from __future__ import annotations

import argparse
import gc
import math
from collections import Counter
from dataclasses import dataclass

DEFAULT_MODEL = "/workspace/Weights/RWKV/rwkv7-g1f-1.5b-20260419-ctx8192.pth"
DEFAULT_PROBLEM = "What is 1+1? Answer with a single integer."


def _logprob_dict_get(logprob_dict, token_id: int) -> float | None:
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


@dataclass(frozen=True)
class Case:
    name: str
    enforce_eager: bool
    enable_chunked_prefill: bool | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None

    def kwargs(self) -> dict:
        result = {"enforce_eager": self.enforce_eager}
        for key in ("enable_chunked_prefill", "max_num_seqs", "max_num_batched_tokens"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


def _run_case(args: argparse.Namespace, case: Case, prompt_ids: list[int]) -> None:
    from vllm import LLM, SamplingParams

    print(f"CASE {case.name}: {case.kwargs()}", flush=True)
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
        logprobs_mode=args.logprobs_mode,
        **case.kwargs(),
    )
    params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=1.0,
        top_k=-1,
        logprobs=0,
        seed=args.seed,
    )
    outputs = llm.generate([{"prompt_token_ids": prompt_ids} for _ in range(args.batch_size)], params)

    generate_logprobs: list[float | None] = []
    first_ids = list(outputs[0].outputs[0].token_ids)
    for output in outputs:
        generation = output.outputs[0]
        for token_id, logprob_dict in zip(generation.token_ids, generation.logprobs or [], strict=False):
            generate_logprobs.append(_logprob_dict_get(logprob_dict, token_id))

    _summarize(f"{case.name} generate", generate_logprobs)
    print(f"{case.name} first_ids={first_ids[:32]}", flush=True)

    full_ids = prompt_ids + first_ids
    prefill_outputs = llm.generate(
        [{"prompt_token_ids": full_ids}],
        SamplingParams(max_tokens=1, temperature=0.0, top_p=1.0, top_k=-1, prompt_logprobs=0),
    )
    prefill_logprobs = [
        _logprob_dict_get(prefill_outputs[0].prompt_logprobs[pos], token_id)
        for pos, token_id in enumerate(full_ids)
        if pos >= len(prompt_ids)
    ]
    _summarize(f"{case.name} prefill_first_response", prefill_logprobs)
    print(
        f"{case.name} first_pairs={list(zip(generate_logprobs[:32], prefill_logprobs[:32], strict=False))}",
        flush=True,
    )

    del llm
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--problem", default=DEFAULT_PROBLEM)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logprobs-mode", default="raw_logprobs", choices=["raw_logprobs", "processed_logprobs"])
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["eager", "graph_chunked"],
        choices=["eager", "graph_chunked", "graph_nochunk"],
    )
    args = parser.parse_args()

    from verl.models.rwkv import build_rwkv_tokenizer

    tokenizer = build_rwkv_tokenizer(pickleable=True)
    prompt = f"User: {args.problem}\n\nAssistant: <think"
    prompt_ids = tokenizer.encode(prompt)
    print(f"prompt_ids={prompt_ids[:16]} prompt_len={len(prompt_ids)}", flush=True)
    if not prompt_ids or prompt_ids[0] != 0:
        raise RuntimeError(f"Expected RWKV leading BOS token id 0, got {prompt_ids[:8]}")

    case_map = {
        "eager": Case("eager", enforce_eager=True),
        "graph_chunked": Case(
            "graph_chunked",
            enforce_eager=False,
            enable_chunked_prefill=True,
            max_num_seqs=args.batch_size,
            max_num_batched_tokens=args.max_model_len * args.batch_size,
        ),
        "graph_nochunk": Case(
            "graph_nochunk",
            enforce_eager=False,
            enable_chunked_prefill=False,
            max_num_seqs=args.batch_size,
            max_num_batched_tokens=args.max_model_len * args.batch_size,
        ),
    }
    for case_name in args.cases:
        _run_case(args, case_map[case_name], prompt_ids)


if __name__ == "__main__":
    main()
