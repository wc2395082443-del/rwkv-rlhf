#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from math_verify import parse, verify
from rwkv.utils import PIPELINE

from reward import extract_response_answer


THIS_DIR = Path(__file__).resolve().parent
EVAL_WKV = "fp32io16"
EVAL_EMB = "cpu"
EVAL_BATCHED_RKV = "off"
EVAL_CMIX_SPARSE = "no-fc"
EVAL_LOWRANK_WEIGHT = "both"
EVAL_ORIG_LINEAR_GROUPS = "att_c2c,ffn_key,head"


@dataclass(frozen=True)
class Task:
    index: int
    problem: str
    answer: str
    subject: str = ""
    level: str = ""
    unique_id: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--albatross-dir", default=str(THIS_DIR))
    parser.add_argument("--model", default="/dev/shm/rwkv7-g1f-1.5b-20260419-ctx8192.pth")
    parser.add_argument("--dataset", default=str(THIS_DIR / "dataset" / "MATH500.jsonl"))
    parser.add_argument("--out-dir", default=str(THIS_DIR / "math500_runs"))
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--rollout", type=int, default=4)
    parser.add_argument("--bsz", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=1500)
    parser.add_argument("--eval-tail-max-tokens", type=int, default=64)
    parser.add_argument("--eval-tail-prefix", default=" so the answer is : ")
    parser.add_argument("--ctx-limit", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.28)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-style",
        choices=("fake_think", "thinking_cot", "plain"),
        default="fake_think",
    )
    parser.add_argument("--torch-extensions-dir", default="")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--verify-workers", type=int, default=8)
    args = parser.parse_args()
    args.wkv = EVAL_WKV
    args.emb = EVAL_EMB
    args.batched_rkv = EVAL_BATCHED_RKV
    args.cmix_sparse = EVAL_CMIX_SPARSE
    args.lowrank_weight = EVAL_LOWRANK_WEIGHT
    args.orig_linear_groups = EVAL_ORIG_LINEAR_GROUPS
    return args

def load_tasks(dataset: str) -> list[Task]:
    rows = []
    with open(dataset, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                Task(
                    index=index,
                    problem=str(item["problem"]),
                    answer=str(item["answer"]),
                    subject=str(item.get("subject", "")),
                    level=str(item.get("level", "")),
                    unique_id=str(item.get("unique_id", index)),
                )
            )
    return rows


@torch.jit.script
def sample_logits(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
    if temperature <= 0.0 or top_p <= 0.0:
        return torch.argmax(logits, dim=-1)

    scaled = logits.float()
    if temperature != 1.0:
        scaled = scaled / float(temperature)

    if top_k > 0:
        k = min(int(top_k), logits.size(-1))
        vals, ids = torch.topk(scaled, k=k, dim=-1, sorted=True)
        probs = torch.softmax(vals, dim=-1)
    else:
        ids = torch.arange(logits.size(-1), device=logits.device).view(1, -1).expand(logits.size(0), -1)
        probs = torch.softmax(scaled, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_order = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative_probs > float(top_p)
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        remove = remove.scatter(1, sorted_order, remove)
        probs = probs.masked_fill(remove, 0.0)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-10)

    picked = torch.multinomial(probs, num_samples=1)
    return ids.gather(1, picked).view(-1)


def copy_single_state_into_batch(batch_state: list[Any], row: int, single_state: list[Any]) -> None:
    batch_state[0][:, :, row : row + 1, :].copy_(single_state[0])
    batch_state[1][:, row : row + 1, :, :, :].copy_(single_state[1])
    batch_state[2][row : row + 1].copy_(single_state[2])


def build_prefill_cache(
    args: argparse.Namespace,
    tasks: list[Task],
    task_indices: list[int],
    model: Any,
    tokenizer: PIPELINE,
    token_device: str,
) -> dict[int, tuple[list[Any], torch.Tensor, int]]:
    cache = {}
    t0 = time.perf_counter()
    for done, task_idx in enumerate(task_indices, 1):
        problem = tasks[task_idx].problem.strip().replace("\r\n", "\n")
        if args.prompt_style == "fake_think":
            prompt = f"User: {problem}\n\nAssistant: <think></think"
        elif args.prompt_style == "thinking_cot":
            prompt = (
                f"User: {problem}\n"
                "Please reason step by step. Close the reasoning with </think>, then output exactly "
                "<answer>final answer</answer> and immediately end the response."
                "\nAssistant: <think>\n"
            )
        elif args.prompt_style == "plain":
            prompt = f"User: {problem}\n\nAssistant:"
        else:
            raise ValueError(f"unknown prompt style: {args.prompt_style}")
        ids = [0] + tokenizer.encode(prompt)
        if len(ids) + args.max_new_tokens > args.ctx_limit:
            ids = ids[-max(1, args.ctx_limit - args.max_new_tokens) :]
        state = model.zero_state(1)
        tokens = torch.tensor(ids, dtype=torch.long, device=token_device)
        logits = model.forward(tokens, state).view(-1)
        cache[task_idx] = ([x.clone() for x in state], logits.clone(), len(ids))
        if args.progress_every > 0 and done % args.progress_every == 0:
            print(
                f"worker={args.worker_rank} prefill_cache {done}/{len(task_indices)} "
                f"elapsed_s={time.perf_counter() - t0:.3f}",
                flush=True,
            )
    torch.cuda.synchronize()
    print(
        f"worker={args.worker_rank} prefill_cache done prompts={len(task_indices)} "
        f"elapsed_s={time.perf_counter() - t0:.3f}",
        flush=True,
    )
    return cache


def generate_dynamic_rollouts(
    args: argparse.Namespace,
    tasks: list[Task],
    work: list[tuple[int, int]],
    model: Any,
    tokenizer: PIPELINE,
    token_device: str,
    prefill_cache: dict[int, tuple[list[Any], torch.Tensor, int]],
) -> list[dict[str, Any]]:
    if not work:
        return []

    B = min(args.bsz, len(work))
    tail_max_tokens = max(0, int(args.eval_tail_max_tokens))
    batch_state = model.zero_state(B)
    pending_pos = 0
    slot_work: list[tuple[int, int] | None] = [None] * B
    prompt_lengths = [0] * B
    generated: list[list[int]] = [[] for _ in range(B)]
    active = [False] * B
    phases = ["idle"] * B
    forced_queues: list[list[int]] = [[] for _ in range(B)]
    token_counts = [0] * B
    main_generated_counts = [0] * B
    tail_generated_counts = [0] * B
    tail_forced_counts = [0] * B
    main_truncated = [False] * B
    out_texts = ["" for _ in range(B)]
    out_last = [0 for _ in range(B)]
    next_cpu = [0 for _ in range(B)]
    token_tensor = torch.empty((B, 1), dtype=torch.long, device=token_device)
    rows: list[dict[str, Any]] = []
    t_decode = time.perf_counter()
    decoded_token_events = 0
    forward_steps = 0
    last_progress_t = t_decode
    last_progress_tokens = 0

    def flush_text(row: int) -> None:
        if out_last[row] >= len(generated[row]):
            return
        pending = tokenizer.decode(generated[row][out_last[row] :])
        if "\ufffd" not in pending:
            out_texts[row] += pending
            out_last[row] = len(generated[row])

    def finish_row(row: int, stop_reason: str) -> None:
        work_item = slot_work[row]
        assert work_item is not None
        task_idx, sample_id = work_item
        flush_text(row)
        completion = out_texts[row].split("\nUser:", 1)[0]
        if completion.startswith(">"):
            completion = completion[1:]
        task = tasks[task_idx]
        tail_applied = bool(main_truncated[row] and (tail_forced_counts[row] or tail_generated_counts[row]))
        rows.append(
            {
                "task_index": task.index,
                "local_task_index": task_idx,
                "sample_id": sample_id,
                "worker_rank": args.worker_rank,
                "problem": task.problem,
                "answer": task.answer,
                "subject": task.subject,
                "level": task.level,
                "unique_id": task.unique_id,
                "prompt_tokens": prompt_lengths[row],
                "generated_tokens": len(generated[row]),
                "main_generated_tokens": main_generated_counts[row],
                "eval_tail_forced_tokens": tail_forced_counts[row],
                "eval_tail_generated_tokens": tail_generated_counts[row],
                "eval_tail_applied": tail_applied,
                "tokens_including_eod": token_counts[row],
                "tokens_including_stop": token_counts[row],
                "ended_eod": stop_reason in ("eod", "eval_tail_eod"),
                "ended_user_stop": stop_reason in ("user_stop", "eval_tail_user_stop"),
                "stop_reason": stop_reason,
                "truncated": bool(main_truncated[row]),
                "completion": completion.strip(),
            }
        )
        slot_work[row] = None
        prompt_lengths[row] = 0
        generated[row] = []
        active[row] = False
        phases[row] = "idle"
        forced_queues[row] = []
        token_counts[row] = 0
        main_generated_counts[row] = 0
        tail_generated_counts[row] = 0
        tail_forced_counts[row] = 0
        main_truncated[row] = False
        out_texts[row] = ""
        out_last[row] = 0
        next_cpu[row] = 0

    def refill_rows(refill: list[int]) -> list[int]:
        nonlocal pending_pos
        assigned = []
        init_logits = []
        for row in refill:
            if pending_pos >= len(work):
                break
            task_idx, sample_id = work[pending_pos]
            pending_pos += 1
            state, logits, prompt_len = prefill_cache[task_idx]
            copy_single_state_into_batch(batch_state, row, state)
            slot_work[row] = (task_idx, sample_id)
            prompt_lengths[row] = prompt_len
            generated[row] = []
            active[row] = True
            phases[row] = "main"
            forced_queues[row] = []
            token_counts[row] = 0
            main_generated_counts[row] = 0
            tail_generated_counts[row] = 0
            tail_forced_counts[row] = 0
            main_truncated[row] = False
            out_texts[row] = ""
            out_last[row] = 0
            assigned.append(row)
            init_logits.append(logits)
        if assigned:
            sampled = sample_logits(torch.stack(init_logits, dim=0), args.temperature, args.top_p, args.top_k)
            for row, token in zip(assigned, sampled.detach().cpu().tolist()):
                next_cpu[row] = int(token)
        return assigned

    def begin_eval_tail(row: int) -> None:
        main_truncated[row] = True
        flush_text(row)
        prefix = str(args.eval_tail_prefix)
        if "</think>" not in out_texts[row]:
            prefix = "\n</think>\n" + prefix
        forced_queues[row] = [int(token) for token in tokenizer.encode(prefix)]
        phases[row] = "force_tail" if forced_queues[row] else "tail"

    def process_next_token(row: int) -> bool:
        nonlocal decoded_token_events
        phase = phases[row]
        token = int(next_cpu[row])
        token_counts[row] += 1
        decoded_token_events += 1

        if phase in ("main", "tail") and token == 0:
            finish_row(row, "eod" if phase == "main" else "eval_tail_eod")
            return False

        generated[row].append(token)
        if phase == "main":
            main_generated_counts[row] += 1
        elif phase == "tail":
            tail_generated_counts[row] += 1
        elif phase == "force_tail":
            tail_forced_counts[row] += 1
        else:
            raise RuntimeError(f"unexpected generation phase: {phase}")

        flush_text(row)
        if "\nUser:" in out_texts[row]:
            finish_row(row, "user_stop" if phase == "main" else "eval_tail_user_stop")
            return False

        if phase == "main" and main_generated_counts[row] >= args.max_new_tokens:
            if tail_max_tokens <= 0:
                main_truncated[row] = True
                finish_row(row, "max_tokens")
                return False
            begin_eval_tail(row)
            # The final main token must update recurrent state before the forced tail begins.
            return True

        if phase == "force_tail" and not forced_queues[row]:
            phases[row] = "tail"

        if phase == "tail" and tail_generated_counts[row] >= tail_max_tokens:
            finish_row(row, "eval_tail_max_tokens")
            return False
        return True

    def choose_next_tokens(logits: torch.Tensor, forward_rows: list[int]) -> None:
        sampled = sample_logits(logits, args.temperature, args.top_p, args.top_k).detach().cpu().tolist()
        for row in forward_rows:
            if phases[row] == "force_tail" and forced_queues[row]:
                next_cpu[row] = forced_queues[row].pop(0)
            else:
                next_cpu[row] = int(sampled[row])

    refill_rows(list(range(B)))
    while any(active) or pending_pos < len(work):
        scan_rows = list(range(B))
        forward_rows = []
        while scan_rows:
            refill = []
            next_scan = []
            for row in scan_rows:
                if active[row]:
                    if process_next_token(row):
                        forward_rows.append(row)
                    else:
                        refill.append(row)
                elif pending_pos < len(work):
                    refill.append(row)
            if refill:
                next_scan = refill_rows(refill)
            scan_rows = next_scan

        if not forward_rows:
            continue
        token_tensor.fill_(0)
        for row in forward_rows:
            token_tensor[row, 0] = next_cpu[row]
        logits = model.forward(token_tensor, batch_state).view(B, -1)
        choose_next_tokens(logits, forward_rows)
        forward_steps += 1
        if args.progress_every > 0 and forward_steps % args.progress_every == 0:
            active_count = sum(int(x) for x in active)
            now = time.perf_counter()
            dt_total = max(now - t_decode, 1e-9)
            dt_window = max(now - last_progress_t, 1e-9)
            delta_tokens = decoded_token_events - last_progress_tokens
            last_progress_t = now
            last_progress_tokens = decoded_token_events
            print(
                f"worker={args.worker_rank} dynamic step={forward_steps} active={active_count}/{B} "
                f"done={len(rows)}/{len(work)} pending={len(work) - pending_pos} "
                f"tokens={decoded_token_events} tps={decoded_token_events / dt_total:.1f} "
                f"window_tps={delta_tokens / dt_window:.1f}",
                flush=True,
            )

    torch.cuda.synchronize()
    print(
        f"worker={args.worker_rank} dynamic done B={B} rows={len(rows)} "
        f"decode_s={time.perf_counter() - t_decode:.3f} tokens={decoded_token_events}",
        flush=True,
    )
    return rows

def run_worker(args: argparse.Namespace, result_queue: Any) -> None:
    random.seed(args.seed + args.worker_rank * 100003)
    torch.manual_seed(args.seed + args.worker_rank * 100003)
    torch.cuda.manual_seed_all(args.seed + args.worker_rank * 100003)
    tasks = load_tasks(args.dataset)
    start = len(tasks) * args.worker_rank // args.num_workers
    end = len(tasks) * (args.worker_rank + 1) // args.num_workers
    task_indices = list(range(start, end))
    work = [(task_idx, sample_id) for task_idx in task_indices for sample_id in range(args.rollout)]
    print(
        f"worker={args.worker_rank}/{args.num_workers} tasks={len(task_indices)} work_items={len(work)} bsz={args.bsz} "
        f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        flush=True,
    )
    sys.path.insert(0, args.albatross_dir)
    import rwkv7_fast_v3a as v3a

    os.chdir(args.albatross_dir)
    v3a.MODEL_PATH = args.model
    v3a.WKV_MODE = args.wkv
    v3a.EMB_DEVICE = args.emb
    v3a.RKV_MODE = args.batched_rkv
    v3a.CMIX_SPARSE = args.cmix_sparse
    v3a.LOWRANK_WEIGHT = args.lowrank_weight
    v3a.ORIG_LINEAR_GROUPS = v3a.parse_orig_linear_groups(args.orig_linear_groups)
    torch.set_grad_enabled(False)
    v3a.load_extensions(v3a.WKV_MODE)
    model = v3a.RWKV7()
    tokenizer = PIPELINE(model, "rwkv_vocab_v20230424")
    token_device = "cpu" if model.emb_cpu else "cuda"
    prefill_cache = build_prefill_cache(args, tasks, task_indices, model, tokenizer, token_device)
    started = time.perf_counter()
    with torch.inference_mode():
        rows = generate_dynamic_rollouts(args, tasks, work, model, tokenizer, token_device, prefill_cache)
    for row in rows:
        result_queue.put({"type": "row", "row": row})
    print(f"worker={args.worker_rank} finished elapsed_s={time.perf_counter() - started:.3f}", flush=True)


def run_worker_process(args: argparse.Namespace, gpu: str, result_queue: Any) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    if args.torch_extensions_dir:
        os.environ["TORCH_EXTENSIONS_DIR"] = args.torch_extensions_dir
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    try:
        run_worker(args, result_queue)
    except Exception as exc:
        result_queue.put(
            {
                "type": "error",
                "worker_rank": args.worker_rank,
                "gpu": gpu,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise


def extract_post_think(completion: str) -> str | None:
    marker = "</think>"
    idx = completion.find(marker)
    if idx < 0:
        return None
    tail = completion[idx + len(marker):]
    boundaries = [pos for token in ("</think>", "<think>", "\nUser:", "\nAssistant:") if (pos := tail.find(token)) >= 0]
    return tail[:min(boundaries)] if boundaries else tail


def extract_format_b_parts(completion: str) -> tuple[str, str] | None:
    if (
        completion.count("<think>") != 0
        or completion.count("</think>") != 1
        or completion.count("<answer>") != 1
        or completion.count("</answer>") != 1
    ):
        return None
    match = re.fullmatch(
        r"\s*(?P<thought>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*",
        completion,
        flags=re.DOTALL,
    )
    if match is None:
        return None
    return match.group("thought"), match.group("answer")


def verify_one(item: dict[str, Any]) -> dict[str, Any]:
    completion = str(item["completion"])
    format_b_parts = extract_format_b_parts(completion)
    has_nonempty_thought = bool(format_b_parts is not None and format_b_parts[0].strip())
    answer_region = format_b_parts[1] if format_b_parts is not None else extract_post_think(completion)
    extracted = extract_response_answer(completion, str(item["answer"])) if answer_region is not None else None
    try:
        if answer_region is None:
            raise ValueError("missing </think> before final answer")
        if extracted is None:
            raise ValueError("no final answer found after </think>")
        gold = parse(f"$\\boxed{{{item['answer']}}}$")
        pred = parse(f"$\\boxed{{{extracted}}}$")
        correct = bool(pred and verify(gold, pred, strict=False))
        error = ""
    except Exception as exc:
        correct = False
        error = f"{type(exc).__name__}: {exc}"
    out = dict(item)
    out["pred_extracted"] = extracted
    out["correct"] = correct
    out["has_think_close"] = answer_region is not None
    out["think_close_count"] = completion.count("</think>")
    out["post_think_boxed_count"] = 0 if answer_region is None else answer_region.count("\\boxed{")
    out["has_nonempty_thought"] = has_nonempty_thought
    out["strict_format"] = bool(has_nonempty_thought and item.get("ended_eod", False))
    out["strict_correct"] = bool(correct and out["strict_format"])
    out["verify_error"] = error
    return out


def run_master(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.dataset)
    config = {
        "model": args.model,
        "albatross_dir": args.albatross_dir,
        "dataset": args.dataset,
        "rollout": args.rollout,
        "bsz": args.bsz,
        "max_new_tokens": args.max_new_tokens,
        "eval_tail_max_tokens": args.eval_tail_max_tokens,
        "eval_tail_prefix": args.eval_tail_prefix,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "sampler_order": "temperature -> top_k -> top_p",
        "penalty": "off",
        "prompt_style": args.prompt_style,
        "wkv": args.wkv,
        "emb": args.emb,
        "batched_rkv": args.batched_rkv,
        "cmix_sparse": args.cmix_sparse,
        "lowrank_weight": args.lowrank_weight,
        "orig_linear_groups": args.orig_linear_groups,
        "torch_extensions_dir": args.torch_extensions_dir,
        "verify_workers": args.verify_workers,
        "seed": args.seed,
        "gpus": args.gpus,
    }

    gpus = [x.strip() for x in args.gpus.replace(",", " ").split() if x.strip()]
    if not gpus:
        raise ValueError("no GPUs selected")
    print(f"master tasks={len(tasks)} rollout={args.rollout} total_generations={len(tasks) * args.rollout}", flush=True)
    print(f"master launching workers on GPUs={gpus}", flush=True)
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=max(128, min(4096, args.bsz * 4)))
    procs = []
    raw_rows = []
    worker_errors = []
    started = time.perf_counter()

    def drain_results(block: bool) -> None:
        while True:
            try:
                msg = result_queue.get(timeout=0.5 if block else 0.0)
            except queue.Empty:
                break
            if msg.get("type") == "row":
                raw_rows.append(msg["row"])
            elif msg.get("type") == "error":
                worker_errors.append(msg)
            else:
                worker_errors.append({"type": "bad_message", "message": repr(msg)[:500]})

    for rank, gpu in enumerate(gpus):
        worker_args = argparse.Namespace(**vars(args))
        worker_args.worker_rank = rank
        worker_args.num_workers = len(gpus)
        proc = ctx.Process(target=run_worker_process, args=(worker_args, gpu, result_queue))
        proc.start()
        procs.append((rank, gpu, proc))

    while any(proc.is_alive() for _rank, _gpu, proc in procs):
        drain_results(block=True)
    for _rank, _gpu, proc in procs:
        proc.join()
    drain_results(block=False)

    failed = [(rank, gpu, proc.exitcode) for rank, gpu, proc in procs if proc.exitcode != 0]
    if worker_errors:
        raise RuntimeError(f"worker errors: {worker_errors[:3]}")
    if failed:
        raise RuntimeError(f"worker failures: {failed}")
    expected_rows = len(tasks) * args.rollout
    if len(raw_rows) != expected_rows:
        raise RuntimeError(f"missing worker rows: got {len(raw_rows)} expected {expected_rows}")

    raw_rows.sort(key=lambda x: (x["task_index"], x["sample_id"]))

    print(f"master verifying rows={len(raw_rows)} workers={args.verify_workers}", flush=True)
    if args.verify_workers <= 1:
        verified = [verify_one(row) for row in raw_rows]
    else:
        with ProcessPoolExecutor(max_workers=args.verify_workers) as pool:
            verified = list(pool.map(verify_one, raw_rows, chunksize=16))
    verified.sort(key=lambda x: (x["task_index"], x["sample_id"]))
    generations_jsonl = out_dir / "generations.jsonl"
    with generations_jsonl.open("w", encoding="utf-8") as f:
        for row in verified:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_task: dict[int, list[dict[str, Any]]] = {}
    for row in verified:
        by_task.setdefault(int(row["task_index"]), []).append(row)
    correct_generations = sum(int(row["correct"]) for row in verified)
    strict_correct_generations = sum(int(row.get("strict_correct", False)) for row in verified)
    strict_format_generations = sum(int(row.get("strict_format", False)) for row in verified)
    eval_tail_applied = sum(int(row.get("eval_tail_applied", False)) for row in verified)
    pass_tasks = sum(1 for rows in by_task.values() if any(row["correct"] for row in rows))
    ended_eod = sum(int(row["ended_eod"]) for row in verified)
    ended_user_stop = sum(int(row.get("ended_user_stop", False)) for row in verified)
    truncated = sum(int(row["truncated"]) for row in verified)
    total = len(verified)
    elapsed = time.perf_counter() - started
    summary = {
        "num_tasks": len(tasks),
        "rollout": args.rollout,
        "total_generations": total,
        "correct_generations": correct_generations,
        "rollout_accuracy": correct_generations / max(total, 1),
        "strict_correct_generations": strict_correct_generations,
        "strict_accuracy": strict_correct_generations / max(total, 1),
        "strict_format_generations": strict_format_generations,
        "strict_format_rate": strict_format_generations / max(total, 1),
        "eval_tail_applied": eval_tail_applied,
        "eval_tail_applied_rate": eval_tail_applied / max(total, 1),
        "pass_at_rollout_accuracy": pass_tasks / max(len(tasks), 1),
        "ended_eod": ended_eod,
        "ended_eod_rate": ended_eod / max(total, 1),
        "ended_user_stop": ended_user_stop,
        "ended_user_stop_rate": ended_user_stop / max(total, 1),
        "truncated": truncated,
        "truncated_rate": truncated / max(total, 1),
        "mean_generated_tokens": sum(row["generated_tokens"] for row in verified) / max(total, 1),
        "mean_tokens_including_eod": sum(row["tokens_including_eod"] for row in verified) / max(total, 1),
        "mean_tokens_including_stop": sum(
            row.get("tokens_including_stop", row["tokens_including_eod"]) for row in verified
        )
        / max(total, 1),
        "elapsed_sec": elapsed,
        "sample_per_sec": total / max(elapsed, 1e-9),
        "token_per_sec": sum(row.get("tokens_including_stop", row["tokens_including_eod"]) for row in verified)
        / max(elapsed, 1e-9),
        "generations_jsonl": str(generations_jsonl),
        "config": config,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("MATH500_RWKV_V3A_RESULT " + json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    run_master(args)


if __name__ == "__main__":
    main()
