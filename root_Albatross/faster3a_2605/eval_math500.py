#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from math_verify import parse, verify
from rwkv.utils import PIPELINE


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
    parser.add_argument("--ctx-limit", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.28)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-style",
        choices=("fake_think", "plain"),
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
    k = min(max(1, int(top_k)), logits.size(-1))
    if temperature <= 0.0 or top_p <= 0.0 or k == 1:
        return torch.argmax(logits, dim=-1)
    vals, ids = torch.topk(logits.float(), k=k, dim=-1, sorted=True)
    probs = torch.softmax(vals if temperature == 1.0 else vals / float(temperature), dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    if top_p < 1.0:
        keep = torch.argmax((cdf >= float(top_p)).to(torch.int32), dim=-1)
        mass = cdf.gather(1, keep.view(-1, 1)).view(-1)
    else:
        mass = cdf[:, -1]
    r = torch.rand((logits.size(0), 1), device=logits.device) * mass.view(-1, 1)
    picked = torch.searchsorted(cdf, r).view(-1, 1)
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
    batch_state = model.zero_state(B)
    pending_pos = 0
    slot_work: list[tuple[int, int] | None] = [None] * B
    prompt_lengths = [0] * B
    generated: list[list[int]] = [[] for _ in range(B)]
    active = [False] * B
    token_counts = [0] * B
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

    def finish_row(row: int, stop_reason: str) -> None:
        work_item = slot_work[row]
        assert work_item is not None
        task_idx, sample_id = work_item
        token_ids = generated[row]
        if out_last[row] < len(token_ids):
            pending = tokenizer.decode(token_ids[out_last[row] :])
            if "\ufffd" not in pending:
                out_texts[row] += pending
        completion = out_texts[row].split("\nUser:", 1)[0]
        if completion.startswith(">"):
            completion = completion[1:]
        task = tasks[task_idx]
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
                "generated_tokens": len(token_ids),
                "tokens_including_eod": token_counts[row],
                "tokens_including_stop": token_counts[row],
                "ended_eod": stop_reason == "eod",
                "ended_user_stop": stop_reason == "user_stop",
                "stop_reason": stop_reason,
                "truncated": stop_reason == "max_tokens",
                "completion": completion.strip(),
            }
        )
        slot_work[row] = None
        prompt_lengths[row] = 0
        generated[row] = []
        active[row] = False
        token_counts[row] = 0
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
            token_counts[row] = 0
            out_texts[row] = ""
            out_last[row] = 0
            assigned.append(row)
            init_logits.append(logits)
        if assigned:
            sampled = sample_logits(torch.stack(init_logits, dim=0), args.temperature, args.top_p, args.top_k)
            for row, token in zip(assigned, sampled.detach().cpu().tolist()):
                next_cpu[row] = int(token)
        return assigned

    def process_next_token(row: int) -> bool:
        nonlocal decoded_token_events
        token = int(next_cpu[row])
        token_counts[row] += 1
        decoded_token_events += 1
        if token == 0:
            finish_row(row, "eod")
            return False
        generated[row].append(token)
        pending = tokenizer.decode(generated[row][out_last[row] :])
        if "\ufffd" not in pending:
            out_texts[row] += pending
            out_last[row] = len(generated[row])
            if "\nUser:" in out_texts[row]:
                finish_row(row, "user_stop")
                return False
        if token_counts[row] >= args.max_new_tokens:
            finish_row(row, "max_tokens")
            return False
        return True

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
        sampled = sample_logits(logits, args.temperature, args.top_p, args.top_k).detach().cpu().tolist()
        for row in forward_rows:
            next_cpu[row] = int(sampled[row])
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


def verify_one(item: dict[str, Any]) -> dict[str, Any]:
    try:
        gold = parse(f"$\\boxed{{{item['answer']}}}$")
        pred = parse(str(item["completion"]))
        correct = bool(pred and verify(gold, pred, strict=False))
        error = ""
    except Exception as exc:
        correct = False
        error = f"{type(exc).__name__}: {exc}"
    out = dict(item)
    out["correct"] = correct
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
