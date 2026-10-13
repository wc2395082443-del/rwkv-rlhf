#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
import random
import time
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

# Keep RWKV env aligned with the GRPO code before model construction.
os.environ.setdefault("RWKV_HEAD_SIZE_A", "64")
os.environ.setdefault("RWKV_MY_TESTING", "x070")
os.environ.setdefault("RWKV_TRAIN_TYPE", "fullstate")
os.environ.setdefault("RWKV_CTXLEN", "8192")
os.environ.setdefault("FUSED_KERNEL", "0")
os.environ.setdefault("WKV", "cuda")

from main import load_train_model_rwkv7_cuda, normalize_model_arg, unfreeze_all_parameters
from utils import build_prompt, append_jsonl, set_seed


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tokenizer(path: str):
    from reference.utils import TRIE_TOKENIZER
    tok = TRIE_TOKENIZER(path)

    def encode(s: str):
        return [int(x) for x in tok.encode(s or "")]

    return encode


def make_examples(rows: List[Dict[str, Any]], encode, prompt_mode: str, teacher_field: str, ctx_len: int, add_eos: bool):
    examples = []
    dropped = 0
    for i, row in enumerate(rows):
        problem = row.get("problem", "")
        completion = row.get(teacher_field, row.get("completion", row.get("response", "")))
        if not str(problem).strip() or not str(completion).strip():
            dropped += 1
            continue
        prompt = build_prompt(problem, mode=prompt_mode)
        prompt_tokens = encode(prompt)
        comp_tokens = encode(str(completion).strip())
        if add_eos:
            comp_tokens = comp_tokens + [0]
        # Keep the full prompt; truncate only pathological completions from the left-end budget.
        max_comp = max(1, int(ctx_len) - len(prompt_tokens) - 2)
        if len(comp_tokens) > max_comp:
            comp_tokens = comp_tokens[:max_comp]
        if len(prompt_tokens) < 1 or len(comp_tokens) < 1:
            dropped += 1
            continue
        examples.append({
            "idx": i,
            "prompt_tokens": prompt_tokens,
            "comp_tokens": comp_tokens,
            "seq_len": len(prompt_tokens) + len(comp_tokens),
            "problem": problem,
            "answer": row.get("answer", ""),
        })
    return examples, dropped


def pad_batch(seqs: List[List[int]], device: str, pad_id: int = 0):
    max_len = max(len(s) for s in seqs)
    return torch.tensor([s + [pad_id] * (max_len - len(s)) for s in seqs], device=device, dtype=torch.long)


def log_grad_norm(model) -> float:
    g2 = 0.0
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                g = p.grad.detach().float()
                g2 += float(g.norm(2).pow(2).item())
    return math.sqrt(g2)


def save_full_ckpt(model, path: str, step: int):
    payload = {"step": int(step), "model": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}}
    torch.save(payload, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--teacher_field", default="teacher_completion")
    ap.add_argument("--prompt_mode", default="trl_doc", choices=["trl_doc", "rwkv_boxed", "question_only"])
    ap.add_argument("--ctx_len", type=int, default=8192)
    ap.add_argument("--model_dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--micro_batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--logit_chunk_tokens", type=int, default=128)
    ap.add_argument("--save_interval", type=int, default=100)
    ap.add_argument("--save_last", type=int, default=1)
    ap.add_argument("--add_eos", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    os.environ["RWKV_CTXLEN"] = str(int(args.ctx_len))

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train.log")
    metrics_path = os.path.join(args.out_dir, "metrics.jsonl")

    def log(msg: str):
        line = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    rows = read_jsonl(args.train_jsonl)
    encode = load_tokenizer(args.tokenizer)
    examples, dropped = make_examples(rows, encode, args.prompt_mode, args.teacher_field, args.ctx_len, bool(args.add_eos))
    if not examples:
        raise RuntimeError("no SFT examples")
    avg_len = sum(e["seq_len"] for e in examples) / len(examples)
    max_len = max(e["seq_len"] for e in examples)
    log(f"SFT data: rows={len(rows)} examples={len(examples)} dropped={dropped} avg_seq={avg_len:.1f} max_seq={max_len}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, pth_path = normalize_model_arg(args.model)
    model, _ = load_train_model_rwkv7_cuda(pth_path, device=device, ctx_len=args.ctx_len, train_type="fullstate", load_dtype=args.model_dtype)
    trainable = unfreeze_all_parameters(model)
    log(f"model loaded: {pth_path} dtype={args.model_dtype} trainable={trainable}")

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.99), eps=1e-8, foreach=False)
    rng = random.Random(args.seed)
    chunk_tokens = max(1, int(args.logit_chunk_tokens))
    batch_size = max(1, int(args.batch_size))
    micro_batch = max(1, int(args.micro_batch))
    t_train = time.time()

    for step in range(1, int(args.steps) + 1):
        t0 = time.time()
        batch = [examples[rng.randrange(len(examples))] for _ in range(batch_size)]
        total_loss_sum = 0.0
        total_tokens = sum(len(e["comp_tokens"]) for e in batch)
        total_tokens = max(1, int(total_tokens))
        opt.zero_grad(set_to_none=True)
        model.train()

        for mb_start in range(0, len(batch), micro_batch):
            mb = sorted(batch[mb_start:mb_start + micro_batch], key=lambda e: e["seq_len"], reverse=True)
            seq_list = [e["prompt_tokens"] + e["comp_tokens"] for e in mb]
            seqs = pad_batch(seq_list, device=device, pad_id=0)
            inp = seqs[:, :-1].contiguous()
            tgt = seqs[:, 1:].contiguous()
            hidden = model.forward_hidden(inp)
            if torch.is_tensor(hidden) and hidden.dim() == 2:
                hidden = hidden.unsqueeze(0)
            mb_loss_sum = torch.zeros((), device=device, dtype=torch.float32)

            for bi, ex in enumerate(mb):
                prompt_len = len(ex["prompt_tokens"])
                comp_len = len(ex["comp_tokens"])
                start_idx = prompt_len - 1
                end_idx = start_idx + comp_len
                for chunk_start in range(start_idx, end_idx, chunk_tokens):
                    chunk_end = min(end_idx, chunk_start + chunk_tokens)
                    h = hidden[bi:bi + 1, chunk_start:chunk_end, :]
                    logits = model.project_logits(h).float().squeeze(0)
                    target = tgt[bi, chunk_start:chunk_end]
                    mb_loss_sum = mb_loss_sum + F.cross_entropy(logits, target, reduction="sum")
                    del logits
            loss = mb_loss_sum / float(total_tokens)
            loss.backward()
            total_loss_sum += float(mb_loss_sum.detach().item())
            del hidden, seqs, inp, tgt, mb_loss_sum, loss
            torch.cuda.empty_cache()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(args.grad_clip))
        grad_norm = log_grad_norm(model)
        opt.step()
        opt.zero_grad(set_to_none=True)
        dt = time.time() - t0
        loss_val = total_loss_sum / float(total_tokens)
        rec = {
            "step": step,
            "split": "sft_train",
            "loss": loss_val,
            "ppl": math.exp(min(20.0, loss_val)),
            "batch_size": batch_size,
            "tokens": total_tokens,
            "avg_comp_len": total_tokens / float(batch_size),
            "grad_norm": grad_norm,
            "lr": float(args.lr),
            "time": dt,
            "tokens_per_sec": total_tokens / max(dt, 1e-6),
            "elapsed": time.time() - t_train,
        }
        append_jsonl(metrics_path, rec)
        log(f"[Step {step}/{args.steps}] loss={loss_val:.4f} ppl={rec['ppl']:.2f} tokens={total_tokens} grad={grad_norm:.3f} tok/s={rec['tokens_per_sec']:.1f} time={dt:.1f}s")

        should_save = (args.save_interval > 0 and step % args.save_interval == 0) or (bool(args.save_last) and step == int(args.steps))
        if should_save:
            ckpt_path = os.path.join(args.out_dir, f"ckpt_step{step}.pth")
            save_full_ckpt(model, ckpt_path, step)
            log(f"saved checkpoint: {ckpt_path}")

    log("SFT done")


if __name__ == "__main__":
    main()
