#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import torch

BASE_DIR = Path("/root/RWKV-LM/RWKV-v7/train_temp")
BASELINE_DIR = Path("/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1")
os.chdir(BASE_DIR)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

import train_rl_baseline as mod

reward_spec = importlib.util.spec_from_file_location("baseline_reward", BASELINE_DIR / "reward.py")
reward_mod = importlib.util.module_from_spec(reward_spec)
reward_spec.loader.exec_module(reward_mod)


def read_jsonl(path, max_samples=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_samples is not None and len(rows) >= max_samples:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_decode(tok, ids):
    try:
        return tok.decode(ids, utf8_errors="replace")
    except Exception:
        try:
            return tok.decode(ids)
        except Exception:
            try:
                return tok.decodeBytes(ids).decode("utf-8", errors="replace")
            except Exception:
                return "".join(chr(int(x) % 256) for x in ids)


def pad_batch(seqs, device):
    max_len = max(len(x) for x in seqs)
    x = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
    for i, ids in enumerate(seqs):
        x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return x


@torch.no_grad()
def score_ref_logps(ref_model, prompts, comps, device, micro_batch):
    out = []
    items = [(p, c) for p, c in zip(prompts, comps)]
    for start in range(0, len(items), micro_batch):
        batch = items[start : start + micro_batch]
        seqs = [p + c for p, c in batch]
        x = pad_batch(seqs, device)
        inp = x[:, :-1].contiguous()
        tgt = x[:, 1:].contiguous()
        logits = ref_model(inp)
        if torch.is_tensor(logits) and logits.dim() == 2:
            logits = logits.unsqueeze(0)
        log_z = torch.logsumexp(logits.float(), dim=-1)
        picked = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).float()
        logp_all = picked - log_z
        for bi, (p, c) in enumerate(batch):
            s = len(p) - 1
            e = s + len(c)
            out.append([float(v) for v in logp_all[bi, s:e].detach().cpu().tolist()])
        del x, inp, tgt, logits, log_z, picked, logp_all
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def choose(records, key):
    best_i = max(range(len(records)), key=lambda i: records[i].get(key, -1e30))
    return records[best_i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_model", required=True)
    ap.add_argument("--ref_model", required=True)
    ap.add_argument("--tokenizer", default="/root/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt")
    ap.add_argument("--train_jsonl", default="/root/RWKV-LM/RWKV7-statetuning/gsm8k_train_formatted.jsonl")
    ap.add_argument("--eval_jsonl", default="/root/RWKV-LM/RWKV7-statetuning/gsm8k_test_formatted_1of8.jsonl")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.6)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--chunk_questions", type=int, default=4)
    ap.add_argument("--score_micro_batch", type=int, default=2)
    ap.add_argument("--save_text", type=int, default=1)
    ap.add_argument("--random_seed", type=int, default=42)
    args0 = ap.parse_args()

    out_dir = Path(args0.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.argv = [
        "pa_eval",
        "--load_model",
        args0.policy_model,
        "--proj_dir",
        str(out_dir),
        "--tokenizer",
        args0.tokenizer,
        "--train_jsonl",
        args0.train_jsonl,
        "--eval_jsonl",
        args0.eval_jsonl,
        "--strategy",
        "deepspeed_stage_3_offload",
        "--precision",
        "bf16",
        "--use_stateful_rollout",
        "1",
        "--max_new_tokens",
        str(args0.max_new_tokens),
        "--micro_batch",
        "1",
        "--rollout_forward_batch",
        "64",
        "--random_seed",
        str(args0.random_seed),
    ]
    t_init = time.time()
    tr_args = mod.parse_args()
    mod.set_seed(int(args0.random_seed))

    rwkv_precision = {"32": "fp32", 32: "fp32", "16": "fp16", 16: "fp16"}.get(tr_args.precision, tr_args.precision)
    os.environ["RWKV_MY_TESTING"] = tr_args.my_testing
    os.environ["RWKV_CTXLEN"] = str(int(tr_args.ctx_len))
    os.environ["RWKV_HEAD_SIZE"] = str(int(tr_args.head_size))
    os.environ["RWKV_FLOAT_MODE"] = rwkv_precision
    os.environ["RWKV_JIT_ON"] = "0"

    sd = mod._normalize_state_dict(mod._torch_load_weights(args0.policy_model))
    tr_args.n_layer, tr_args.n_embd, tr_args.vocab_size, tr_args.dim_ffn = mod._infer_arch(sd)
    tr_args.dim_att = tr_args.n_embd

    tok = mod.TRIE_TOKENIZER(args0.tokenizer)

    policy_model = mod.PaddedRWKV(tr_args)
    policy_model.load_state_dict(sd, strict=True)
    policy_model = mod._cast_ref_model_dtype(policy_model.to("cuda"))
    policy_model.eval()
    for p in policy_model.parameters():
        p.requires_grad = False

    ref_sd = mod._normalize_state_dict(mod._torch_load_weights(args0.ref_model))
    ref_model = mod.PaddedRWKV(tr_args)
    ref_model.load_state_dict(ref_sd, strict=True)
    ref_model = mod._cast_ref_model_dtype(ref_model.to("cuda"))
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    rollout_args = types.SimpleNamespace(
        MODEL_NAME=str(Path(args0.policy_model).with_suffix("")),
        vocab_size=int(tr_args.vocab_size),
    )
    rollout_model = mod.RWKV_x070(rollout_args)
    rollout_model.eval()
    cfg = types.SimpleNamespace(tune_mode="state", rollout_forward_batch=64)
    infer = mod.TrainTempBatchInference(
        infer_model=rollout_model,
        train_model=policy_model,
        encode_fn=tok.encode,
        decode_fn=lambda ids: safe_decode(tok, ids),
        device="cuda",
        cfg=cfg,
    )

    max_samples = None if args0.max_samples <= 0 else args0.max_samples
    data = read_jsonl(args0.eval_jsonl, max_samples=max_samples)
    records_path = out_dir / "pa_eval.jsonl"
    if records_path.exists():
        records_path.unlink()

    totals = {
        "n": 0,
        "samples": 0,
        "correct_samples": 0,
        "first_correct": 0,
        "pass8": 0,
        "pa_mean_correct": 0,
        "pa_sum_correct": 0,
        "pa_mean_pos_correct": 0,
        "oracle_correct": 0,
        "truncated": 0,
        "gen_tokens": 0,
    }
    hist = {str(i): 0 for i in range(args0.group_size + 1)}
    t0 = time.time()

    with open(records_path, "w", encoding="utf-8") as fout:
        for start in range(0, len(data), args0.chunk_questions):
            batch = data[start : start + args0.chunk_questions]
            problems = [ex.get("problem", "") for ex in batch]
            answers = [ex.get("solution", ex.get("ground_truth", ex.get("answer", ex.get("original_answer", "")))) for ex in batch]
            prompts_text = [mod._baseline_mod.build_prompt(p) for p in problems]
            prompt_tokens = []
            for ps in prompts_text:
                ids = tok.encode(ps)
                max_prompt_len = int(tr_args.ctx_len) - int(args0.max_new_tokens) - 4
                if len(ids) > max_prompt_len:
                    ids = ids[-max_prompt_len:]
                prompt_tokens.append(ids)

            comp_tokens, policy_logps, comp_texts, truncated = infer.generate_group_parallel(
                prompt_tokens_list=prompt_tokens,
                group_size=args0.group_size,
                max_new_tokens=args0.max_new_tokens,
                temperature=args0.temperature,
                top_p=args0.top_p,
                top_k=args0.top_k,
            )
            cleanup = getattr(rollout_model, "cleanup_stateful_rollout", None)
            if cleanup is not None:
                cleanup()

            flat_prompts = []
            for p in prompt_tokens:
                flat_prompts.extend([p] * args0.group_size)
            ref_logps = score_ref_logps(ref_model, flat_prompts, comp_tokens, "cuda", args0.score_micro_batch)

            for bi, (problem, answer) in enumerate(zip(problems, answers)):
                sample_records = []
                for j in range(args0.group_size):
                    flat = bi * args0.group_size + j
                    text = comp_texts[flat]
                    toks = comp_tokens[flat]
                    reward, is_correct, is_format_correct, details = reward_mod.calculate_reward_details(
                        text=text,
                        ground_truth=answer,
                        token_length=len(toks),
                        min_tokens=200,
                        max_tokens=args0.max_new_tokens,
                        length_weight=0.0,
                        repeat_ngram=False,
                        repeat_penalty=0.0,
                        zstd_threshold=2.5,
                        zstd_penalty_weight=0.0,
                    )
                    pl = policy_logps[flat]
                    rl = ref_logps[flat]
                    m = min(len(pl), len(rl), len(toks))
                    if m > 0:
                        diffs = [float(pl[k]) - float(rl[k]) for k in range(m)]
                        pa_sum = sum(diffs)
                        pa_mean = pa_sum / m
                        pa_pos = sum(1 for x in diffs if x > 0.0) / m
                    else:
                        pa_sum = -1e30
                        pa_mean = -1e30
                        pa_pos = -1e30
                    rec = {
                        "sample_idx": j,
                        "is_correct": bool(is_correct),
                        "is_format_correct": bool(is_format_correct),
                        "truncated": bool(truncated[flat]),
                        "gen_len": len(toks),
                        "pa_sum": pa_sum,
                        "pa_mean": pa_mean,
                        "pa_pos_frac": pa_pos,
                        "pred_extracted": details.get("extracted_answer"),
                        "gt_extracted": details.get("ground_truth_answer"),
                    }
                    if args0.save_text:
                        rec["response"] = text
                    sample_records.append(rec)

                num_correct = sum(int(x["is_correct"]) for x in sample_records)
                hist[str(num_correct)] += 1
                first = sample_records[0]
                pa_mean_best = choose(sample_records, "pa_mean")
                pa_sum_best = choose(sample_records, "pa_sum")
                pa_pos_best = choose(sample_records, "pa_pos_frac")
                out = {
                    "idx": start + bi,
                    "problem": problem,
                    "ground_truth": answer,
                    "num_correct": num_correct,
                    "first_correct": bool(first["is_correct"]),
                    "pass8_correct": bool(num_correct > 0),
                    "pa_mean_correct": bool(pa_mean_best["is_correct"]),
                    "pa_sum_correct": bool(pa_sum_best["is_correct"]),
                    "pa_pos_frac_correct": bool(pa_pos_best["is_correct"]),
                    "pa_mean_selected": pa_mean_best["sample_idx"],
                    "pa_sum_selected": pa_sum_best["sample_idx"],
                    "pa_pos_selected": pa_pos_best["sample_idx"],
                    "samples": sample_records,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")

                totals["n"] += 1
                totals["samples"] += args0.group_size
                totals["correct_samples"] += num_correct
                totals["first_correct"] += int(first["is_correct"])
                totals["pass8"] += int(num_correct > 0)
                totals["pa_mean_correct"] += int(pa_mean_best["is_correct"])
                totals["pa_sum_correct"] += int(pa_sum_best["is_correct"])
                totals["pa_mean_pos_correct"] += int(pa_pos_best["is_correct"])
                totals["oracle_correct"] += int(num_correct > 0)
                totals["truncated"] += sum(int(x["truncated"]) for x in sample_records)
                totals["gen_tokens"] += sum(int(x["gen_len"]) for x in sample_records)

            done = totals["n"]
            elapsed = time.time() - t0
            partial = {
                "done": done,
                "sample_avg_acc": totals["correct_samples"] / max(1, totals["samples"]),
                "first_acc": totals["first_correct"] / max(1, done),
                "pass8": totals["pass8"] / max(1, done),
                "pa_mean_acc": totals["pa_mean_correct"] / max(1, done),
                "pa_sum_acc": totals["pa_sum_correct"] / max(1, done),
                "tok_per_s": totals["gen_tokens"] / max(1e-9, elapsed),
            }
            print(json.dumps(partial, ensure_ascii=False), flush=True)

    n = max(1, totals["n"])
    summary = {
        "policy_model": args0.policy_model,
        "ref_model": args0.ref_model,
        "eval_jsonl": args0.eval_jsonl,
        "group_size": args0.group_size,
        "temperature": args0.temperature,
        "top_p": args0.top_p,
        "top_k": args0.top_k,
        "max_new_tokens": args0.max_new_tokens,
        "total_questions": totals["n"],
        "sample_avg_acc": totals["correct_samples"] / max(1, totals["samples"]),
        "first_sample_acc": totals["first_correct"] / n,
        "pass8": totals["pass8"] / n,
        "pa_mean_rerank_acc": totals["pa_mean_correct"] / n,
        "pa_sum_rerank_acc": totals["pa_sum_correct"] / n,
        "pa_pos_frac_rerank_acc": totals["pa_mean_pos_correct"] / n,
        "num_correct_hist": hist,
        "trunc_rate": totals["truncated"] / max(1, totals["samples"]),
        "avg_gen_len": totals["gen_tokens"] / max(1, totals["samples"]),
        "elapsed_sec": time.time() - t0,
        "init_sec": t0 - t_init,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
