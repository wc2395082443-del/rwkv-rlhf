#!/usr/bin/env python3
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk
from rwkv.utils import PIPELINE
from tqdm import tqdm

import rwkv7_fast_v3a as v3a

THIS_DIR = Path(__file__).resolve().parent
TEMPLATE = """User: You are a very talented expert in <SUBJECT>. Answer this question:
<Q>
A. <|A|>
B. <|B|>
C. <|C|>
D. <|D|>

Assistant: The answer is"""
CHOICES = [" A", " B", " C", " D"]
WKV = "fp32io16"
EMB = "cpu"
BATCHED_RKV = "off"
CMIX_SPARSE = "no-fc"
LOWRANK_WEIGHT = "both"
ORIG_LINEAR_GROUPS = "att_c2c,ffn_key,head"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=v3a.MODEL_PATH)
    parser.add_argument("--dataset", default=str(THIS_DIR / "dataset" / "mmlu_test_dataset"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--bsz", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-choices", action="store_true")
    parser.add_argument("--no-sort", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    v3a.MODEL_PATH = args.model
    v3a.WKV_MODE = WKV
    v3a.EMB_DEVICE = EMB
    v3a.RKV_MODE = BATCHED_RKV
    v3a.CMIX_SPARSE = CMIX_SPARSE
    v3a.LOWRANK_WEIGHT = LOWRANK_WEIGHT
    v3a.ORIG_LINEAR_GROUPS = v3a.parse_orig_linear_groups(ORIG_LINEAR_GROUPS)
    v3a.load_extensions(v3a.WKV_MODE)
    model = v3a.RWKV7()
    tokenizer = PIPELINE(model, "rwkv_vocab_v20230424")
    choice_tokens = [tokenizer.encode(x) for x in CHOICES]
    if not all(len(x) == 1 for x in choice_tokens):
        raise RuntimeError(f"MMLU choice tokens are not single tokens: {choice_tokens}")
    choice_tokens = torch.tensor([x[0] for x in choice_tokens], dtype=torch.long, device="cuda")

    samples = load_from_disk(args.dataset)
    if hasattr(samples, "keys"):
        samples = samples[args.split]
    if args.limit > 0:
        samples = samples.select(range(min(args.limit, len(samples))))
    items = [make_item(i, sample, tokenizer, args.shuffle_choices) for i, sample in enumerate(samples)]
    if not args.no_sort:
        items.sort(key=lambda x: x["len"])
    print_format_example(items)

    total = correct = 0
    loss_sum = 0.0
    rows = []
    subject_stats: dict[str, list[int]] = {}
    token_device = "cpu" if model.emb_cpu else "cuda"
    pbar = tqdm(range(0, len(items), args.bsz), desc="MMLU")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for start in pbar:
        batch = items[start:start + args.bsz]
        B = len(batch)
        T = max(x["len"] for x in batch)
        tokens = torch.zeros((B, T), dtype=torch.long, device=token_device)
        last = torch.empty((B,), dtype=torch.long, device="cuda")
        gt = torch.empty((B,), dtype=torch.long, device="cuda")
        for row, item in enumerate(batch):
            ids = item["ids"]
            tokens[row, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=token_device)
            last[row] = len(ids) - 1
            gt[row] = item["answer"]

        state = model.zero_state(B)
        logits = model.forward_last_at(tokens, state, last).float()
        logp = F.log_softmax(logits, dim=-1).index_select(1, choice_tokens)
        pred = torch.argmax(logp, dim=-1)
        loss = -logp[torch.arange(B, device="cuda"), gt]
        pred_cpu = pred.detach().cpu().tolist()
        loss_cpu = loss.detach().cpu().tolist()
        gt_cpu = gt.detach().cpu().tolist()
        for row, item in enumerate(batch):
            ok = int(pred_cpu[row] == gt_cpu[row])
            correct += ok
            total += 1
            loss_sum += float(loss_cpu[row])
            stat = subject_stats.setdefault(item["subject"], [0, 0])
            stat[0] += ok
            stat[1] += 1
            if args.out:
                rows.append({"idx": item["idx"], "subject": item["subject"], "answer": gt_cpu[row], "pred": pred_cpu[row], "correct": bool(ok), "loss": float(loss_cpu[row]), "len": item["len"]})
        pbar.set_description(f"MMLU acc={correct / total:.5f} loss={loss_sum / total:.4f}")

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"MMLU_RESULT total={total} correct={correct} acc={correct / total:.6f} mean_loss={loss_sum / total:.6f} time_s={dt:.3f} sample_s={total / dt:.3f} bsz={args.bsz}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": {"total": total, "correct": correct, "acc": correct / total, "mean_loss": loss_sum / total, "time_s": dt, "sample_s": total / dt, "bsz": args.bsz}, "subjects": subject_stats, "rows": sorted(rows, key=lambda x: x["idx"])}, f, ensure_ascii=False, indent=2)

def make_item(idx: int, sample, tokenizer: PIPELINE, shuffle_choices: bool) -> dict:
    choices = list(sample["choices"])
    answer = int(sample["answer"])
    if shuffle_choices and not any("Both" in x for x in choices):
        original = choices[answer]
        random.shuffle(choices)
        answer = choices.index(original)
    prompt = (
        TEMPLATE.replace("<Q>", sample["question"])
        .replace("<|A|>", choices[0])
        .replace("<|B|>", choices[1])
        .replace("<|C|>", choices[2])
        .replace("<|D|>", choices[3])
        .replace("<SUBJECT>", sample["subject"].replace("_", " "))
    )
    ids = [0] + tokenizer.encode(prompt.replace("\r\n", "\n").strip())
    return {"idx": idx, "ids": ids, "len": len(ids), "answer": answer, "subject": sample["subject"], "prompt": prompt}

def print_format_example(items: list[dict]) -> None:
    if not items:
        return
    item = min(items, key=lambda x: x["idx"])
    print("Format example:")
    print("-" * 80)
    print(item["prompt"])
    print("-" * 80)

if __name__ == "__main__":
    main()
