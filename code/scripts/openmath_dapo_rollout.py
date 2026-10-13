import argparse
import importlib.util
import json
import os
import random
import statistics
import time
from pathlib import Path

import torch
from math_verify import parse, verify
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_rwkv_tokenizer(tokenizer_path: str, vocab_path: str):
    spec = importlib.util.spec_from_file_location("rwkv_tokenizer", tokenizer_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TRIE_TOKENIZER(vocab_path)


def repeats(text: str) -> bool:
    words = text.split()
    for width in (30, 20, 15, 12, 10, 8):
        for start in range(max(0, len(words) - 3 * width + 1)):
            block = words[start : start + width]
            if block == words[start + width : start + 2 * width] == words[start + 2 * width : start + 3 * width]:
                return True
    return False


def correct(gold_answer: str, completion: str) -> bool:
    try:
        gold = parse(f"$\\boxed{{{gold_answer}}}$")
        prediction = parse(completion)
        return bool(prediction and verify(gold, prediction, strict=False))
    except Exception:
        return False


def make_rwkv_completion(completion: str, answer: str) -> str:
    reasoning = completion.strip().replace("<think>", "").replace("</think>", "").strip()
    return f"<think>\n{reasoning}\n</think>\n<answer>\n{answer}\n</answer>"


def write_summary(path: Path, rows: list[dict], started: float, target: int) -> None:
    generated = [row["generated_tokens"] for row in rows]
    accepted = [row for row in rows if row["accepted"]]
    elapsed = time.time() - started
    summary = {
        "target": target,
        "completed": len(rows),
        "accepted": len(accepted),
        "accuracy": sum(row["correct"] for row in rows) / max(1, len(rows)),
        "trunc_rate": sum(row["truncated"] for row in rows) / max(1, len(rows)),
        "repeat_rate": sum(row["repeat"] for row in rows) / max(1, len(rows)),
        "rwkv_over_ctx_rate": sum(row["rwkv_over_ctx"] for row in rows) / max(1, len(rows)),
        "avg_generated_tokens": statistics.mean(generated) if generated else 0,
        "total_generated_tokens": sum(generated),
        "elapsed_sec_current_process": elapsed,
        "tokens_per_sec_current_process": sum(generated) / max(1, elapsed),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-new", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rwkv-tokenizer", required=True)
    parser.add_argument("--rwkv-vocab", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "generations.jsonl"
    accepted_path = output_dir / "accepted_rwkv_sft.jsonl"
    summary_path = output_dir / "summary.json"

    source_rows = []
    with open(args.data, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                source_rows.append(json.loads(line))
    if args.limit > 0:
        source_rows = source_rows[: args.limit]

    completed_ids = set()
    result_rows = []
    if result_path.exists():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            result_rows.append(row)
            completed_ids.add(row["unique_id"])

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    rwkv_tokenizer = load_rwkv_tokenizer(args.rwkv_tokenizer, args.rwkv_vocab)

    started = time.time()
    pending = [row for row in source_rows if row.get("unique_id") not in completed_ids]
    print(f"loaded={len(source_rows)} resume={len(completed_ids)} pending={len(pending)} batch={args.batch}", flush=True)

    for offset in range(0, len(pending), args.batch):
        batch_rows = pending[offset : offset + args.batch]
        prompts = []
        for row in batch_rows:
            content = (
                "Solve the following math problem. Make sure to put the answer "
                "(and only answer) inside \\boxed{}.\n\n" + row["problem"].strip()
            )
            prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        prompt_width = encoded.input_ids.shape[1]
        max_new_tokens = max(1, min(args.max_new, 8192 - prompt_width))
        batch_started = time.time()
        with torch.inference_mode():
            sequences = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        batch_elapsed = time.time() - batch_started

        new_results = []
        new_accepted = []
        for index, source in enumerate(batch_rows):
            generated_ids = sequences[index, prompt_width:]
            generated_tokens = int((generated_ids != tokenizer.pad_token_id).sum().item())
            completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            is_correct = correct(str(source["answer"]), completion)
            is_truncated = generated_tokens >= max_new_tokens - 1
            is_repeat = repeats(completion)
            rwkv_completion = make_rwkv_completion(completion, str(source["answer"]))
            rwkv_problem = (
                "User: " + source["problem"].strip()
                + "\nPlease reason step by step. Close the reasoning with </think>, then output exactly "
                + "<answer>final answer</answer> and immediately end the response.\nAssistant: <think>\n"
            )
            rwkv_tokens = len(rwkv_tokenizer.encode(rwkv_problem + rwkv_completion))
            rwkv_over_ctx = rwkv_tokens > 8192
            accepted = is_correct and not is_truncated and not is_repeat and not rwkv_over_ctx
            result = {
                "unique_id": source.get("unique_id"),
                "problem": source["problem"],
                "answer": source["answer"],
                "completion": completion,
                "generated_tokens": generated_tokens,
                "correct": is_correct,
                "truncated": is_truncated,
                "repeat": is_repeat,
                "rwkv_tokens": rwkv_tokens,
                "rwkv_over_ctx": rwkv_over_ctx,
                "accepted": accepted,
                "elapsed_batch_sec": batch_elapsed,
            }
            new_results.append(result)
            if accepted:
                new_accepted.append(
                    {
                        "problem": rwkv_problem,
                        "answer": source["answer"],
                        "teacher_completion": rwkv_completion,
                        "teacher": "openmath_nemotron_1.5b",
                        "unique_id": source.get("unique_id"),
                        "rwkv_tokens": rwkv_tokens,
                        "source_generated_tokens": generated_tokens,
                    }
                )

        with result_path.open("a", encoding="utf-8") as handle:
            for row in new_results:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with accepted_path.open("a", encoding="utf-8") as handle:
            for row in new_accepted:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        result_rows.extend(new_results)
        write_summary(summary_path, result_rows, started, len(source_rows))
        done = len(result_rows)
        correct_count = sum(row["correct"] for row in result_rows)
        accepted_count = sum(row["accepted"] for row in result_rows)
        total_tokens = sum(row["generated_tokens"] for row in result_rows)
        elapsed = time.time() - started
        print(
            f"done={done}/{len(source_rows)} correct={correct_count} accepted={accepted_count} "
            f"batch_s={batch_elapsed:.1f} process_tps={total_tokens / max(1, elapsed):.1f}",
            flush=True,
        )

    write_summary(summary_path, result_rows, started, len(source_rows))
    print("ROLLOUT_DONE", flush=True)


if __name__ == "__main__":
    main()
