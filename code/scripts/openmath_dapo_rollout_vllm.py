import argparse
import contextlib
import importlib.util
import io
import json
import random
import statistics
import time
from pathlib import Path

from math_verify import parse, verify
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def load_rwkv_tokenizer(tokenizer_path: str, vocab_path: str):
    spec = importlib.util.spec_from_file_location("rwkv_tokenizer", tokenizer_path)
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
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
    return f"{reasoning}\n</think>\n<answer>\n{answer}\n</answer>"


def summary(rows: list[dict], target: int, started: float) -> dict:
    generated = [row["generated_tokens"] for row in rows]
    elapsed = time.time() - started
    return {
        "target": target,
        "completed": len(rows),
        "accepted": sum(row["accepted"] for row in rows),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-new", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rwkv-tokenizer", required=True)
    parser.add_argument("--rwkv-vocab", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "generations.jsonl"
    accepted_path = output_dir / "accepted_rwkv_sft.jsonl"
    summary_path = output_dir / "summary.json"

    source_rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit > 0:
        source_rows = source_rows[: args.limit]

    result_rows = []
    completed_ids = set()
    if result_path.exists():
        for line in result_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                result_rows.append(row)
                completed_ids.add(row["unique_id"])
    pending = [row for row in source_rows if row.get("unique_id") not in completed_ids]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rwkv_tokenizer = load_rwkv_tokenizer(args.rwkv_tokenizer, args.rwkv_vocab)
    engine = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.92,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
    )

    started = time.time()
    print(
        f"loaded={len(source_rows)} resume={len(result_rows)} pending={len(pending)} "
        f"chunk={args.chunk} max_num_seqs={args.max_num_seqs}",
        flush=True,
    )
    for offset in range(0, len(pending), args.chunk):
        chunk_rows = pending[offset : offset + args.chunk]
        prompts = []
        sampling = []
        for index, row in enumerate(chunk_rows):
            content = (
                "Solve the following math problem. Make sure to put the answer "
                "(and only answer) inside \\boxed{}.\n\n" + row["problem"].strip()
            )
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            prompts.append(prompt)
            sampling.append(
                SamplingParams(
                    temperature=0.6,
                    top_p=0.95,
                    max_tokens=max(1, min(args.max_new, 8192 - prompt_tokens)),
                    seed=args.seed + len(result_rows) + index,
                )
            )

        chunk_started = time.time()
        outputs = engine.generate(prompts, sampling, use_tqdm=False)
        chunk_elapsed = time.time() - chunk_started
        new_results = []
        new_accepted = []
        for source, output in zip(chunk_rows, outputs):
            candidate = output.outputs[0]
            completion = candidate.text.strip()
            generated_tokens = len(candidate.token_ids)
            is_correct = correct(str(source["answer"]), completion)
            is_truncated = candidate.finish_reason == "length"
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
                "finish_reason": candidate.finish_reason,
                "elapsed_chunk_sec": chunk_elapsed,
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
        current_summary = summary(result_rows, len(source_rows), started)
        summary_path.write_text(json.dumps(current_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"done={len(result_rows)}/{len(source_rows)} correct={sum(row['correct'] for row in result_rows)} "
            f"accepted={current_summary['accepted']} chunk_s={chunk_elapsed:.1f} "
            f"process_tps={current_summary['tokens_per_sec_current_process']:.1f}",
            flush=True,
        )

    summary_path.write_text(json.dumps(summary(result_rows, len(source_rows), started), ensure_ascii=False, indent=2), encoding="utf-8")
    print("ROLLOUT_DONE", flush=True)


if __name__ == "__main__":
    main()
