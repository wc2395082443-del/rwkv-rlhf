#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import re
import time
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
import pandas as pd
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except Exception:
    zstd = None
    ZSTD_AVAILABLE = False

try:
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
    MATH_VERIFY_AVAILABLE = True
except Exception:
    ExprExtractionConfig = None
    LatexExtractionConfig = None
    parse = None
    verify = None
    MATH_VERIFY_AVAILABLE = False


SYSTEM_PROMPT = (
    "You are a careful math solver. Solve the problem step by step. "
    "End your response with exactly one final answer in the form The answer is \\boxed{...}."
)


BOXED_RE = re.compile(r"\\boxed\{", re.DOTALL)
FALLBACK_NUM_RE = re.compile(r"-?(?:\d+\.\d+|\d+|\.\d+)(?:/\d+)?")
TEXT_WRAPPER_RE = re.compile(r"\\text\{([^{}]*)\}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--eval_jsonl", type=str, default=None)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_eval_samples", type=int, default=None)
    ap.add_argument("--max_prompt_length", type=int, default=768)
    ap.add_argument("--max_completion_length", type=int, default=768)

    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--logging_steps", type=int, default=1)
    ap.add_argument("--eval_steps", type=int, default=0)

    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--num_iterations", type=int, default=1)
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--num_generations", type=int, default=8)

    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--generation_batch_size", type=int, default=None)

    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--disable_chat_template", action="store_true")
    ap.add_argument("--eval_before", action="store_true")
    ap.add_argument("--eval_after", action="store_true")
    ap.add_argument("--eval_max_new_tokens", type=int, default=768)
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--periodic_eval_steps", type=int, default=50)
    ap.add_argument("--reward_correct", type=float, default=1.0)
    ap.add_argument("--reward_incorrect", type=float, default=0.0)
    return ap.parse_args()


def read_records(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    lower = path.lower()
    if lower.endswith('.jsonl') or lower.endswith('.json'):
        items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
                if limit is not None and len(items) >= limit:
                    break
        return items
    if lower.endswith('.parquet'):
        df = pd.read_parquet(path)
        if limit is not None:
            df = df.head(limit)
        return df.to_dict(orient='records')
    raise ValueError(f'Unsupported dataset format: {path}')


def build_prompt(problem: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Problem:\n{problem}\n\n"
        "Please reason carefully and finish with: The answer is \\boxed{your_answer}."
    )


def normalize_prompt_record(item: Dict[str, Any], use_chat_template: bool) -> Any:
    if "prompt" in item and item["prompt"] is not None:
        prompt = item["prompt"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if isinstance(prompt, list):
            normalized = []
            for turn in prompt:
                if isinstance(turn, dict):
                    normalized.append({"role": str(turn.get("role", "user")), "content": str(turn.get("content", ""))})
                else:
                    normalized.append({"role": "user", "content": str(turn)})
            return normalized if use_chat_template else "\n\n".join(x["content"] for x in normalized)
        return str(prompt)
    problem = str(item.get("problem", ""))
    if use_chat_template:
        return [{"role": "user", "content": build_prompt(problem)}]
    return build_prompt(problem)


def extract_boxed(text: str) -> str:
    last = None
    for m in BOXED_RE.finditer(text):
        start = m.end()
        depth = 1
        i = start
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last = text[start:i]
                    break
            i += 1
    return (last or "").strip()


def normalize_answer(text: str) -> str:
    s = text.strip()
    s = s.replace("$", "")
    s = s.replace(",", "")
    s = s.replace(" ", "")
    s = s.replace("\n", "")
    s = s.replace("\\,", "")
    s = s.replace("\\!", "")
    s = s.rstrip(".")
    return s.lower()


def extract_final_answer(text: str) -> str:
    boxed = extract_boxed(text)
    if boxed:
        return normalize_answer(boxed)
    matches = FALLBACK_NUM_RE.findall(text)
    if matches:
        return normalize_answer(matches[-1])
    return ""


def textify_completion(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def has_repeated_ngrams(text: str, n: int = 16, repeat: int = 5) -> bool:
    if not text or n <= 0 or repeat <= 1:
        return False
    tokens = re.findall(r"\w+|[^\w\s]", text)
    if len(tokens) < n * repeat:
        return False
    counts = {}
    for idx in range(len(tokens) - n + 1):
        gram = tuple(tokens[idx : idx + n])
        cnt = counts.get(gram, 0) + 1
        counts[gram] = cnt
        if cnt >= repeat:
            return True
    return False


def compute_zstd_ratio(text: str) -> float:
    data = text.encode("utf-8", errors="ignore")
    if not data:
        return 0.0
    if not ZSTD_AVAILABLE:
        return 0.0
    comp = zstd.ZstdCompressor(level=3).compress(data)
    return float(len(data) / max(1, len(comp)))


def strip_latex_text(text: str) -> str:
    prev = None
    cur = text
    while prev != cur:
        prev = cur
        cur = TEXT_WRAPPER_RE.sub(r"\1", cur)
    return cur


def answer_string_candidates(text: str) -> List[str]:
    raw = str(text or "").strip()
    cands: List[str] = []
    for item in [
        raw,
        strip_latex_text(raw),
        extract_boxed(raw),
        extract_final_answer(raw),
        strip_latex_text(extract_boxed(raw)),
    ]:
        norm = normalize_answer(item) if item else ""
        if norm and norm not in cands:
            cands.append(norm)
    return cands


def math_verify_match(pred_text: str, gold_text: str) -> bool:
    if not MATH_VERIFY_AVAILABLE:
        return False
    extraction_config = [LatexExtractionConfig(), ExprExtractionConfig()]
    pred_variants: List[str] = []
    gold_variants: List[str] = []
    for item in [pred_text, extract_boxed(pred_text), extract_final_answer(pred_text), strip_latex_text(pred_text)]:
        item = str(item or "").strip()
        if item and item not in pred_variants:
            pred_variants.append(item)
    for item in [gold_text, extract_boxed(gold_text), extract_final_answer(gold_text), strip_latex_text(gold_text)]:
        item = str(item or "").strip()
        if item and item not in gold_variants:
            gold_variants.append(item)
    for pred_variant in pred_variants:
        pred_parsed = parse(
            pred_variant,
            extraction_config=extraction_config,
            fallback_mode="first_match",
            extraction_mode="any_match",
            parsing_timeout=5,
            raise_on_error=False,
        )
        if not pred_parsed:
            continue
        for gold_variant in gold_variants:
            gold_parsed = parse(
                gold_variant,
                extraction_config=extraction_config,
                fallback_mode="first_match",
                extraction_mode="any_match",
                parsing_timeout=5,
                raise_on_error=False,
            )
            if gold_parsed and verify(
                gold_parsed,
                pred_parsed,
                strict=True,
                timeout_seconds=5,
                raise_on_error=False,
            ):
                return True
    return False


def judge_correct(pred_text: str, gold_text: str) -> bool:
    if math_verify_match(pred_text, gold_text):
        return True
    pred_candidates = answer_string_candidates(pred_text)
    gold_candidates = answer_string_candidates(gold_text)
    return any(pred == gold for pred in pred_candidates for gold in gold_candidates)


def _effective_gen_stats(gen_row: torch.Tensor, pad_token_id: int, eos_token_id: Optional[int], max_new_tokens: int) -> tuple[int, bool]:
    ids = gen_row.tolist()
    actual_len = len(ids)
    while actual_len > 0 and ids[actual_len - 1] == pad_token_id:
        actual_len -= 1
    truncated = True
    if eos_token_id is not None:
        for tok in ids[:actual_len]:
            if tok == eos_token_id:
                truncated = False
                break
    if actual_len < max_new_tokens:
        truncated = False
    return actual_len, truncated


def correctness_reward_factory(correct_reward: float, incorrect_reward: float):
    def reward_func(completions, solution, **kwargs):
        rewards: List[float] = []
        for completion, gold in zip(completions, solution):
            pred_text = textify_completion(completion)
            rewards.append(correct_reward if judge_correct(pred_text, str(gold)) else incorrect_reward)
        return rewards

    return reward_func


def make_dataset(records: List[Dict[str, Any]], use_chat_template: bool) -> Dataset:
    rows = []
    for item in records:
        rows.append(
            {
                "prompt": normalize_prompt_record(item, use_chat_template),
                "problem": item.get("problem", ""),
                "solution": item.get("solution", item.get("answer", item.get("reward_model", {}).get("ground_truth", ""))),
                "answer": item.get("answer", item.get("reward_model", {}).get("ground_truth", "")),
                "original_answer": item.get("original_answer", ""),
            }
        )
    return Dataset.from_list(rows)


@torch.inference_mode()
def evaluate_model(
    model,
    tokenizer,
    records: List[Dict[str, Any]],
    use_chat_template: bool,
    batch_size: int,
    max_new_tokens: int,
    save_path: Optional[str] = None,
    step: Optional[int] = None,
    split: str = "eval",
) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    total_len = 0
    total_trunc = 0
    total_repeat = 0
    total_no_answer = 0
    total_zstd = 0.0
    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        prompts = [normalize_prompt_record(x, use_chat_template) for x in batch]
        if use_chat_template:
            texts = [
                tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
                for p in prompts
            ]
        else:
            texts = prompts
        toks = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=getattr(tokenizer, "model_max_length", 2048),
        )
        toks = {k: v.to(model.device) for k, v in toks.items()}
        out = model.generate(
            **toks,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_len = toks["input_ids"].shape[1]
        gen_ids = out[:, prompt_len:]
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        for gen_row, text, ex in zip(gen_ids, texts, batch):
            gen_len, truncated = _effective_gen_stats(
                gen_row,
                tokenizer.pad_token_id,
                tokenizer.eos_token_id,
                max_new_tokens,
            )
            repeat_flag = has_repeated_ngrams(text, n=16, repeat=5)
            no_answer = len(answer_string_candidates(text)) == 0
            zstd_ratio = compute_zstd_ratio(text)
            is_correct = judge_correct(
                text,
                str(ex.get("solution", ex.get("answer", ex.get("reward_model", {}).get("ground_truth", "")))),
            )
            total += 1
            total_len += gen_len
            total_trunc += int(truncated)
            total_repeat += int(repeat_flag)
            total_no_answer += int(no_answer)
            total_zstd += zstd_ratio
            if is_correct:
                correct += 1
            if save_path is not None:
                rows.append(
                    {
                        "step": step,
                        "split": split,
                        "problem": ex.get("problem", ""),
                        "answer": ex.get("answer", ex.get("reward_model", {}).get("ground_truth", "")),
                        "response": text,
                        "is_correct": is_correct,
                        "truncated": bool(truncated),
                        "repeat_16gram_5": bool(repeat_flag),
                        "no_answer": bool(no_answer),
                        "gen_len": gen_len,
                        "zstd_ratio": zstd_ratio,
                    }
                )
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "step": step,
        "split": split,
        "accuracy": (correct / total) if total else 0.0,
        "acc": (correct / total) if total else 0.0,
        "total": total,
        "correct": correct,
        "avg_length": (total_len / total) if total else 0.0,
        "trunc_rate": (total_trunc / total) if total else 0.0,
        "repeat_rate": (total_repeat / total) if total else 0.0,
        "repeat_16gram_rate": (total_repeat / total) if total else 0.0,
        "no_answer_rate": (total_no_answer / total) if total else 0.0,
        "avg_zstd_ratio": (total_zstd / total) if total else 0.0,
        "eval_time": time.time() - t0,
    }


def dump_metrics(path: str, metrics: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=False) + "\n")


class PeriodicEvalCallback(TrainerCallback):
    def __init__(
        self,
        eval_records: List[Dict[str, Any]],
        tokenizer,
        use_chat_template: bool,
        batch_size: int,
        max_new_tokens: int,
        out_dir: str,
        every_steps: int,
    ):
        self.eval_records = eval_records
        self.tokenizer = tokenizer
        self.use_chat_template = use_chat_template
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.out_dir = out_dir
        self.every_steps = every_steps
        self.metrics_path = os.path.join(out_dir, "metrics.jsonl")
        self.eval_dir = os.path.join(out_dir, "eval_by_step")
        os.makedirs(self.eval_dir, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        if not self.eval_records or self.every_steps <= 0:
            return control
        if not state.is_world_process_zero:
            return control
        step = int(state.global_step)
        if step <= 0 or step % self.every_steps != 0:
            return control
        model = kwargs.get("model")
        save_path = os.path.join(self.eval_dir, f"eval_step_{step}.jsonl")
        metrics = evaluate_model(
            model=model,
            tokenizer=self.tokenizer,
            records=self.eval_records,
            use_chat_template=self.use_chat_template,
            batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens,
            save_path=save_path,
            step=step,
            split="full_eval",
        )
        dump_metrics(self.metrics_path, metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        model.train()
        return control


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        from trl import GRPOConfig, GRPOTrainer
    except Exception as e:
        raise RuntimeError(
            "TRL ???????: pip install trl"
        ) from e

    train_records = read_records(args.train_jsonl, args.max_train_samples)
    if not train_records:
        raise RuntimeError("?????")
    eval_records = read_records(args.eval_jsonl, args.max_eval_samples) if args.eval_jsonl else []

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError("tokenizer ?? pad_token_id")
    tokenizer.padding_side = "left"

    use_chat_template = bool(getattr(tokenizer, "chat_template", None)) and not args.disable_chat_template
    train_dataset = make_dataset(train_records, use_chat_template)
    eval_dataset = make_dataset(eval_records, use_chat_template) if eval_records else None

    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_init_kwargs = {
        "torch_dtype": model_dtype,
        "low_cpu_mem_usage": True,
    }

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[x.strip() for x in args.lora_target_modules.split(",") if x.strip()],
    )

    training_args = GRPOConfig(
        output_dir=args.out_dir,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        eval_strategy="steps" if (args.eval_steps and eval_dataset is not None) else "no",
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        report_to="none",
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=True,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        beta=args.beta,
        num_iterations=args.num_iterations,
        epsilon=args.epsilon,
        model_init_kwargs=model_init_kwargs,
        use_cache=False,
        log_completions=True,
        num_completions_to_print=2,
    )

    reward_func = correctness_reward_factory(args.reward_correct, args.reward_incorrect)
    callbacks = []
    if eval_records and args.periodic_eval_steps > 0:
        callbacks.append(
            PeriodicEvalCallback(
                eval_records=eval_records,
                tokenizer=tokenizer,
                use_chat_template=use_chat_template,
                batch_size=args.eval_batch_size,
                max_new_tokens=args.eval_max_new_tokens,
                out_dir=args.out_dir,
                every_steps=args.periodic_eval_steps,
            )
        )
    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        reward_funcs=reward_func,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    metrics_path = os.path.join(args.out_dir, "offline_eval.jsonl")
    periodic_metrics_path = os.path.join(args.out_dir, "metrics.jsonl")
    if args.eval_before and eval_records:
        pre = evaluate_model(
            model=trainer.model,
            tokenizer=tokenizer,
            records=eval_records,
            use_chat_template=use_chat_template,
            batch_size=args.eval_batch_size,
            max_new_tokens=args.eval_max_new_tokens,
            save_path=os.path.join(args.out_dir, "eval_by_step", "pre_eval_step_0.jsonl"),
            step=0,
            split="pre_eval",
        )
        pre["stage"] = "before"
        dump_metrics(metrics_path, pre)
        dump_metrics(periodic_metrics_path, pre)
        print(json.dumps(pre, ensure_ascii=False), flush=True)

    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    if args.eval_after and eval_records:
        post = evaluate_model(
            model=trainer.model,
            tokenizer=tokenizer,
            records=eval_records,
            use_chat_template=use_chat_template,
            batch_size=args.eval_batch_size,
            max_new_tokens=args.eval_max_new_tokens,
            save_path=os.path.join(args.out_dir, "eval_by_step", f"post_eval_step_{args.max_steps}.jsonl"),
            step=args.max_steps,
            split="post_eval",
        )
        post["stage"] = "after"
        dump_metrics(metrics_path, post)
        dump_metrics(periodic_metrics_path, post)
        print(json.dumps(post, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

