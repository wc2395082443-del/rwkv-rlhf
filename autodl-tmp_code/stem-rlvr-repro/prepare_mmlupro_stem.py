import argparse, json, random
from pathlib import Path
from datasets import Dataset, DatasetDict

STEM_CATEGORIES = {
    "math", "physics", "chemistry", "biology", "computer science", "engineering"
}
PROMPTS = {
    "cot": "Please reason step by step, and put your final answer choice within \\boxed{} (for example, \\boxed{A}).",
    "brief": "Choose the correct option. Think briefly if needed, but keep the solution short. End your response exactly with the final answer in \\boxed{} (for example, \\boxed{A}).",
    "answer_only": "Choose the correct option. Output only the final answer in \\boxed{} (for example, \\boxed{A}).",
}


def build_prompt(raw: str, style: str) -> str:
    return raw.strip() + "\n\n" + PROMPTS[style]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/root/autodl-tmp/ext/OPD/datasets/test_data/MMLU-Pro/test.json")
    ap.add_argument("--out", default="/root/autodl-tmp/stem-rlvr-repro/data/mmlupro_stem_trl")
    ap.add_argument("--prompt_style", default="cot", choices=list(PROMPTS))
    ap.add_argument("--eval_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    items = json.load(open(args.input))
    rows = []
    for i, x in enumerate(items):
        cat = str(x.get("category", "")).lower()
        if cat not in STEM_CATEGORIES:
            continue
        answer = str(x["answer"]).strip().upper()
        prompt_text = build_prompt(x["prompt"], args.prompt_style)
        rows.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "prompt_text": prompt_text,
            "solution": answer,
            "answer": answer,
            "category": cat,
            "option_num": int(x.get("option_num", 0) or 0),
            "source_index": i,
            "prompt_style": args.prompt_style,
        })

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_eval = int(round(len(rows) * args.eval_ratio))
    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = DatasetDict({"train": Dataset.from_list(train_rows), "eval": Dataset.from_list(eval_rows)})
    ds.save_to_disk(str(out))
    for split, arr in [("train", train_rows), ("eval", eval_rows)]:
        with open(out / f"{split}.jsonl", "w") as f:
            for r in arr:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print({"total_stem": len(rows), "train": len(train_rows), "eval": len(eval_rows), "out": str(out), "prompt_style": args.prompt_style})
    from collections import Counter
    print("train_cat", Counter(r["category"] for r in train_rows))
    print("eval_cat", Counter(r["category"] for r in eval_rows))

if __name__ == "__main__":
    main()
