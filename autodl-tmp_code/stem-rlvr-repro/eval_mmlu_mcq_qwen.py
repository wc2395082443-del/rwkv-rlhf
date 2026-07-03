import argparse, json, time
from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from mcq_reward import extract_choice


def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dataset_path", default="/root/autodl-tmp/stem-rlvr-repro/data/mmlupro_stem_trl")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--attn_implementation", default="sdpa")
    args = ap.parse_args()

    ds = load_from_disk(args.dataset_path)[args.split]
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    rows = list(ds)

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    no_parse = 0
    total = 0
    t0 = time.time()
    with open(out_path, "w") as f:
        for batch in batched(rows, args.batch_size):
            prompts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            gen_kwargs = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            if args.temperature and args.temperature > 0:
                gen_kwargs.update(dict(do_sample=True, temperature=args.temperature, top_p=args.top_p))
            else:
                gen_kwargs.update(dict(do_sample=False))
            with torch.no_grad():
                gen = model.generate(**enc, **gen_kwargs)
            input_len = enc.input_ids.shape[1]
            texts = tok.batch_decode(gen[:, input_len:], skip_special_tokens=True)
            for r, text in zip(batch, texts):
                pred = extract_choice(text)
                ok = pred == str(r["solution"]).strip().upper()
                correct += int(ok)
                no_parse += int(pred is None)
                total += 1
                f.write(json.dumps({
                    "source_index": r.get("source_index"),
                    "category": r.get("category"),
                    "gold": r["solution"],
                    "pred": pred,
                    "correct": ok,
                    "completion": text,
                }, ensure_ascii=False) + "\n")
    n = len(rows)
    summary = {"model": args.model_path, "split": args.split, "n": n, "acc": correct / max(n,1), "no_parse": no_parse / max(n,1), "elapsed_s": time.time()-t0, "out": str(out_path)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    with open(out_path.with_suffix(out_path.suffix + ".summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
