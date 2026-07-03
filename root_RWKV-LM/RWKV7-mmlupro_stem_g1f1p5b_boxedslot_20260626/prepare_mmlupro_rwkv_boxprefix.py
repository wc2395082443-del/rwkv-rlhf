import json
from pathlib import Path

SRC = Path("/root/autodl-tmp/stem-rlvr-repro/data/mmlupro_stem_answeronly_trl")
OUT = Path("/root/RWKV-LM/RWKV7-mmlupro_stem_g1f1p5b_20260624/data_mmlupro_stem_boxprefix")
OUT.mkdir(parents=True, exist_ok=True)

for split in ["train", "eval"]:
    in_path = SRC / f"{split}.jsonl"
    out_name = "mmlupro_stem_train_rwkv.jsonl" if split == "train" else "mmlupro_stem_eval_rwkv.jsonl"
    out_path = OUT / out_name
    n = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            prompt_text = row.get("prompt_text") or row["prompt"][0]["content"]
            # Force classification-style completion for RWKV; generated text only needs to complete "A}".
            problem = "User: " + prompt_text.strip() + "\n\nAssistant: \\boxed{"
            answer = str(row.get("answer") or row.get("solution")).strip().upper()
            obj = {
                "id": row.get("source_index", n),
                "problem": problem,
                "answer": answer,
                "solution": answer,
                "category": row.get("category"),
                "option_num": row.get("option_num"),
                "prompt_style": "rwkv_boxprefix",
            }
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    print(split, n, out_path)
