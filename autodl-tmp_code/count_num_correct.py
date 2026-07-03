import json
from collections import Counter
path = "/root/autodl-tmp/base_gsm8k_real_pass8_rolloutparams_20260422/pass8_eval.jsonl"
c = Counter()
total = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        c[int(obj["num_correct"])] += 1
        total += 1
print("total", total)
for k in range(9):
    v = c.get(k, 0)
    print(k, v, f"{v/total:.6f}")

