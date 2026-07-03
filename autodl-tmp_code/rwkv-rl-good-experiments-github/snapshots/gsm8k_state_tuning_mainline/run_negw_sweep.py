#!/root/miniconda3/bin/python3
import json
import os
import shutil
import subprocess
import time
from datetime import datetime

ROOT = "/root/RWKV-LM/RWKV7-statetuning"
BASE_LOG = os.path.join(ROOT, "log")
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
SWEEP_DIR = os.path.join(BASE_LOG, f"negw_sweep_{TS}")
os.makedirs(SWEEP_DIR, exist_ok=True)

summary_path = os.path.join(SWEEP_DIR, "summary_step100_full_eval.csv")
progress_path = os.path.join(SWEEP_DIR, "progress.log")
latest_path = os.path.join(BASE_LOG, "negw_sweep_latest.txt")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("neg_w,run_dir,acc,repeat,zstd,trunc,no_answer,status\\n")

with open(latest_path, "w", encoding="utf-8") as f:
    f.write(SWEEP_DIR + "\\n")

def log(msg: str):
    line = f"[{datetime.now().strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(line + "\\n")

log(f"START sweep_dir={SWEEP_DIR}")

for i in range(1, 11):
    neg_w = round(i / 10, 1)
    tag = str(neg_w).replace('.', 'p')
    out_dir = os.path.join(SWEEP_DIR, f"grpo_negw{tag}_100")
    os.makedirs(out_dir, exist_ok=True)
    log(f"START neg_w={neg_w} out={out_dir}")

    cmd = [
        "/root/miniconda3/bin/python3", os.path.join(ROOT, "main.py"),
        "--train_jsonl", os.path.join(ROOT, "gsm8k_train_formatted.jsonl"),
        "--eval_jsonl", os.path.join(ROOT, "gsm8k_test_formatted.jsonl"),
        "--model", "/root/RWKV-LM/rwkv7-g1b-1.5b-20251202-ctx8192.pth",
        "--tokenizer", "/root/RWKV-LM/rwkv_vocab_v20230424.txt",
        "--out_dir", out_dir,
        "--total_steps", "100",
        "--save_interval", "100",
        "--eval_interval", "5",
        "--eval_sample_ratio", "0.2",
        "--eval_top_k", "500",
        "--max_new_tokens", "1024",
        "--lr", "1e-4",
        "--ppo_epochs", "1",
        "--kl_coef", "0.05",
        "--neg_adv_weight", str(neg_w),
        "--zstd_threshold", "2.8",
        "--zstd_penalty_weight", "0.2",
    ]

    with open(os.path.join(out_dir, "train_stdout.log"), "w", encoding="utf-8") as fout:
        rc = subprocess.run(cmd, stdout=fout, stderr=subprocess.STDOUT).returncode

    status = "ok" if rc == 0 else f"train_rc_{rc}"
    acc = repeat = zstd = trunc = no_answer = ""

    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    if not os.path.isfile(metrics_path):
        status = "missing_metrics"
    else:
        row = None
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("split") == "full_eval" and int(d.get("step", -1)) == 100:
                    row = d
        if row is None:
            status = "missing_full_eval_step100"
        else:
            acc = row.get("accuracy", "")
            repeat = row.get("repeat_rate", row.get("repeat_16gram_rate", ""))
            zstd = row.get("avg_zstd_ratio", "")
            trunc = row.get("trunc_rate", "")
            no_answer = row.get("no_answer_rate", "")

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write(f"{neg_w},{out_dir},{acc},{repeat},{zstd},{trunc},{no_answer},{status}\\n")

    for sub in ["responses_by_step", "eval_by_step"]:
        p = os.path.join(out_dir, sub)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)

    for fn in os.listdir(out_dir):
        if fn.startswith("ckpt_step") and fn.endswith(".pth"):
            try:
                os.remove(os.path.join(out_dir, fn))
            except OSError:
                pass

    log(f"DONE neg_w={neg_w} status={status} acc={acc} repeat={repeat} zstd={zstd} trunc={trunc} no_answer={no_answer}")

log(f"ALL_DONE summary={summary_path}")
