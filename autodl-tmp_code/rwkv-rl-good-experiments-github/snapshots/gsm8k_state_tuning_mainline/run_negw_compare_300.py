#!/root/miniconda3/bin/python3
import csv
import json
import os
import shutil
import subprocess
from datetime import datetime

ROOT = "/root/RWKV-LM/RWKV7-statetuning"
LOG_ROOT = os.path.join(ROOT, "log")
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ROOT = os.path.join(LOG_ROOT, f"negw_compare_06_vs_10_300_{TS}")
os.makedirs(RUN_ROOT, exist_ok=True)

summary_csv = os.path.join(RUN_ROOT, "summary_step300_full_eval.csv")
progress_log = os.path.join(RUN_ROOT, "progress.log")
latest_txt = os.path.join(LOG_ROOT, "negw_compare_latest.txt")

with open(summary_csv, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["neg_w", "run_dir", "step300_acc", "repeat", "zstd", "trunc", "no_answer", "status"])

with open(latest_txt, "w", encoding="utf-8") as f:
    f.write(RUN_ROOT + "\n")

def log(msg: str):
    line = f"[{datetime.now().strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(progress_log, "a", encoding="utf-8") as f:
        f.write(line + "\n")

base_args = [
    "/root/miniconda3/bin/python3", os.path.join(ROOT, "main.py"),
    "--train_jsonl", os.path.join(ROOT, "gsm8k_train_formatted.jsonl"),
    "--eval_jsonl", os.path.join(ROOT, "gsm8k_test_formatted.jsonl"),
    "--model", "/root/RWKV-LM/rwkv7-g1b-1.5b-20251202-ctx8192.pth",
    "--tokenizer", "/root/RWKV-LM/rwkv_vocab_v20230424.txt",
    "--total_steps", "300",
    "--save_interval", "300",
    "--eval_interval", "5",
    "--eval_sample_ratio", "0.2",
    "--eval_top_k", "500",
    "--max_new_tokens", "1024",
    "--lr", "1e-4",
    "--ppo_epochs", "1",
    "--kl_coef", "0.05",
    "--zstd_threshold", "2.8",
    "--zstd_penalty_weight", "0.2",
]

for neg_w in [0.6, 1.0]:
    tag = str(neg_w).replace('.', 'p')
    out_dir = os.path.join(RUN_ROOT, f"grpo_negw{tag}_300")
    os.makedirs(out_dir, exist_ok=True)
    log(f"START neg_w={neg_w} out={out_dir}")

    cmd = base_args + ["--out_dir", out_dir, "--neg_adv_weight", str(neg_w)]
    with open(os.path.join(out_dir, "train_stdout.log"), "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode

    status = "ok" if rc == 0 else f"train_rc_{rc}"
    acc = repeat = zstd = trunc = no_answer = ""

    metrics = os.path.join(out_dir, "metrics.jsonl")
    if not os.path.isfile(metrics):
        status = "missing_metrics"
    else:
        row = None
        with open(metrics, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("split") == "full_eval" and int(d.get("step", -1)) == 300:
                    row = d
        if row is None:
            status = "missing_full_eval_step300"
        else:
            acc = row.get("accuracy", "")
            repeat = row.get("repeat_rate", row.get("repeat_16gram_rate", ""))
            zstd = row.get("avg_zstd_ratio", "")
            trunc = row.get("trunc_rate", "")
            no_answer = row.get("no_answer_rate", "")

    with open(summary_csv, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([neg_w, out_dir, acc, repeat, zstd, trunc, no_answer, status])

    # save disk while preserving logs/metrics/plots
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

log(f"ALL_DONE summary={summary_csv}")
