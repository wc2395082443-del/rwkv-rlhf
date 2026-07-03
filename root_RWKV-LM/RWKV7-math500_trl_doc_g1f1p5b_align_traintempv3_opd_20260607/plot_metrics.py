#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os


def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def _get_val(it, *keys):
    for key in keys:
        if key in it and it[key] is not None:
            return it[key]
    return None


def _get_kind(it):
    tag = it.get("tag")
    split = it.get("split")
    return tag if tag is not None else split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="metrics.jsonl path")
    ap.add_argument("--out", required=True, help="output png")
    args = ap.parse_args()

    if not os.path.isfile(args.metrics):
        print(f"Error: File {args.metrics} not found.")
        return

    items = read_jsonl(args.metrics)
    if not items:
        print("Error: No valid data found in metrics file.")
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Error: matplotlib is not installed.")
        return

    # Initialize data lists
    train_acc_steps, train_acc = [], []
    eval_acc_steps, eval_acc = [], []

    eval_delta_steps, eval_acc_delta = [], []
    eval_preeval_steps, eval_preeval_acc = [], []

    train_trunc_steps, train_trunc = [], []
    eval_trunc_steps, eval_trunc = [], []

    train_repeat_steps, train_repeat = [], []
    eval_repeat_steps, eval_repeat = [], []

    train_len_steps, train_len = [], []
    eval_len_steps, eval_len = [], []

    train_reward_steps, train_reward = [], []
    eval_reward_steps, eval_reward = [], []

    train_entropy_steps, train_entropy = [], []
    eval_entropy_steps, eval_entropy = [], []

    train_zstd_ratio_steps, train_zstd_ratio = [], []
    eval_zstd_ratio_steps, eval_zstd_ratio = [], []
    train_zstd_penalty_steps, train_zstd_penalty = [], []
    eval_zstd_penalty_steps, eval_zstd_penalty = [], []

    eval_step_steps, eval_trunc_wrong = [], []
    eval_repeat16 = []
    eval_no_ans = []
    eval_corr_r = []
    eval_fmt_r = []

    full_eval = {
        "full_pre": [],
        "full_ckpt": [],
        "full_post": [],
        "pre_eval": [],
        "full_eval": [],
        "post_eval": [],
    }

    for it in items:
        step = it.get("step")
        if step is None:
            continue

        kind = _get_kind(it)
        acc = _get_val(it, "accuracy", "acc")
        trunc = _get_val(it, "trunc_rate")
        avg_len = _get_val(it, "avg_length", "avg_len")
        avg_reward = _get_val(it, "avg_reward")
        repeat_rate = _get_val(it, "repeat_rate", "repeat_16gram_rate")
        avg_entropy = _get_val(it, "avg_entropy")
        avg_zstd_ratio = _get_val(it, "avg_zstd_ratio")
        avg_zstd_penalty = _get_val(it, "avg_zstd_penalty")
        preeval_acc = _get_val(it, "preeval_acc")
        eval_acc_delta_val = _get_val(it, "eval_acc_delta")

        if kind in ("train", None):
            if acc is not None:
                train_acc_steps.append(step)
                train_acc.append(acc)
            if trunc is not None:
                train_trunc_steps.append(step)
                train_trunc.append(trunc)
            if avg_len is not None:
                train_len_steps.append(step)
                train_len.append(avg_len)
            if avg_reward is not None:
                train_reward_steps.append(step)
                train_reward.append(avg_reward)
            if repeat_rate is not None:
                train_repeat_steps.append(step)
                train_repeat.append(repeat_rate)
            if avg_entropy is not None:
                train_entropy_steps.append(step)
                train_entropy.append(avg_entropy)
            if avg_zstd_ratio is not None:
                train_zstd_ratio_steps.append(step)
                train_zstd_ratio.append(avg_zstd_ratio)
            if avg_zstd_penalty is not None:
                train_zstd_penalty_steps.append(step)
                train_zstd_penalty.append(avg_zstd_penalty)

        elif kind == "eval":
            if acc is not None:
                eval_acc_steps.append(step)
                eval_acc.append(acc)
            if trunc is not None:
                eval_trunc_steps.append(step)
                eval_trunc.append(trunc)
            if avg_len is not None:
                eval_len_steps.append(step)
                eval_len.append(avg_len)
            if avg_reward is not None:
                eval_reward_steps.append(step)
                eval_reward.append(avg_reward)
            if repeat_rate is not None:
                eval_repeat_steps.append(step)
                eval_repeat.append(repeat_rate)
            if avg_entropy is not None:
                eval_entropy_steps.append(step)
                eval_entropy.append(avg_entropy)
            if avg_zstd_ratio is not None:
                eval_zstd_ratio_steps.append(step)
                eval_zstd_ratio.append(avg_zstd_ratio)
            if avg_zstd_penalty is not None:
                eval_zstd_penalty_steps.append(step)
                eval_zstd_penalty.append(avg_zstd_penalty)
            if preeval_acc is not None:
                eval_preeval_steps.append(step)
                eval_preeval_acc.append(preeval_acc)
            if eval_acc_delta_val is not None:
                eval_delta_steps.append(step)
                eval_acc_delta.append(eval_acc_delta_val)

            eval_step_steps.append(step)
            eval_trunc_wrong.append(_get_val(it, "trunc_wrong_rate"))
            eval_repeat16.append(_get_val(it, "repeat_16gram_rate"))
            eval_no_ans.append(_get_val(it, "no_answer_rate"))
            eval_corr_r.append(_get_val(it, "avg_correct_reward"))
            eval_fmt_r.append(_get_val(it, "avg_format_reward"))
        else:
            if kind in full_eval:
                if acc is not None:
                    full_eval[kind].append((step, acc))
            elif isinstance(kind, str) and kind.startswith("full_"):
                if acc is not None:
                    full_eval.setdefault(kind, []).append((step, acc))

    fig, axes = plt.subplots(9, 1, figsize=(10, 26), sharex=True)

    # 1. Accuracy
    # Do not plot train_acc here: for RL it is a noisy on-policy training statistic
    # and is not directly comparable to full-eval accuracy.
    if eval_acc_steps:
        axes[0].plot(eval_acc_steps, eval_acc, label="eval_acc")
    if eval_preeval_steps:
        axes[0].plot(eval_preeval_steps, eval_preeval_acc, label="preeval_acc", linestyle="--", color="tab:gray")
    for name, series in full_eval.items():
        if not series:
            continue
        xs = [p[0] for p in series]
        ys = [p[1] for p in series]
        axes[0].scatter(xs, ys, s=24, label=name, marker='x')
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left')

    # 2. Eval Acc Delta vs pre_eval subset
    if eval_delta_steps:
        axes[1].plot(eval_delta_steps, eval_acc_delta, label="eval_acc_delta", color="tab:blue")
    axes[1].set_ylabel("Eval Acc Delta")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper left')

    # 3. Reward
    if train_reward_steps:
        axes[2].plot(train_reward_steps, train_reward, label="train_reward", color="orange")
    if eval_reward_steps:
        axes[2].plot(eval_reward_steps, eval_reward, label="eval_reward", color="darkorange", linestyle="--")
    axes[2].set_ylabel("Avg Reward")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper left')

    # 4. Truncation
    if train_trunc_steps:
        axes[3].plot(train_trunc_steps, train_trunc, label="train_trunc", color="red")
    if eval_trunc_steps:
        axes[3].plot(eval_trunc_steps, eval_trunc, label="eval_trunc", color="darkred", linestyle="--")
    axes[3].set_ylabel("Trunc Rate")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper left')

    # 5. Repeat rate
    if train_repeat_steps:
        axes[4].plot(train_repeat_steps, train_repeat, label="train_repeat", color="green")
    if eval_repeat_steps:
        axes[4].plot(eval_repeat_steps, eval_repeat, label="eval_repeat", color="darkgreen", linestyle="--")
    axes[4].set_ylabel("Repeat Rate")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(loc='upper left')

    # 6. Length
    if train_len_steps:
        axes[5].plot(train_len_steps, train_len, label="train_len", color="purple")
    if eval_len_steps:
        axes[5].plot(eval_len_steps, eval_len, label="eval_len", color="indigo", linestyle="--")
    axes[5].set_ylabel("Avg Length")
    axes[5].grid(True, alpha=0.3)
    axes[5].legend(loc='upper left')

    # 7. Entropy
    if train_entropy_steps:
        axes[6].plot(train_entropy_steps, train_entropy, label="train_entropy", color="teal")
    if eval_entropy_steps:
        axes[6].plot(eval_entropy_steps, eval_entropy, label="eval_entropy", color="darkcyan", linestyle="--")
    axes[6].set_ylabel("Avg Entropy")
    axes[6].grid(True, alpha=0.3)
    axes[6].legend(loc='upper left')

    # 8. Zstd
    if train_zstd_ratio_steps:
        axes[7].plot(train_zstd_ratio_steps, train_zstd_ratio, label="train_zstd_ratio", color="tab:blue")
    if eval_zstd_ratio_steps:
        axes[7].plot(eval_zstd_ratio_steps, eval_zstd_ratio, label="eval_zstd_ratio", color="tab:blue", linestyle="--")
    if train_zstd_penalty_steps:
        axes[7].plot(train_zstd_penalty_steps, train_zstd_penalty, label="train_zstd_penalty", color="tab:brown")
    if eval_zstd_penalty_steps:
        axes[7].plot(eval_zstd_penalty_steps, eval_zstd_penalty, label="eval_zstd_penalty", color="tab:brown", linestyle="--")
    axes[7].set_ylabel("Zstd")
    axes[7].grid(True, alpha=0.3)
    axes[7].legend(loc='upper left', ncol=2, fontsize='small')

    # 9. Detailed Eval Analysis
    if eval_step_steps:
        axes[8].plot(eval_step_steps, eval_trunc_wrong, label="trunc_wrong", alpha=0.7)
        axes[8].plot(eval_step_steps, eval_repeat16, label="repeat16@5", alpha=0.7)
        axes[8].plot(eval_step_steps, eval_no_ans, label="no_ans", alpha=0.7)
        axes[8].plot(eval_step_steps, eval_corr_r, label="corr_r", linestyle=":")
        axes[8].plot(eval_step_steps, eval_fmt_r, label="fmt_r", linestyle=":")
    axes[8].set_ylabel("Eval Metrics")
    axes[8].set_xlabel("Step")
    axes[8].grid(True, alpha=0.3)
    axes[8].legend(loc='upper left', ncol=2, fontsize='small')

    fig.tight_layout()
    print(f"Saving plot to {args.out}")
    fig.savefig(args.out, dpi=120)


if __name__ == "__main__":
    main()
