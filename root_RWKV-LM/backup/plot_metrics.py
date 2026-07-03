#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os


def read_jsonl(path):
    items = []
    with open(path,  r, encoding=utf-8) as f:
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
    tag = it.get(tag)
    split = it.get(split)
    return tag if tag is not None else split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(--metrics, required=True, help=metrics.jsonl path)
    ap.add_argument(--out, required=True, help=output png)
    args = ap.parse_args()

    if not os.path.isfile(args.metrics):
        return

    items = read_jsonl(args.metrics)
    if not items:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    train_acc_steps, train_acc = [], []
    eval_acc_steps, eval_acc = [], []
    train_trunc_steps, train_trunc = [], []
    eval_trunc_steps, eval_trunc = [], []
    train_len_steps, train_len = [], []
    eval_len_steps, eval_len = [], []
    train_reward_steps, train_reward = [], []
    eval_reward_steps, eval_reward = [], []
    full_eval = {full_pre: [], full_ckpt: [], full_post: [], pre_eval: []}

    for it in items:
        step = it.get(step)
        if step is None:
            continue

        kind = _get_kind(it)
        acc = _get_val(it, accuracy, acc)
        trunc = _get_val(it, trunc_rate)
        avg_len = _get_val(it, avg_length, avg_len)
        avg_reward = _get_val(it, avg_reward)

        if kind in (train, None):
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
        elif kind == eval:
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
        else:
            if kind in full_eval:
                if acc is not None:
                    full_eval[kind].append((step, acc))
            elif isinstance(kind, str) and kind.startswith(full_):
                if acc is not None:
                    full_eval.setdefault(kind, []).append((step, acc))

    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

    # Accuracy
    if train_acc_steps:
        axes[0].plot(train_acc_steps, train_acc, label=train_acc)
    if eval_acc_steps:
        axes[0].plot(eval_acc_steps, eval_acc, label=eval_acc)
    for name, series in full_eval.items():
        if not series:
            continue
        xs = [p[0] for p in series]
        ys = [p[1] for p in series]
        axes[0].scatter(xs, ys, s=18, label=name)
    axes[0].set_ylabel(acc)
    axes[0].legend()

    # Reward
    if train_reward_steps:
        axes[1].plot(train_reward_steps, train_reward, label=train_reward)
    if eval_reward_steps:
        axes[1].plot(eval_reward_steps, eval_reward, label=eval_reward)
    axes[1].set_ylabel(avg_reward)
    axes[1].legend()

    # Truncation
    if train_trunc_steps:
        axes[2].plot(train_trunc_steps, train_trunc, label=train_trunc)
    if eval_trunc_steps:
        axes[2].plot(eval_trunc_steps, eval_trunc, label=eval_trunc)
    axes[2].set_ylabel(trunc_rate)
    axes[2].legend()

    # Length
    if train_len_steps:
        axes[3].plot(train_len_steps, train_len, label=train_len)
    if eval_len_steps:
        axes[3].plot(eval_len_steps, eval_len, label=eval_len)
    axes[3].set_ylabel(avg_length)
    axes[3].set_xlabel(step)
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(args.out, dpi=120)


if __name__ == __main__:
    main()
