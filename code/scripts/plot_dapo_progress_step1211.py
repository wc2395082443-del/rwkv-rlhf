import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS = [
    "/hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_100_20260820_154937",
    "/hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue100to200_eval512_20260821_131710",
    "/hy-tmp/runs/g1i_step155_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue200to300_eval512_20260822_073045",
    "/hy-tmp/runs/g1i_step300_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue300to400_eval384_20260824_122246",
    "/hy-tmp/runs/g1i_step400_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue400to500_eval384_20260825_1355_fixed",
    "/hy-tmp/runs/g1i_step500_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue500to600_eval384_20260826_0000_fixed",
    "/hy-tmp/runs/g1i_step600_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue600to800_eval384_20260827",
    "/hy-tmp/runs/g1i_step800_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue800to1000_eval384_ckpt50_20260829_1810",
    "/hy-tmp/runs/g1i_step1000_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue1000to1200_eval384_ckpt50_20260901_1100",
    "/hy-tmp/runs/g1i_step1200_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue1200to1300_eval384_ckpt50_20260903_1600",
]


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def moving_average(values, window=10):
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    if len(values) >= window:
        out[window - 1 :] = np.convolve(values, np.ones(window) / window, mode="valid")
    return out


def padded_limits(values, minimum_zero=False):
    values = np.asarray(values, dtype=float)
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    pad = max((high - low) * 0.12, 0.6 if high < 100 else 30.0)
    if minimum_zero:
        low = max(0.0, low - pad)
    else:
        low -= pad
    return low, high + pad


def plot_series(ax, steps, values, title, ylabel, percent=False, color="#78b7d2"):
    scale = 100.0 if percent else 1.0
    raw = np.asarray(values, dtype=float) * scale
    ax.plot(steps, raw, color=color, alpha=0.34, linewidth=0.75, label="raw")
    ax.plot(steps, moving_average(raw), color="#ef4136", linewidth=2.4, label="MA10")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.18, linestyle="--")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(*padded_limits(raw, minimum_zero=True))


def add_eval(ax, eval_rows, value_key, percent=False):
    points = [
        (step, row[value_key])
        for step, row in sorted(eval_rows.items())
        if step > 0 and value_key in row
    ]
    if not points:
        return
    scale = 100.0 if percent else 1.0
    x = np.asarray([point[0] for point in points])
    y = np.asarray([point[1] * scale for point in points])
    ax.plot(x, y, color="black", linewidth=1.1, marker="o", markersize=4.5, label="eval", zorder=5)
    y_span = ax.get_ylim()[1] - ax.get_ylim()[0]
    for index, (step, value) in enumerate(zip(x, y)):
        offset = y_span * (0.035 if index % 2 == 0 else -0.055)
        label = f"eval {step}: {value:.1f}"
        ax.text(step, value + offset, label, fontsize=7, color="black", ha="center", va="center")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper left", fontsize=8)


def main(output_path):
    train_by_step = {}
    eval_by_step = {}
    for run in RUNS:
        run_path = Path(run)
        for row in read_jsonl(run_path / "metrics.jsonl"):
            if row.get("split") == "train":
                train_by_step[int(row["step"])] = row
        for row in read_jsonl(run_path / "validation_metrics.jsonl"):
            eval_by_step[int(row["step"])] = row

    if not train_by_step:
        raise RuntimeError("No training metrics found")
    current_step = max(train_by_step)
    expected = set(range(1, current_step + 1))
    missing = sorted(expected - set(train_by_step))
    if missing:
        raise RuntimeError(f"Incomplete metrics: missing {len(missing)} steps, first={missing[:10]}")

    rows = [train_by_step[step] for step in range(1, current_step + 1)]
    steps = np.asarray([row["step"] for row in rows])
    fig, axes = plt.subplots(3, 2, figsize=(18, 11.1), sharex=True)

    plot_series(axes[0, 0], steps, [row["avg_reward"] for row in rows], "avg_reward", "percent (%)", True)
    plot_series(axes[0, 1], steps, [row["accuracy"] for row in rows], "accuracy", "percent (%)", True)
    add_eval(axes[0, 1], eval_by_step, "strict_accuracy", True)

    selected_acc = np.asarray([row["selected_accuracy"] for row in rows]) * 100.0
    selected_strict = np.asarray([row["selected_strict_accuracy"] for row in rows]) * 100.0
    axes[1, 0].plot(steps, selected_acc, color="#78b7d2", alpha=0.28, linewidth=0.7, label="selected_acc raw")
    axes[1, 0].plot(steps, moving_average(selected_acc), color="#ef4136", linewidth=2.2, label="selected_acc MA10")
    axes[1, 0].plot(steps, selected_strict, color="#75c6b6", alpha=0.28, linewidth=0.7, label="selected_strict raw")
    axes[1, 0].plot(steps, moving_average(selected_strict), color="#ed8b19", linewidth=2.2, label="selected_strict MA10")
    axes[1, 0].set_title("selected_acc + selected_strict", fontsize=14, fontweight="bold")
    axes[1, 0].set_ylabel("percent (%)")
    axes[1, 0].grid(True, alpha=0.18, linestyle="--")
    axes[1, 0].set_ylim(*padded_limits(np.concatenate([selected_acc, selected_strict]), minimum_zero=True))
    axes[1, 0].legend(loc="upper left", fontsize=8)
    add_eval(axes[1, 0], eval_by_step, "strict_accuracy", True)

    plot_series(axes[1, 1], steps, [row["avg_length"] for row in rows], "avg_length", "generated tokens", False)
    add_eval(axes[1, 1], eval_by_step, "mean_generated_tokens", False)
    plot_series(axes[2, 0], steps, [row["repeat_rate"] for row in rows], "repeat_rate", "percent (%)", True)
    plot_series(axes[2, 1], steps, [row["trunc_rate"] for row in rows], "trunc_rate", "percent (%)", True)
    add_eval(axes[2, 1], eval_by_step, "truncated_rate", True)

    for ax in axes[2, :]:
        ax.set_xlabel("global step")

    eval_steps = sorted(step for step in eval_by_step if 0 < step <= current_step)
    latest_eval = max(eval_steps) if eval_steps else "none"
    title = (
        "DAPO RL results | official-style boxed prompt (User/Assistant wrapper) | "
        "q32 x 16 | candidate64 | max_new6144 | overlong_buffer512 | eval batch384"
    )
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.958,
        f"global steps 1-{current_step} | missing_count={len(missing)} | "
        f"eval points={len(eval_steps)} | latest eval step{latest_eval}",
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0.025, 0.025, 0.995, 0.935))
    fig.savefig(output_path, dpi=190, format="jpg")
    print(json.dumps({"current_step": current_step, "missing_count": len(missing), "eval_steps": eval_steps, "output": output_path}))


if __name__ == "__main__":
    main(sys.argv[1])
