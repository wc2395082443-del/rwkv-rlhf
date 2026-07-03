from pathlib import Path
path = Path('/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1/plot_metrics.py')
text = path.read_text(encoding='utf-8')
old = """    # 1. Accuracy\n    if train_acc_steps:\n        axes[0].plot(train_acc_steps, train_acc, label=\"train_acc\")\n    if eval_acc_steps:\n        axes[0].plot(eval_acc_steps, eval_acc, label=\"eval_acc\")\n    for name, series in full_eval.items():\n"""
new = """    # 1. Accuracy\n    # Do not plot train_acc here: for RL it is a noisy on-policy training statistic\n    # and is not directly comparable to full-eval accuracy.\n    if eval_acc_steps:\n        axes[0].plot(eval_acc_steps, eval_acc, label=\"eval_acc\")\n    if eval_preeval_steps:\n        axes[0].plot(eval_preeval_steps, eval_preeval_acc, label=\"preeval_acc\", linestyle=\"--\", color=\"tab:gray\")\n    for name, series in full_eval.items():\n"""
if old not in text:
    raise SystemExit('target block not found')
text = text.replace(old, new, 1)
path.write_text(text, encoding='utf-8')
print('patched', path)

