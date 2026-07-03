path = '/root/RWKV-LM/RWKV-v7/train_temp/train_rl_baseline.py'
keys = ['hard_triggered', 'hard_buffer_target_samples', 'needed_questions', '_pop_hard_batch']
with open(path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        if any(k in line for k in keys):
            print(f'{i}: {line.rstrip()}')

