import os, json
sets = {
    'stable_baseline': [
        '/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1/log/hb_decoupled_negw06_ttl4_cd4_20260305_193559/metrics.jsonl',
        '/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1/log/hb_decoupled_negw06_ttl4_cd4_repro_20260312_192125/metrics.jsonl',
        '/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1/log/hb_decoupled_negw06_ttl4_cd4_repro_k3loss_20260312_214201/metrics.jsonl',
    ],
    'k1_unstable': [
        '/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1/log/kl10_restart_20260307_191651/k1_reward_kl0.01/metrics.jsonl',
        '/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1/log/kl10_restart_20260307_191651/k1_reward_kl0.03/metrics.jsonl',
        '/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1/log/kl10_restart_20260308_012833/k1_reward_kl0.03/metrics.jsonl',
        '/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1/log/kl10_restart_20260308_012833/k1_reward_kl0.05/metrics.jsonl',
    ],
}
for name, paths in sets.items():
    rows = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                if obj.get('split') != 'train' or int(obj.get('extra_step_ran', 0)) != 1:
                    continue
                rows.append(obj)
    print('===', name, 'rows', len(rows))
    if not rows:
        continue
    for key in ['extra_avg_kl', 'kl', 'extra_grad_norm', 'avg_kl']:
        vals = [float(r.get(key, 0.0) or 0.0) for r in rows]
        vals.sort()
        def pct(p):
            i = min(len(vals)-1, int(round((len(vals)-1)*p)))
            return vals[i]
        print(key, 'p50', pct(0.5), 'p90', pct(0.9), 'p95', pct(0.95), 'max', max(vals))
    top = max(rows, key=lambda r: float(r.get('extra_avg_kl',0.0) or 0.0))
    print('top_step', top.get('step'), 'top_extra_avg_kl', top.get('extra_avg_kl'), 'path', paths[0] if len(paths)==1 else 'multiple')

