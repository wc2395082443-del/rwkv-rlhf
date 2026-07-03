import json, os
paths = [
'/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1/log/hb_decoupled_negw06_ttl4_cd4_20260305_193559/metrics.jsonl',
'/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1/log/hb_decoupled_negw06_ttl4_cd4_repro_20260312_192125/metrics.jsonl',
'/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1/log/hb_decoupled_negw06_ttl4_cd4_repro_k3loss_20260312_214201/metrics.jsonl',
]
for path in paths:
    print('===', path)
    if not os.path.exists(path):
        print('missing')
        continue
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            print(obj)
            if i >= 2:
                break

