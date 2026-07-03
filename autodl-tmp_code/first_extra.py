import json, os
paths = {
    'current_nobuffer': '/root/autodl-tmp/baseline_nobuffer_bestcfg_20260422_174502/run/metrics.jsonl',
    'old_best_autodltmp': '/root/autodl-tmp/baseline_bf16rollout_bf16_500_20260418_123934/run/metrics.jsonl',
    'old_hb_best_named': '/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1/log/hb_decoupled_negw06_ttl4_cd4_20260305_193559/metrics.jsonl',
    'old_hb_repro': '/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1/log/hb_decoupled_negw06_ttl4_cd4_repro_20260312_192125/metrics.jsonl',
}
for name, path in paths.items():
    print('===', name, path)
    if not os.path.exists(path):
        print('missing')
        continue
    first_extra = None
    first_trigger = None
    first_selected = None
    last_step = None
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r=json.loads(line)
            if r.get('split') != 'train':
                continue
            last_step = r.get('step')
            if first_extra is None and int(r.get('extra_step_ran',0) or 0) == 1:
                first_extra = r
            if first_trigger is None and int(r.get('hard_buffer_triggered',0) or 0) == 1:
                first_trigger = r
            if first_selected is None and int(r.get('hard_buffer_selected',0) or 0) > 0:
                first_selected = r
    print('last_step', last_step)
    for label, r in [('first_extra_step_ran', first_extra), ('first_hard_triggered', first_trigger), ('first_hard_selected', first_selected)]:
        if r is None:
            print(label, None)
        else:
            keys=['step','step_type','samples','accuracy','groups_total','groups_used','groups_all_wrong','groups_all_correct','hard_buffer_size','hard_buffer_added','hard_buffer_eligible','hard_buffer_selected','hard_buffer_triggered','extra_step_ran','extra_samples','extra_groups_total','extra_groups_used','extra_groups_all_wrong','extra_groups_all_correct','avg_kl','extra_avg_kl','grad_norm','extra_grad_norm']
            print(label, {k:r.get(k) for k in keys if k in r})

