path='/root/RWKV-LM/RWKV-v7/train_temp/train_rl_baseline.py'
keys=['pending_extra','_pending_extra_batch','run_extra_only','step_type =']
with open(path,'r',encoding='utf-8') as f:
    for i,line in enumerate(f,1):
        if any(k in line for k in keys):
            print(f'{i}: {line.rstrip()}')

