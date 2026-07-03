import torch, pprint
path='/root/autodl-tmp/log/verl_llama_math500_smoke_vllm_tuned_20260403_044118/global_step_4/data.pt'
obj=torch.load(path, map_location='cpu')
s=obj['_snapshot']
print(type(s))
if isinstance(s, dict):
    print(s.keys())
    for k,v in s.items():
        print('KEY',k,'TYPE',type(v))
        if isinstance(v, list):
            print('LEN',len(v))
            if len(v): print('FIRST_TYPE',type(v[0]))
