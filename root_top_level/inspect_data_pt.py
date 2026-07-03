import torch
path='/root/autodl-tmp/log/verl_llama_math500_smoke_vllm_tuned_20260403_044118/global_step_4/data.pt'
obj=torch.load(path, map_location='cpu')
print(type(obj))
if isinstance(obj, dict):
    print(obj.keys())
    for k,v in obj.items():
        print('KEY',k,'TYPE',type(v))
