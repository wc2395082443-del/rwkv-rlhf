path='/root/RWKV-LM/RWKV-v7/train_temp/train_rl_baseline.py'
keys=['save_interval','save_model','checkpoint','torch.save','save_ckpt','save(']
with open(path,'r',encoding='utf-8') as f:
    for i,line in enumerate(f,1):
        if any(k in line for k in keys):
            print(f'{i}: {line.rstrip()}')

