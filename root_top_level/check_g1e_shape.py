import torch, os
p = '/root/autodl-tmp/rwkv_models/rwkv7-g1e-1.5b-20260309-ctx8192.pth'
sd = torch.load(p, map_location='cpu')
layers = max(int(k.split('.')[1]) for k in sd if k.startswith('blocks.')) + 1
print('exists', os.path.exists(p))
print('emb', tuple(sd['emb.weight'].shape))
print('head', tuple(sd['head.weight'].shape))
print('layers', layers)
print('n_embd', sd['emb.weight'].shape[1])
print('w1', tuple(sd['blocks.0.att.w1'].shape))
print('g1', tuple(sd['blocks.0.att.g1'].shape))
print('ffn', tuple(sd['blocks.0.ffn.key.weight'].shape))
print('recv', tuple(sd['blocks.0.att.receptance.weight'].shape))