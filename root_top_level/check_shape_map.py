import os, sys, types
from pathlib import Path
BASE = Path('/root/RWKV-LM/RWKV-v7/train_temp')
BASELINE = Path('/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1')
os.chdir(str(BASE))
os.environ['RWKV_MY_TESTING'] = 'x070'
os.environ['RWKV_CTXLEN'] = '4096'
os.environ['RWKV_HEAD_SIZE'] = '64'
os.environ['RWKV_FLOAT_MODE'] = 'bf16'
os.environ['RWKV_JIT_ON'] = '0'
import torch
for p in [str(BASE), str(BASELINE)]:
    if p not in sys.path:
        sys.path.insert(0, p)
from train_rl_baseline import _torch_load_weights, _normalize_state_dict, _infer_arch, PaddedRWKV
from reference.rwkv7 import RWKV_x070
ckpt = '/root/autodl-tmp/rwkv_models/rwkv7-g1e-1.5b-20260309-ctx8192.pth'
sd = _normalize_state_dict(_torch_load_weights(ckpt))
class Args: pass
args = Args()
args.n_layer, args.n_embd, args.vocab_size, args.dim_ffn = _infer_arch(sd)
args.dim_att = args.n_embd
args.head_size = 64
args.ctx_len = 4096
args.my_testing = 'x070'
args.dropout = 0.0
args.grad_cp = 0
m = PaddedRWKV(args)
m.load_state_dict(sd, strict=True)
rollout_args = types.SimpleNamespace(MODEL_NAME=str(Path(ckpt).with_suffix('')), vocab_size=args.vocab_size)
r = RWKV_x070(rollout_args)
keys = [
    ('blocks.0.att.x_r', m.blocks[0].att.x_r.shape, r.z['blocks.0.att.x_r'].shape),
    ('blocks.0.att.w0', m.blocks[0].att.w0.shape, r.z['blocks.0.att.w0'].shape),
    ('blocks.0.att.w1', m.blocks[0].att.w1.shape, r.z['blocks.0.att.w1'].shape),
    ('blocks.0.att.a0', m.blocks[0].att.a0.shape, r.z['blocks.0.att.a0'].shape),
    ('blocks.0.att.v0', m.blocks[0].att.v0.shape, r.z['blocks.0.att.v0'].shape),
    ('blocks.0.att.g1', m.blocks[0].att.g1.shape, r.z['blocks.0.att.g1'].shape),
    ('blocks.0.att.k_k', m.blocks[0].att.k_k.shape, r.z['blocks.0.att.k_k'].shape),
    ('blocks.0.att.r_k', m.blocks[0].att.r_k.shape, r.z['blocks.0.att.r_k'].shape),
    ('blocks.0.att.receptance.weight', m.blocks[0].att.receptance.weight.shape, r.z['blocks.0.att.receptance.weight'].shape),
    ('blocks.0.ffn.x_k', m.blocks[0].ffn.x_k.shape, r.z['blocks.0.ffn.x_k'].shape),
    ('blocks.0.ffn.key.weight', m.blocks[0].ffn.key.weight.shape, r.z['blocks.0.ffn.key.weight'].shape),
    ('emb.weight', m.emb.weight.shape, r.z['emb.weight'].shape),
    ('ln_out.weight', m.ln_out.weight.shape, r.z['ln_out.weight'].shape),
    ('head.weight', m.head.weight.shape, r.z['head.weight'].shape),
]
for name, a, b in keys:
    print(name, 'train', tuple(a), 'roll', tuple(b))
