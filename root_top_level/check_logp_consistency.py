import os, sys, types, json
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
import torch.nn.functional as F
for p in [str(BASE), str(BASELINE)]:
    if p not in sys.path:
        sys.path.insert(0, p)
from train_rl_baseline import _torch_load_weights, _normalize_state_dict, _infer_arch, PaddedRWKV
from reference.utils import TRIE_TOKENIZER
from reference.rwkv7 import RWKV_x070
ckpt = '/root/autodl-tmp/rwkv_models/rwkv7-g1e-1.5b-20260309-ctx8192.pth'
vocab = '/root/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt'
train_jsonl = '/root/autodl-tmp/data/gsm8k/gsm8k_train_formatted.jsonl'
tok = TRIE_TOKENIZER(vocab)
encode = lambda s: tok.encode(s)
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
train_model = PaddedRWKV(args)
train_model.load_state_dict(sd, strict=True)
train_model = train_model.to('cuda').bfloat16().eval()
rollout_args = types.SimpleNamespace(MODEL_NAME=str(Path(ckpt).with_suffix('')), vocab_size=args.vocab_size)
rollout_model = RWKV_x070(rollout_args)
def build_prompt(problem: str):
    return f'User: {problem}\nAssistant:'
with open(train_jsonl, 'r', encoding='utf-8') as f:
    first = json.loads(next(f))
problem = first['problem']
prompt = build_prompt(problem)
prompt_ids = encode(prompt)
suffix_ids = encode(' Let us solve this step by step.')[:24]
seq = prompt_ids + suffix_ids
train_step = []
roll_step = []
for i in range(len(seq)-1):
    pref = seq[:i+1]
    with torch.no_grad():
        logits_t = train_model(torch.tensor([pref], dtype=torch.long, device='cuda'))
        lp_t = F.log_softmax(logits_t[:, -1, :].float(), dim=-1)[0, seq[i+1]].item()
    logits_r = rollout_model.forward_batch([pref], rollout_model.generate_zero_state(1))
    lp_r = F.log_softmax(logits_r.float(), dim=-1)[0, seq[i+1]].item()
    train_step.append(lp_t)
    roll_step.append(lp_r)
diffs = [abs(a-b) for a,b in zip(train_step, roll_step)]
print('tokens_compared', len(diffs))
print('max_abs_diff', max(diffs) if diffs else 0.0)
print('mean_abs_diff', sum(diffs)/len(diffs) if diffs else 0.0)
for i,(a,b,d) in enumerate(list(zip(train_step, roll_step, diffs))[:40]):
    print(f'{i}: train={a:.6f} rollout={b:.6f} diff={d:.6f}')
