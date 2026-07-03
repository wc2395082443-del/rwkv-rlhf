import os, sys, types
from pathlib import Path
for _ninja_dir in ['/root/miniconda3/bin', '/usr/bin', '/bin']:
    if os.path.isfile(os.path.join(_ninja_dir, 'ninja')) and _ninja_dir not in os.environ.get('PATH', ''):
        os.environ['PATH'] = _ninja_dir + os.pathsep + os.environ.get('PATH', '')
        break
BASE = Path('/root/RWKV-LM/RWKV7-statetuning_hardbuffer_v1')
os.chdir(str(BASE))
os.environ['RWKV_MY_TESTING'] = 'x070'
os.environ['RWKV_CTXLEN'] = '4096'
os.environ['RWKV_HEAD_SIZE'] = '64'
os.environ['RWKV_FLOAT_MODE'] = 'bf16'
os.environ['RWKV_JIT_ON'] = '0'
import torch
import torch.nn.functional as F
sys.path.insert(0, str(BASE))
from reference.rwkv7 import RWKV_x070
from reference.utils import TRIE_TOKENIZER
ckpt = '/root/autodl-tmp/rwkv_models/rwkv7-g1e-1.5b-20260309-ctx8192.pth'
vocab = '/root/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt'
tok = TRIE_TOKENIZER(vocab)
encode = lambda s: tok.encode(s)
args = types.SimpleNamespace(MODEL_NAME=str(Path(ckpt).with_suffix('')), vocab_size=65536)
m = RWKV_x070(args)
seq = encode('User: 1+1?\nAssistant: Let us solve this')[:24]
state = m.generate_zero_state(1)
logits1 = m.forward_batch([seq], state)
val1 = F.log_softmax(logits1.float(), dim=-1)[0, seq[-1]].item()
m.z['head.weight'].zero_()
state = m.generate_zero_state(1)
logits2 = m.forward_batch([seq], state)
val2 = F.log_softmax(logits2.float(), dim=-1)[0, seq[-1]].item()
print('before', val1)
print('after ', val2)
print('changed', abs(val1-val2))
