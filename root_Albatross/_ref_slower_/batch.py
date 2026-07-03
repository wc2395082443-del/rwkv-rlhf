########################################################################################################
#
# The RWKV-7 "Goose" Language Model - https://github.com/BlinkDL/RWKV-LM
#
########################################################################################################

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch, copy, time, random, json, math, gc
from tqdm import tqdm
from torch.nn import functional as F
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

########################################################################################################

args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64
#
# model download: https://huggingface.co/BlinkDL/rwkv7-g1
#
args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1a-0.1b-20250728-ctx4096"
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1a-0.4b-20250905-ctx4096"
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1-1.5b-20250429-ctx4096"
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1-2.9b-20250519-ctx4096"
# args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g0a-7.2b-20250829-ctx4096"

print(f'\nUsing CUDA fp16. Loading {args.MODEL_NAME} ...\n')

from reference.rwkv7 import RWKV_x070
model = RWKV_x070(args)

from reference.utils import TRIE_TOKENIZER, sampler_simple_batch
tokenizer = TRIE_TOKENIZER("reference/rwkv_vocab_v20230424.txt")

########################################################################################################

# prompts = ["The apple can be", "The cat can be"]
# prompts = ["The apple can't be", "The cat can't be"]
prompts = ["The apple can be", "The cat can't be", "他们发现，这", "Q: 1+1=?\nA: 1+1=2."]
BATCH_SIZE = len(prompts)

state = model.generate_zero_state(BATCH_SIZE)
init_outs = model.forward_batch([tokenizer.encode(prompt) for prompt in prompts], state)

for n in range(BATCH_SIZE):
    print(prompts[n])
    init_out = init_outs[n]
    probs = F.softmax(init_out.float(), dim=-1) # compute softmax in float (more accurate)
    _, indices = torch.topk(probs, 5) # print top-5 possibilities
    for i in range(len(indices)):
        token_id = indices[i].item()
        token = tokenizer.decode([token_id])
        token_prob = probs[token_id].item()
        print(repr(token), f'[probability {token_prob:.2%}]')
    if n != BATCH_SIZE-1:
        print()

########################################################################################################

prompts = ["也许", "我看到", "他们发现", "我认为", "哈哈", "这是一个有趣的", "List of Emojis:"]
BATCH_SIZE = len(prompts)
# prompts = ["这是一个有趣的"] * BATCH_SIZE
# prompts = ["他们发现"] * BATCH_SIZE
# prompts = ["我看到"] * BATCH_SIZE

state = model.generate_zero_state(BATCH_SIZE)
out = model.forward_batch([tokenizer.encode(prompt) for prompt in prompts], state)

tokens = []
GENERATE_LENGTH = 10
for i in range(GENERATE_LENGTH):
    new_tokens = sampler_simple_batch(out, noise=0).tolist()
    tokens.append(new_tokens)
    out = model.forward_batch(new_tokens, state)

tokens = np.transpose(np.array(tokens), axes=(1,0,2)).squeeze(-1)

print('\n')
for n in range(BATCH_SIZE):
    print(prompts[n], end='')
    print(tokenizer.decode(tokens[n], utf8_errors="ignore"))
    print('#'*80)

########################################################################################################

BATCH_SIZE=256
print(f'BATCH_SIZE {BATCH_SIZE} LAMBADA eval')

def eval_qa_batch(todo, print_interval, pad_eod = True, loss_mode = False, BATCH_SIZE = 1):
    xsum = 0
    xcnt = 0
    xacc = 0

    fwd_tokens = []
    fwd_desc = []

    for i in range(len(todo)):

        # get src and dst
        d = todo[i]
        if pad_eod:
            src = [0] + tokenizer.encode(d[0])
        else:
            src = tokenizer.encode(d[0])
        dst = tokenizer.encode(d[1])

        # store jobs
        fwd_tokens.append(src+dst)
        fwd_desc.append((src, dst))
        
        if len(fwd_tokens) >= BATCH_SIZE or i == len(todo)-1:
            
            # batch fwd
            out_batch = model.forward_batch(fwd_tokens, model.generate_zero_state(BATCH_SIZE), full_output=True)

            # process output
            for j in range(len(fwd_desc)):
                
                out = out_batch[j]
                src, dst = fwd_desc[j]

                logits = 0
                correct = True                
                for n in range(len(dst)):
                    ooo = out[len(src)-1+n].float()
                    probs = F.softmax(ooo, dim=-1)
                    logits += math.log(probs[dst[n]])
                    if torch.argmax(probs).item() != dst[n]:
                        correct = False

                xcnt += 1
                xsum += logits
                xacc += 1 if correct else 0
                if xcnt % print_interval == 0 or xcnt == len(todo):
                    if loss_mode:
                        print('loss', round(-xsum / xcnt, 2), 'acc', round(xacc/xcnt*100, 1))
                    else:
                        print(xcnt, 'ppl', round(math.exp(-xsum / xcnt), 2), 'acc', round(xacc/xcnt*100, 1))
            
            fwd_tokens = []
            fwd_desc = []

with open(f"eval/lambada_test.jsonl", "r", encoding="utf-8") as f:
    todo = [json.loads(line) for line in f]
    todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]

eval_qa_batch(todo, print_interval=1000, BATCH_SIZE=BATCH_SIZE)
