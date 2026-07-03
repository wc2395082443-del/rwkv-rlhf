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

SHOW_SPEED_PERCENTILE = 50

print("\n\n### Note: this will torch.compile so please wait a few minutes ###\n\n")

########################################################################################################

args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64
#
# model download: https://huggingface.co/BlinkDL/rwkv7-g1
#
args.MODEL_NAME = "/mnt/e/RWKV-Runner/models/rwkv7-g1f-7.2b-20260414-ctx8192"

print(f'\nUsing CUDA fp16. Loading {args.MODEL_NAME} ...\n')

from reference.rwkv7 import RWKV_x070
model = RWKV_x070(args)

PARAM_BYTES = 2
active_params = 0
for k,v in model.z.items():
    if 'emb' not in k:
        active_params += v.numel()
active_GB = active_params/1e9*PARAM_BYTES
print(f'\nActive params = {round(active_params/1e9,2)} B = {round(active_GB,2)} GB (gigabytes)')

from reference.utils import TRIE_TOKENIZER, sampler_simple, sampler_simple_batch
tokenizer = TRIE_TOKENIZER("reference/rwkv_vocab_v20230424.txt")

########################################################################################################

def xprint(s):
    c0, c1 = 3, 80-len(s)-3
    print(f"\n{'#'*c0} {s} {'#'*c1}\n")

# xprint("Basic")

# prompt = "The Eiffel tower is in the city of"
# print(prompt)

# init_out = model.forward(tokenizer.encode(prompt), model.generate_zero_state(0))
# probs = F.softmax(init_out.float(), dim=-1) # compute softmax in float (more accurate)
# _, indices = torch.topk(probs, 5) # print top-5 possibilities
# for i in range(len(indices)):
#     token_id = indices[i].item()
#     token = tokenizer.decode([token_id])
#     token_prob = probs[token_id].item()
#     print(repr(token), f'[probability {token_prob:.2%}]')

# ########################################################################################################

# xprint("Batch")

# prompts = ["The apple can be", "The cat can't be", "Q: 1+1=?\nA: 1+1=2."]
# tokens = [tokenizer.encode(prompt) for prompt in prompts]

# print(tokens)
# for prompt in prompts:
#     print(prompt)
#     init_out = model.forward(tokenizer.encode(prompt), model.generate_zero_state(0))
#     probs = F.softmax(init_out.float(), dim=-1) # compute softmax in float (more accurate)
#     _, indices = torch.topk(probs, 5) # print top-5 possibilities
#     for i in range(len(indices)):
#         token_id = indices[i].item()
#         token = tokenizer.decode([token_id])
#         token_prob = probs[token_id].item()
#         print(repr(token), f'[probability {token_prob:.2%}]')

# idx = tokens
# B = len(idx)
# state = model.generate_zero_state(B)

# TT = max(len(h) for h in idx)
# idx1 = torch.full((B, TT), 0, dtype=torch.long, device="cuda")
# Tm = torch.zeros(B, dtype=torch.int, device="cuda")
# for i in range(B):
#     t = TT - len(idx[i])
#     Tm[i] = t
#     idx1[i, t:] = torch.tensor(idx[i], dtype=torch.long, device="cuda")
# lens = torch.tensor([len(z) for z in tokens], dtype=torch.int32, device="cuda")

# # att_mask = (torch.arange(TT, device="cuda") < Tm.unsqueeze(1)).unsqueeze(-1)
# # return self.forward_prefill(B, idx1, state, att_mask=att_mask, full_output=full_output)


# init_outs = model.forward_seq_batch_right(idx1, state, lens)
# for n in range(len(prompts)):
#     print(prompts[n])
#     init_out = init_outs[n]
#     probs = F.softmax(init_out.float(), dim=-1) # compute softmax in float (more accurate)
#     _, indices = torch.topk(probs, 5) # print top-5 possibilities
#     for i in range(len(indices)):
#         token_id = indices[i].item()
#         token = tokenizer.decode([token_id], utf8_errors="replace")
#         token_prob = probs[token_id].item()
#         print(repr(token), f'[probability {token_prob:.2%}]')
#     if n != len(prompts)-1:
#         print()
# exit(0)

########################################################################################################

from torch.profiler import schedule
from torch.profiler import profile, ProfilerActivity, record_function
# my_schedule = schedule(skip_first=1, wait=1, warmup=1, active=1)
xprint("Decode")

prompt = "User: simulate SpaceX mars landing using python\n\nAssistant: <think"
LENGTH_PER_TRIAL = 256
TEMPERATURE = 1.0
TOP_P = 0.0
print(prompt, end="")

all_tokens = []
out_last = 0
state = model.generate_zero_state(0)
out = model.forward(tokenizer.encode(prompt), state)
token = sampler_simple(out, noise=0).item()

times = []
all_times = []
t000 = time.perf_counter()


for i in range(LENGTH_PER_TRIAL):
    t00 = time.perf_counter()
    all_tokens += [token]
    try:
        tmp = tokenizer.decode(all_tokens[out_last:], utf8_errors="strict")
        print(tmp, end="", flush=True) # only print when we have a valid utf-8 string
        out_last = i+1
    except:
        pass
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.forward(token, state, with_sampling=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append(t1 - t0)
    all_times.append(t1 - t00)
    token = out.item()
times = np.percentile(times, SHOW_SPEED_PERCENTILE)
all_times = np.percentile(all_times, SHOW_SPEED_PERCENTILE)
print(f'\n\nToken/s = {round(1/times,2)} (forward), {round(1/all_times,2)} (full) || Bandwidth = {round(active_GB/times,2)} GB/s || {round(time.perf_counter()-t000,3)}s')


# LENGTH_PER_TRIAL = 8
# times = []
# all_times = []
# t000 = time.perf_counter()

# with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     schedule=torch.profiler.schedule(skip_first=2, wait=2, warmup=2, active=2),
# ) as p:
#     for i in range(LENGTH_PER_TRIAL):
#         # t00 = time.perf_counter()
#         # token = sampler_simple(out, noise=0).item()
#         all_tokens += [token]
#         try:
#             tmp = tokenizer.decode(all_tokens[out_last:], utf8_errors="strict")
#             print(tmp, end="", flush=True) # only print when we have a valid utf-8 string
#             out_last = i+1
#         except:
#             pass
#         # torch.cuda.synchronize()
#         # t0 = time.perf_counter()
#         out = model.forward(token, state, with_sampling=True)
#         token = out.item()
#         p.step()
#         # torch.cuda.synchronize()
#         # t1 = time.perf_counter()
#         # times.append(t1 - t0)
#         # all_times.append(t1 - t00)
# # times = np.percentile(times, SHOW_SPEED_PERCENTILE)
# # all_times = np.percentile(all_times, SHOW_SPEED_PERCENTILE)
# # print(f'\n\nToken/s = {round(1/times,2)} (forward), {round(1/all_times,2)} (full) || Bandwidth = {round(active_GB/times,2)} GB/s || {round(time.perf_counter()-t000,3)}s')
# p.export_chrome_trace("trace_sampling.json")
# exit(0)
#######################################################################################################

xprint("Decode (CUDAGraph)")

prompt = "User: simulate SpaceX mars landing using python\n\nAssistant: <think"
LENGTH_PER_TRIAL = 256
TEMPERATURE = 1.0
TOP_P = 0.0
print(prompt, end="")

all_tokens = []
out_last = 0
state = model.generate_zero_state(0)
out = model.forward(tokenizer.encode(prompt), state)
token = sampler_simple(out, noise=0).item()
# token = model.forward(tokenizer.encode(prompt), state, with_sampling=True)

x = model.z['emb.weight'][token]

static_input = torch.empty_like(x, device="cuda")
static_state = copy.deepcopy(state)
# static_state = [None, None, None]
# static_state[0] = torch.empty_like(state[0], device="cuda")
# static_state[1] = torch.empty_like(state[1], device="cuda")
# static_state[2] = torch.empty_like(state[2], device="cuda")
static_output = torch.empty((1,), dtype=torch.int32, requires_grad=False, device="cuda")
static_output = model.forward(static_input, static_state, with_sampling=True)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model.forward(static_input, static_state, with_sampling=True)

static_input.copy_(x)
for i in range(len(state)):
    static_state[i].copy_(state[i])
# static_output.copy_(out)
static_output[0] = token

times = []
all_times = []
t000 = time.perf_counter()
for i in range(0, LENGTH_PER_TRIAL):
    t00 = time.perf_counter()
    # token = sampler_simple(static_output, noise=0).item()
    token = static_output.item()
    all_tokens += [token]
    try:
        tmp = tokenizer.decode(all_tokens[out_last:], utf8_errors="strict")
        print(tmp, end="", flush=True) # only print when we have a valid utf-8 string
        out_last = i+1
    except:
        pass

    static_input.copy_(model.z['emb.weight'][token])

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g.replay()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append(t1 - t0)
    all_times.append(t1 - t00)
times = np.percentile(times, SHOW_SPEED_PERCENTILE)
all_times = np.percentile(all_times, SHOW_SPEED_PERCENTILE)
print(f'\n\nToken/s = {round(1/times,2)} (forward), {round(1/all_times,2)} (full) || Bandwidth = {round(active_GB/times,2)} GB/s || {round(time.perf_counter()-t000,3)}s')

exit(0)
#######################################################################################################

xprint("Decode (batch)")

for BSZ in [960, 960, 960, 960]:
# for BSZ in [512, 512, 512, 512]:
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    state = model.generate_zero_state(BSZ)

    time.sleep(1)
    if BSZ == 2:
        prompts = ["The apple can be", "The cat can't be"]
    else:
        prompts = ["The apple can be" for _ in range(BSZ)]
    nnn = len(prompts)
    tokens = [tokenizer.encode(prompt) for prompt in prompts]
    LENGTH_PER_TRIAL = 32
    # TEMPERATURE = 1.0
    # TOP_P = 0.0

    if BSZ == 2:
        print('wait', end='')
    all_tokens = []
    out = model.forward_batch(tokens, state)

    times = []
    all_times = []
    t000 = time.perf_counter()
    for i in range(LENGTH_PER_TRIAL):
        t00 = time.perf_counter()
        token = sampler_simple_batch(out, noise=0).tolist()
        all_tokens += [token]
        if BSZ == 2:
            print('.', end='', flush=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.forward_batch(token, state)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        all_times.append(t1 - t00)

    times = np.percentile(times, SHOW_SPEED_PERCENTILE)
    all_times = np.percentile(all_times, SHOW_SPEED_PERCENTILE)

    del state
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    if BSZ == 2:
        print('\n')
        for n in range(nnn):
            print(prompts[n], end='')
            aaa_tokens = []
            for i in range(LENGTH_PER_TRIAL):
                aaa_tokens += all_tokens[i][n]
            print(tokenizer.decode(aaa_tokens, utf8_errors="ignore"))
            print('#'*80)

    print(f'Bsz {BSZ} || Token/s = {round(nnn/times,2)} (forward), {round(nnn/all_times,2)} (full) || {round(time.perf_counter()-t000,3)}s')

#######################################################################################################

# xprint("Prefill")

# raw = open("eval/calibration_data_v5_rc.txt").read()
# tokens = tokenizer.encode(raw)
# # print(len(tokens))

# for stage in range(8, 12+1):
#     CTX_LEN = 2**stage
#     loss = 0
#     a = 0
#     cnt = 0
    
#     times = []
#     while a+CTX_LEN < len(tokens):
#         src = tokens[a:a+CTX_LEN]

#         torch.cuda.synchronize()
#         t0 = time.perf_counter()
#         prob = model.forward(src[:-1], model.generate_zero_state(0), full_output=True)
#         torch.cuda.synchronize()
#         t1 = time.perf_counter()
#         times.append(t1 - t0)
            
#         prob = F.softmax(prob.float(), dim=-1)
#         for j in range(CTX_LEN-1):
#             loss -= math.log(prob[j][src[j+1]])
#             cnt += 1
#         a += CTX_LEN

#     times = np.percentile(times, SHOW_SPEED_PERCENTILE)
#     print(f'CTX_LEN {CTX_LEN} : avg loss {round(loss/cnt,4)} || prefill {round((CTX_LEN-1)/times)} token/s = {round((CTX_LEN-1)/times * active_params * 2/1e12, 2)} TFLOPS')

# exit(0)
# #######################################################################################################

# xprint("Arithmetic")

def eval_qa(todo, print_interval, pad_eod = True, loss_mode = False):
    xsum = 0
    xcnt = 0
    xacc = 0
    for d in todo:
        if pad_eod:
            src = [0] + tokenizer.encode(d[0])
        else:
            src = tokenizer.encode(d[0])
        dst = tokenizer.encode(d[1])

        logits = 0
        correct = True
        
        out = model.forward(src+dst, model.generate_zero_state(0), full_output=True)

        for i in range(len(dst)):
            ooo = out[len(src)-1+i].float()
            probs = F.softmax(ooo, dim=-1)
            logits += math.log(probs[dst[i]])
            if torch.argmax(probs).item() != dst[i]:
                correct = False

        xcnt += 1
        xsum += logits
        xacc += 1 if correct else 0
        if xcnt % print_interval == 0 or xcnt == len(todo):
            if loss_mode:
                print('loss', round(-xsum / xcnt, 2), 'acc', round(xacc/xcnt*100, 1))
            else:
                print(xcnt, 'ppl', round(math.exp(-xsum / xcnt), 2), 'acc', round(xacc/xcnt*100, 1))

# x1, x2 = 1, 2
# magic = (5**(0.5)-1)/2
# for stage in range(2,7+1):
#     todo = []
#     NUMBER_LIMIT = 10**stage
#     for i in range(200):
#         x1 += i
#         x2 += i*i
#         s1 = int(magic * x1 * NUMBER_LIMIT) % NUMBER_LIMIT
#         s2 = int(magic * x2 * NUMBER_LIMIT) % NUMBER_LIMIT
#         # todo.append([f'\nAssistant: {s1}+{s2}=',str(s1+s2)])
#         # todo.append([f'\nAssistant: {s1}-{s2}=',str(s1-s2)])
#         todo.append([f'\nA: 123+321=444\n{s1}+{s2}=',str(s1+s2)]) # better prompt
#         todo.append([f'\nA: 123-321=-198\n{s1}-{s2}=',str(s1-s2)]) # better prompt
#     # print(todo)
#     print(f"Len {stage} : ", end="")
#     eval_qa(todo, 99999999, pad_eod=False, loss_mode=True)

# #######################################################################################################

# xprint("Repeat")

# class LCG:
#     def __init__(self, seed=42):
#         self.m = 2**32  # Modulus
#         self.a = 1664525  # Multiplier
#         self.c = 1013904223  # Increment
#         self.state = seed
#     def _generate(self):
#         self.state = (self.a * self.state + self.c) % self.m
#         return self.state
#     def randint(self, min_val, max_val):
#         if min_val > max_val:
#             raise ValueError("min_val cannot be greater than max_val")
#         range_size = max_val - min_val + 1
#         return min_val + self._generate() % range_size
# lcg = LCG()
# def generate_random_number_string(n, generator):
#     if not isinstance(n, int) or n <= 0:
#         raise ValueError("Number of digits N must be a positive integer.")
#     if n == 1:
#         return str(generator.randint(0, 9))
#     first_digit = str(generator.randint(1, 9))
#     remaining_digits = [str(generator.randint(0, 9)) for _ in range(n - 1)]
#     return first_digit + "".join(remaining_digits)
# def generate_random_string(n, generator):
#     if not isinstance(n, int) or n <= 0:
#         raise ValueError("Number of digits N must be a positive integer.")
#     ccccc = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
#     chars = [ccccc[generator.randint(0, len(ccccc)-1)] for _ in range(n)]
#     return "".join(chars)

# for stage in range(4):
#     todo = []
#     l_max = 0
#     l_min = 1e10
#     for i in range(100):
#         l = round(pow(2,(stage+i/100)) * 100)
#         l_min = min(l, l_min)
#         l_max = max(l, l_max)
#         s = generate_random_string(l, lcg)
#         todo.append([f'\nYou must remember the secret is {s}. Repeat: the secret is', f' {s}'])
#     print(f"Len {l_min} to {l_max} : ", end="")
#     eval_qa(todo, 99999999, loss_mode=True)

#######################################################################################################

xprint('LAMBADA')

with open(f"eval/lambada_test.jsonl", "r", encoding="utf-8") as f:
    todo = [json.loads(line) for line in f]
    todo = [[doc['text'].rsplit(' ', 1)[0], " " + doc['text'].rsplit(' ', 1)[1]] for doc in todo]

eval_qa(todo, 1000)

########################################################################################################

xprint('MMLU')

from datasets import load_from_disk
mmlu_test = load_from_disk("eval/mmlu_test_dataset")

TEMPLATE = '''User: You are a very talented expert in <SUBJECT>. Answer this question:
<Q>
A. <|A|>
B. <|B|>
C. <|C|>
D. <|D|>

Assistant: The answer is'''

CHOICES = [" A", " B", " C", " D"]

SHUFFLE = False

correct = 0
total = 0
pbar = tqdm(total=len(mmlu_test))

choices_token = [tokenizer.encode(x) for x in CHOICES]
assert all([len(x) == 1 for x in choices_token])
choices_token = [x[0] for x in choices_token]

for idx, sample in enumerate(mmlu_test):
    question = sample["question"]
    choices = sample["choices"]
    subject = sample["subject"]
    gt = sample["answer"]

    if SHUFFLE and not any(["Both" in x for x in choices]):  # exclude choices like "Both A and B"
        original_gt_text = choices[gt]
        np.random.shuffle(choices)
        gt = choices.index(original_gt_text)

    all_prefix = (
        TEMPLATE.replace("<Q>", question)
        .replace("<|A|>", choices[0])
        .replace("<|B|>", choices[1])
        .replace("<|C|>", choices[2])
        .replace("<|D|>", choices[3])
        .replace("<SUBJECT>", subject.replace("_", " "))
    )

    if idx == 0:
        print(f"Format example:")
        print("-" * 80)
        print(all_prefix)
        print("-" * 80)
        format_example = all_prefix

    all_prefix_ids = [0] + tokenizer.encode(all_prefix.replace('\r\n','\n').strip())

    logits = model.forward(all_prefix_ids, model.generate_zero_state(0), full_output=False)
    
    neg_log_prob = F.log_softmax(logits, dim=-1)
    target_prob = neg_log_prob[choices_token]
    
    if torch.argmax(target_prob).item() == gt:
        correct += 1
    total += 1
    pbar.set_description(f"Correct: {correct} - Total: {total} - Accuracy: {correct / total:.5f}")
    pbar.update(1)
pbar.close()
print()