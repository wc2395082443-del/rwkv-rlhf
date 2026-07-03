########################################################################################################
#
# RWKV-7 Optimized Benchmark with Hybrid CUDA Kernels
#
########################################################################################################

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch, copy, time, random, json, math, gc, sys, os
from tqdm import tqdm
from torch.nn import functional as F

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

SHOW_SPEED_PERCENTILE = 50

########################################################################################################

args = types.SimpleNamespace()
args.vocab_size = 65536
args.head_size = 64
args.MODEL_NAME = "C:\\RWKV-LM\\rwkv7-g1b-1.5b-20251202-ctx8192"

print(f'\n{"="*80}')
print(f'RWKV-7 Optimized Benchmark with Hybrid CUDA Kernels')
print(f'{"="*80}')
print(f'Loading model: {args.MODEL_NAME}\n')

# 导入优化的模型
current_dir = os.path.dirname(os.path.abspath(__file__))
reference_dir = os.path.join(current_dir, 'reference')

if os.path.exists(reference_dir):
    sys.path.insert(0, reference_dir)
    from reference.rwkv7 import RWKV_x070, kernel_manager
else:
    from reference.rwkv7 import RWKV_x070, kernel_manager

model = RWKV_x070(args)

PARAM_BYTES = 2
active_params = 0
for k,v in model.z.items():
    if 'emb' not in k:
        active_params += v.numel()
active_GB = active_params/1e9*PARAM_BYTES
print(f'\nActive params = {round(active_params/1e9,2)} B = {round(active_GB,2)} GB\n')

# 导入 tokenizer
if os.path.exists(reference_dir):
    from reference.utils import TRIE_TOKENIZER, sampler_simple, sampler_simple_batch
    tokenizer = TRIE_TOKENIZER(os.path.join(reference_dir, "rwkv_vocab_v20230424.txt"))
else:
    from reference.utils import TRIE_TOKENIZER, sampler_simple, sampler_simple_batch
    tokenizer = TRIE_TOKENIZER("reference/rwkv_vocab_v20230424.txt")

########################################################################################################

def xprint(s):
    c0, c1 = 3, 80-len(s)-3
    print(f"\n{'#'*c0} {s} {'#'*c1}\n")

########################################################################################################

xprint("Kernel Performance Analysis")

print("Testing different scenarios to find optimal kernel dispatch threshold...\n")

# 测试不同的 (B, T) 组合
test_cases = [
    (1, 1, "Single token"),
    (1, 64, "Short sequence"),
    (1, 256, "Medium sequence"),
    (1, 1024, "Long sequence"),
    (1, 4096, "Very long sequence"),
    (2, 1, "Small batch single token"),
    (8, 1, "Medium batch single token"),
    (32, 1, "Large batch single token"),
    (8, 64, "Medium batch short seq"),
    (8, 256, "Medium batch medium seq"),
]

results = []

for B, T, desc in test_cases:
    print(f"Testing: {desc} (B={B}, T={T})")
    
    # 准备数据
    if B == 1:
        tokens = list(range(T))
        state = model.generate_zero_state(0)
    else:
        tokens = [list(range(T)) for _ in range(B)]
        state = model.generate_zero_state(B)
    
    # 预热
    if B == 1:
        model.forward(tokens, state)
    else:
        model.forward_batch(tokens, state)
    
    # 测试
    times = []
    for _ in range(10):
        if B == 1:
            state = model.generate_zero_state(0)
        else:
            state = model.generate_zero_state(B)
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        if B == 1:
            out = model.forward(tokens, state)
        else:
            out = model.forward_batch(tokens, state)
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    avg_time = np.mean(times)
    min_time = np.min(times)
    throughput = B * T / avg_time
    
    kernel_used = kernel_manager.get_kernel_name(B, T)
    
    results.append({
        'B': B,
        'T': T,
        'desc': desc,
        'avg_time': avg_time,
        'min_time': min_time,
        'throughput': throughput,
        'kernel': kernel_used
    })
    
    print(f"  Kernel: {kernel_used:15s} | Time: {avg_time*1000:6.2f}ms | Throughput: {throughput:7.1f} tok/s\n")

# 打印汇总表
print("\n" + "="*100)
print("Performance Summary:")
print("="*100)
print(f"{'Scenario':<30} {'B':>3} {'T':>5} {'Kernel':>15} {'Time(ms)':>10} {'Throughput':>12}")
print("-"*100)
for r in results:
    print(f"{r['desc']:<30} {r['B']:>3} {r['T']:>5} {r['kernel']:>15} {r['avg_time']*1000:>10.2f} {r['throughput']:>12.1f}")
print("="*100)

########################################################################################################

xprint("Basic Test")

prompt = "The Eiffel tower is in the city of"
print(f"Prompt: {prompt}")

init_out = model.forward(tokenizer.encode(prompt), model.generate_zero_state(0))
probs = F.softmax(init_out.float(), dim=-1)
_, indices = torch.topk(probs, 5)
for i in range(len(indices)):
    token_id = indices[i].item()
    token = tokenizer.decode([token_id])
    token_prob = probs[token_id].item()
    print(repr(token), f'[probability {token_prob:.2%}]')

########################################################################################################

xprint("Decode (with kernel auto-dispatch)")

prompt = "User: simulate SpaceX mars landing using python\n\nAssistant: <think"
LENGTH_PER_TRIAL = 256
print(prompt, end="")

all_tokens = []
out_last = 0
state = model.generate_zero_state(0)
out = model.forward(tokenizer.encode(prompt), state)

times = []
all_times = []
t000 = time.perf_counter()
for i in range(LENGTH_PER_TRIAL):
    t00 = time.perf_counter()
    token = sampler_simple(out, noise=0).item()
    all_tokens += [token]
    try:
        tmp = tokenizer.decode(all_tokens[out_last:], utf8_errors="strict")
        print(tmp, end="", flush=True)
        out_last = i+1
    except:
        pass
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.forward(token, state)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append(t1 - t0)
    all_times.append(t1 - t00)

times_median = np.percentile(times, SHOW_SPEED_PERCENTILE)
all_times_median = np.percentile(all_times, SHOW_SPEED_PERCENTILE)
total_time = time.perf_counter() - t000

print(f'\n\n{"="*80}')
print(f'Decode Performance:')
print(f'  Token/s: {round(1/times_median,2)} (forward only), {round(1/all_times_median,2)} (full pipeline)')
print(f'  Bandwidth: {round(active_GB/times_median,2)} GB/s')
print(f'  Total time: {round(total_time,3)}s')
print(f'{"="*80}')

# 打印 kernel 使用统计
model.print_kernel_stats()

########################################################################################################

xprint("Batch Performance Comparison")

for BSZ in [2, 8, 32, 128, 512]:
    torch.cuda.empty_cache()
    gc.collect()

    state = model.generate_zero_state(BSZ)
    
    if BSZ <= 2:
        prompts = ["The apple can be", "The cat can't be"][:BSZ]
    else:
        prompts = ["The apple can be" for _ in range(BSZ)]
    
    tokens = [tokenizer.encode(prompt) for prompt in prompts]
    LENGTH_PER_TRIAL = 32

    # 预热
    out = model.forward_batch(tokens, state)
    
    times = []
    for i in range(LENGTH_PER_TRIAL):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        token = sampler_simple_batch(out, noise=0).tolist()
        out = model.forward_batch(token, state)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times_median = np.percentile(times, SHOW_SPEED_PERCENTILE)
    throughput = BSZ / times_median
    
    kernel_used = kernel_manager.get_kernel_name(BSZ, 1)
    
    print(f'BSZ {BSZ:4d} | Kernel: {kernel_used:15s} | {throughput:7.1f} tok/s | {times_median*1000:6.2f}ms/batch')
    
    del state
    torch.cuda.empty_cache()
    gc.collect()

########################################################################################################

xprint("Prefill Performance (different sequence lengths)")

test_sequences = [64, 256, 1024, 4096]
if os.path.exists("eval/calibration_data_v5_rc.txt"):
    raw = open("eval/calibration_data_v5_rc.txt").read()
    all_tokens = tokenizer.encode(raw)
    
    for seq_len in test_sequences:
        if seq_len > len(all_tokens):
            continue
        
        tokens = all_tokens[:seq_len]
        state = model.generate_zero_state(0)
        
        # 预热
        model.forward(tokens[:-1], state, full_output=True)
        
        times = []
        for _ in range(5):
            state = model.generate_zero_state(0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            prob = model.forward(tokens[:-1], state, full_output=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        
        avg_time = np.mean(times)
        throughput = (seq_len-1) / avg_time
        tflops = throughput * active_params * 2 / 1e12
        
        kernel_used = kernel_manager.get_kernel_name(1, seq_len-1)
        
        print(f'SeqLen {seq_len:5d} | Kernel: {kernel_used:15s} | {throughput:7.1f} tok/s | {tflops:5.2f} TFLOPS')
else:
    print("Calibration data not found, skipping prefill test")

########################################################################################################

# 打印最终统计
print("\n" + "="*80)
print("Final Kernel Usage Statistics:")
print("="*80)
model.print_kernel_stats()

# 保存配置
kernel_manager.save_config()
print(f"Kernel configuration saved to: {kernel_manager.config_file}\n")

