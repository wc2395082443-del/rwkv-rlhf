########################################################################################################
#
# The RWKV-7 "Goose" Language Model - Fixed dtype handling
#
########################################################################################################

from typing import List
import os
import json
current_path = os.path.dirname(os.path.abspath(__file__))

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch._C._jit_set_autocast_mode(False)

import torch.nn as nn
from torch.nn import functional as F

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyStatic = torch.jit.script

DTYPE = torch.half

########################################################################################################
# CUDA Kernel Manager
########################################################################################################

from torch.utils.cpp_extension import load
HEAD_SIZE = 64
CHUNK_LEN = 16

current_path = os.path.dirname(os.path.abspath(__file__))
cuda_dir = os.path.join(current_path, "cuda")

class CUDAKernelManager:
    """管理和调度不同的 CUDA 核心"""
    
    def __init__(self):
        self.simple_kernel_available = False
        self.statepassing_kernel_available = False
        self.use_auto_dispatch = True
        self.dispatch_threshold = None
        self.performance_cache = {}
        
        # 加载配置
        self.config_file = os.path.join(current_path, "cuda_kernel_config.json")
        self.load_config()
        
        # 编译 CUDA 核心
        self._compile_kernels()
    
    def _compile_kernels(self):
        """编译两个 CUDA 核心"""
        
        # 1. 编译简单核心
        print("=" * 80)
        print("Compiling Simple CUDA Kernel (optimized for single token)...")
        print("=" * 80)
        
        simple_cu = os.path.join(cuda_dir, "rwkv7_state_fwd_fp16.cu")
        simple_cpp = os.path.join(cuda_dir, "rwkv7_state_fwd_fp16.cpp")
        
        if os.path.exists(simple_cu) and os.path.exists(simple_cpp):
            try:
                windows_flags = [
                    "-res-usage", 
                    "--use_fast_math", 
                    "-Xcompiler", "/O2", 
                    "-Xptxas", "-O3",
                    "--extra-device-vectorization",
                    "-allow-unsupported-compiler",
                    "-D_N_=64"
                ]
                
                load(
                    name="rwkv7_state_fwd_fp16",
                    sources=[simple_cpp, simple_cu],
                    is_python_module=False,
                    verbose=True,
                    extra_cuda_cflags=windows_flags,
                    extra_include_paths=[cuda_dir]
                )
                
                # 检查函数是否可用
                try:
                    _ = torch.ops.rwkv7_state_fwd_fp16.forward_seq
                    _ = torch.ops.rwkv7_state_fwd_fp16.forward_one
                    self.simple_kernel_available = True
                    print("✓ Simple kernel compiled and loaded successfully!")
                    print("  Available functions: forward_seq, forward_one")
                except Exception as e:
                    print(f"✗ Simple kernel compiled but functions not accessible: {e}")
                    
            except Exception as e:
                print(f"✗ Simple kernel compilation failed: {e}")
        else:
            print(f"✗ Simple kernel files not found")
            print(f"  Expected: {simple_cu}")
            print(f"  Expected: {simple_cpp}")
        
        # 2. 编译 statepassing 核心
        print("\n" + "=" * 80)
        print("Compiling StatePassing CUDA Kernel (optimized for long sequences)...")
        print("=" * 80)
        
        statepassing_cu = os.path.join(cuda_dir, "rwkv7_statepassing_clampw.cu")
        statepassing_cpp = os.path.join(cuda_dir, "rwkv7_statepassing_clampw.cpp")
        
        if os.path.exists(statepassing_cu) and os.path.exists(statepassing_cpp):
            try:
                flags = [
                    '-res-usage', 
                    f'-D_N_={HEAD_SIZE}', 
                    f"-D_CHUNK_LEN_={CHUNK_LEN}", 
                    "--use_fast_math", 
                    "-O3", 
                    "-Xptxas", "-O3",
                    "--extra-device-vectorization"
                ]
                
                load(
                    name="rwkv7_statepassing_clampw",
                    sources=[statepassing_cpp, statepassing_cu],
                    is_python_module=False,
                    verbose=True,
                    extra_cuda_cflags=flags
                )
                
                # 检查函数是否可用
                try:
                    _ = torch.ops.rwkv7_statepassing_clampw.forward
                    self.statepassing_kernel_available = True
                    print("✓ StatePassing kernel compiled and loaded successfully!")
                except Exception as e:
                    print(f"✗ StatePassing kernel compiled but function not accessible: {e}")
                    
            except Exception as e:
                print(f"✗ StatePassing kernel compilation failed: {e}")
        else:
            print(f"✗ StatePassing kernel files not found")
        
        print("\n" + "=" * 80)
        print(f"Kernel Status:")
        print(f"  Simple Kernel: {'✓ Available' if self.simple_kernel_available else '✗ Not Available'}")
        print(f"  StatePassing Kernel: {'✓ Available' if self.statepassing_kernel_available else '✗ Not Available'}")
        print("=" * 80 + "\n")
    
    def load_config(self):
        """加载性能配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.dispatch_threshold = config.get('dispatch_threshold')
                    self.performance_cache = config.get('performance_cache', {})
            except:
                pass
    
    def save_config(self):
        """保存性能配置"""
        config = {
            'dispatch_threshold': self.dispatch_threshold,
            'performance_cache': self.performance_cache
        }
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def should_use_statepassing(self, B, T):
        """决定是否使用 statepassing 核心"""
        if not self.statepassing_kernel_available:
            return False
        if not self.simple_kernel_available:
            return True
        
        if self.dispatch_threshold is None:
            return (B >= 8 and T >= 1) or (T >= 512)
        
        return B * T >= self.dispatch_threshold
    
    def get_kernel_name(self, B, T):
        """获取将要使用的核心名称"""
        if self.should_use_statepassing(B, T):
            return "StatePassing"
        else:
            return "Simple"

# 全局 kernel manager
kernel_manager = CUDAKernelManager()

########################################################################################################
# 在模块加载时确定 kernel 配置
########################################################################################################

USE_SIMPLE_KERNEL = kernel_manager.simple_kernel_available
USE_STATEPASSING_KERNEL = kernel_manager.statepassing_kernel_available

print(f"Kernel Configuration (for JIT compilation):")
print(f"  Simple Kernel: {'✓ Enabled' if USE_SIMPLE_KERNEL else '✗ Disabled'}")
print(f"  StatePassing Kernel: {'✓ Enabled' if USE_STATEPASSING_KERNEL else '✗ Disabled'}")
print()

########################################################################################################
# CUDA Operators - 正确处理数据类型
########################################################################################################

# Simple Kernel
if USE_SIMPLE_KERNEL:
    class WKV_7_Simple_One(torch.autograd.Function):
        @staticmethod
        def forward(ctx, state, r, w, k, v, a, b):
            with torch.no_grad():
                C = r.shape[0]
                H = C // HEAD_SIZE
                assert all(x.dtype == DTYPE for x in [r,w,k,v,a,b])
                assert all(x.is_contiguous() for x in [r,w,k,v,a,b])
                y = torch.empty((C), device=k.device, dtype=DTYPE, memory_format=torch.contiguous_format)
                
                # 确保 state 是 fp16 并且连续
                state_original_dtype = state.dtype
                if state.dtype != DTYPE:
                    state_fp16 = state.to(DTYPE).contiguous()
                else:
                    state_fp16 = state if state.is_contiguous() else state.contiguous()
                
                elapsed_t = torch.zeros((1,), dtype=torch.int32, device=k.device)
                
                torch.ops.rwkv7_state_fwd_fp16.forward_one(
                    1, C, H, state_fp16, r, w, k, v, a, b, y, elapsed_t
                )
                
                # 如果原始 state 是 float32，需要转换回去
                if state_original_dtype != DTYPE:
                    state.copy_(state_fp16.to(state_original_dtype))
                elif not state.is_contiguous():
                    state.copy_(state_fp16)
                
                return y
    
    class WKV_7_Simple_Seq(torch.autograd.Function):
        @staticmethod
        def forward(ctx, state, r, w, k, v, a, b):
            with torch.no_grad():
                T, C = r.size()
                H = C // HEAD_SIZE
                assert all(x.dtype == DTYPE for x in [r,w,k,v,a,b])
                assert all(x.is_contiguous() for x in [r,w,k,v,a,b])
                y = torch.empty((T, C), device=k.device, dtype=DTYPE, memory_format=torch.contiguous_format)
                
                # 确保 state 是 fp16 并且连续
                state_original_dtype = state.dtype
                if state.dtype != DTYPE:
                    state_fp16 = state.to(DTYPE).contiguous()
                else:
                    state_fp16 = state if state.is_contiguous() else state.contiguous()
                
                elapsed_t = torch.zeros((1,), dtype=torch.int32, device=k.device)
                
                torch.ops.rwkv7_state_fwd_fp16.forward_seq(
                    1, T, C, H, state_fp16, r, w, k, v, a, b, y, elapsed_t
                )
                
                # 如果原始 state 是 float32，需要转换回去
                if state_original_dtype != DTYPE:
                    state.copy_(state_fp16.to(state_original_dtype))
                elif not state.is_contiguous():
                    state.copy_(state_fp16)
                
                return y
    
    class WKV_7_Simple_Batch(torch.autograd.Function):
        @staticmethod
        def forward(ctx, state, r, w, k, v, a, b):
            with torch.no_grad():
                B, T, C = r.size()
                H = C // HEAD_SIZE
                assert all(x.dtype == DTYPE for x in [r,w,k,v,a,b])
                assert all(x.is_contiguous() for x in [r,w,k,v,a,b])
                y = torch.empty((B, T, C), device=k.device, dtype=DTYPE, memory_format=torch.contiguous_format)
                
                elapsed_t = torch.zeros((B,), dtype=torch.int32, device=k.device)
                
                state_original_dtype = state.dtype
                
                # 对每个 batch 分别处理
                for i in range(B):
                    # 确保 state 是 fp16 并且连续
                    if state.dtype != DTYPE:
                        state_fp16 = state[i:i+1].to(DTYPE).contiguous()
                    else:
                        state_i = state[i:i+1]
                        state_fp16 = state_i if state_i.is_contiguous() else state_i.contiguous()
                    
                    torch.ops.rwkv7_state_fwd_fp16.forward_seq(
                        1, T, C, H, 
                        state_fp16, 
                        r[i], w[i], k[i], v[i], a[i], b[i], 
                        y[i], 
                        elapsed_t[i:i+1]
                    )
                    
                    # 如果原始 state 是 float32，需要转换回去
                    if state_original_dtype != DTYPE:
                        state[i:i+1].copy_(state_fp16.to(state_original_dtype))
                    elif not state[i:i+1].is_contiguous():
                        state[i:i+1].copy_(state_fp16)
                
                return y

# StatePassing Kernel
if USE_STATEPASSING_KERNEL:
    class WKV_7_StatePassing(torch.autograd.Function):
        @staticmethod
        def forward(ctx, state, r, w, k, v, a, b):
            with torch.no_grad():
                B, T, H, N = r.shape
                assert T % CHUNK_LEN == 0, f"T={T} must be divisible by CHUNK_LEN={CHUNK_LEN}"
                assert all(i.dtype == DTYPE for i in [r,w,k,v,a,b])
                assert all(i.is_contiguous() for i in [r,w,k,v,a,b])
                
                y = torch.empty_like(v)
                sT = torch.empty_like(state)
                s = torch.empty(B, H, T//CHUNK_LEN, N, N, dtype=torch.float32, device=r.device)
                sa = torch.empty(B, T, H, N, dtype=torch.float32, device=r.device)
                
                torch.ops.rwkv7_statepassing_clampw.forward(state, r, w, k, v, a, b, y, sT, s, sa)
                
                state.copy_(sT)
                return y.view(B, T, H*N)

########################################################################################################
# 定义 Kernel 函数
########################################################################################################

# 序列操作
if USE_STATEPASSING_KERNEL:
    def RWKV7_OP_IMPL(state, r, w, k, v, a, b):
        """使用 StatePassing kernel"""
        T, C = r.size()
        H = C // HEAD_SIZE
        
        r = r.view(T, H, HEAD_SIZE).unsqueeze(0)
        w = w.view(T, H, HEAD_SIZE).unsqueeze(0)
        k = k.view(T, H, HEAD_SIZE).unsqueeze(0)
        v = v.view(T, H, HEAD_SIZE).unsqueeze(0)
        a = a.view(T, H, HEAD_SIZE).unsqueeze(0)
        b = b.view(T, H, HEAD_SIZE).unsqueeze(0)
        
        orig_T = T
        if T % CHUNK_LEN != 0:
            pad_len = CHUNK_LEN - (T % CHUNK_LEN)
            r = F.pad(r, (0, 0, 0, 0, 0, pad_len))
            w = F.pad(w, (0, 0, 0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
            a = F.pad(a, (0, 0, 0, 0, 0, pad_len))
            b = F.pad(b, (0, 0, 0, 0, 0, pad_len))
        
        out = WKV_7_StatePassing.apply(state, r, w, k, v, a, b)
        return out.view(-1, C)[:orig_T]
    
    print("Using StatePassing kernel for sequence operations")
    
elif USE_SIMPLE_KERNEL:
    def RWKV7_OP_IMPL(state, r, w, k, v, a, b):
        """使用 Simple kernel"""
        return WKV_7_Simple_Seq.apply(state, r, w, k, v, a, b)
    
    print("Using Simple kernel for sequence operations")
    
else:
    def RWKV7_OP_IMPL(state, r, w, k, v, a, b):
        raise NotImplementedError("No CUDA kernel available")
    print("⚠ No CUDA kernel available")

# 单 token 操作
if USE_SIMPLE_KERNEL:
    def RWKV7_ONE_OP_IMPL(state, r, w, k, v, a, b):
        return WKV_7_Simple_One.apply(state, r, w, k, v, a, b)
    print("Using Simple kernel for single token operations")
else:
    def RWKV7_ONE_OP_IMPL(state, r, w, k, v, a, b):
        result = RWKV7_OP_IMPL(state, r.unsqueeze(0), w.unsqueeze(0), 
                               k.unsqueeze(0), v.unsqueeze(0), 
                               a.unsqueeze(0), b.unsqueeze(0))
        return result.squeeze(0)
    print("Using fallback for single token operations")

# 批处理操作
if USE_STATEPASSING_KERNEL:
    def RWKV7_BATCH_OP_IMPL(state, r, w, k, v, a, b):
        B, T, C = r.size()
        H = C // HEAD_SIZE
        
        r = r.view(B, T, H, HEAD_SIZE)
        w = w.view(B, T, H, HEAD_SIZE)
        k = k.view(B, T, H, HEAD_SIZE)
        v = v.view(B, T, H, HEAD_SIZE)
        a = a.view(B, T, H, HEAD_SIZE)
        b = b.view(B, T, H, HEAD_SIZE)
        
        orig_T = T
        if T % CHUNK_LEN != 0:
            pad_len = CHUNK_LEN - (T % CHUNK_LEN)
            r = F.pad(r, (0, 0, 0, 0, 0, pad_len))
            w = F.pad(w, (0, 0, 0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
            a = F.pad(a, (0, 0, 0, 0, 0, pad_len))
            b = F.pad(b, (0, 0, 0, 0, 0, pad_len))
        
        out = WKV_7_StatePassing.apply(state, r, w, k, v, a, b)
        return out[:, :orig_T, :].contiguous()
    print("Using StatePassing kernel for batch operations")
elif USE_SIMPLE_KERNEL:
    def RWKV7_BATCH_OP_IMPL(state, r, w, k, v, a, b):
        return WKV_7_Simple_Batch.apply(state, r, w, k, v, a, b)
    print("Using Simple kernel for batch operations")
else:
    def RWKV7_BATCH_OP_IMPL(state, r, w, k, v, a, b):
        raise NotImplementedError("No CUDA kernel available")
    print("⚠ No CUDA kernel available")

print()

########################################################################################################
# JIT 安全的包装函数
########################################################################################################

def RWKV7_OP(state, r, w, k, v, a, b):
    return RWKV7_OP_IMPL(state, r, w, k, v, a, b)

def RWKV7_ONE_OP(state, r, w, k, v, a, b):
    return RWKV7_ONE_OP_IMPL(state, r, w, k, v, a, b)

def RWKV7_BATCH_OP(state, r, w, k, v, a, b):
    return RWKV7_BATCH_OP_IMPL(state, r, w, k, v, a, b)

########################################################################################################
# RWKV Model
########################################################################################################

class RWKV_x070(MyModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        args.head_size = 64
        self.eval()
        
        self.z = torch.load(args.MODEL_NAME + '.pth', map_location='cpu', mmap=True)
        z = self.z
        self.n_head, self.head_size = z['blocks.0.att.r_k'].shape
        args.n_embd = self.n_head * self.head_size

        assert HEAD_SIZE == self.head_size
        assert self.head_size == args.head_size

        keys = list(z.keys())
        max_layer = -1
        for k in keys:
            if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                z[k] = z[k].t()
            z[k] = z[k].squeeze().to(dtype=DTYPE, device="cuda")
            if k.endswith('att.r_k'): z[k] = z[k].flatten()
            z[k] = z[k].contiguous()
            kk = k.split('.')
            if kk[0] == 'blocks':
                max_layer = max(max_layer, int(kk[1]))
        args.n_layer = max_layer + 1
        
        print(f"\n{'='*80}")
        print(f"Model Configuration:")
        print(f"  Layers: {args.n_layer}")
        print(f"  Embedding: {args.n_embd}")
        print(f"  Heads: {self.n_head}")
        print(f"  Head Size: {self.head_size}")
        print(f"  CUDA Kernels: Hybrid (Auto-dispatch)")
        print(f"{'='*80}\n")
        
        self.n_layer, self.n_embd = args.n_layer, args.n_embd

        z['emb.weight'] = F.layer_norm(z['emb.weight'], (args.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])
        z['blocks.0.att.v0'] = z['blocks.0.att.a0']
        z['blocks.0.att.v1'] = z['blocks.0.att.a1']
        z['blocks.0.att.v2'] = z['blocks.0.att.a2']
        
        self.kernel_stats = {'Simple': 0, 'StatePassing': 0}

    def generate_zero_state(self, bsz):
        args = self.args
        state = [None, None]
        if bsz >= 1:
            state[0] = torch.zeros((args.n_layer, 2, bsz, args.n_embd), dtype=DTYPE, requires_grad=False, device="cuda")
            state[1] = torch.zeros((args.n_layer, bsz, args.n_embd // args.head_size, args.head_size, args.head_size), dtype=torch.float32, requires_grad=False, device="cuda")
        else:
            state[0] = torch.zeros((args.n_layer, 2, args.n_embd), dtype=DTYPE, requires_grad=False, device="cuda")
            state[1] = torch.zeros((args.n_layer, args.n_embd // args.head_size, args.head_size, args.head_size), dtype=torch.float32, requires_grad=False, device="cuda")
        return state

    def forward(self, idx, state, full_output=False):
        if type(idx) is list:
            if len(idx) > 1:
                kernel_name = kernel_manager.get_kernel_name(1, len(idx))
                self.kernel_stats[kernel_name] = self.kernel_stats.get(kernel_name, 0) + 1
                return self.forward_seq(idx, state, full_output)
            else:
                self.kernel_stats['Simple'] = self.kernel_stats.get('Simple', 0) + 1
                return self.forward_one(idx[0], state)
        else:
            self.kernel_stats['Simple'] = self.kernel_stats.get('Simple', 0) + 1
            return self.forward_one(idx, state)

    def forward_batch(self, tokens, state, full_output=False):
        assert type(tokens) is list
        lengths = [len(x) for x in tokens]
        if len(set(lengths)) == 1 and full_output == False:
            kernel_name = kernel_manager.get_kernel_name(len(tokens), lengths[0])
            self.kernel_stats[kernel_name] = self.kernel_stats.get(kernel_name, 0) + 1
            return self.forward_batch_same_length(tokens, state, full_output)

        bsz = len(tokens)
        pos = [0] * bsz

        if full_output == False:
            out = torch.empty((bsz, self.args.vocab_size), dtype=DTYPE, requires_grad=False, device="cuda")
        else:
            out = [torch.empty((0, self.args.vocab_size), dtype=DTYPE, requires_grad=False, device="cuda") for _ in range(bsz)]
        while True:
            active = [i for i in range(bsz) if pos[i] < lengths[i]]
            if not active:
                break
            step = min(lengths[i] - pos[i] for i in active)
            batch_tokens = [tokens[i][pos[i]:pos[i]+step] for i in active]
            batch_state = [state[0][:,:,active],state[1][:,active]]
            new_out = self.forward_batch_same_length(batch_tokens, batch_state, full_output)
            for k, i in enumerate(active):
                if full_output == False:
                    out[i] = new_out[k]
                else:
                    out[i] = torch.cat([out[i], new_out[k]], dim=0)
                state[0][:,:,i] = batch_state[0][:,:,k]
                state[1][:,i] = batch_state[1][:,k]
                pos[i] += step
        return out

    def forward_batch_same_length(self, tokens, state, full_output=False):
        assert type(tokens) is list
        assert len(set([len(x) for x in tokens])) == 1
        return self.forward_seq_batch(tokens, state, full_output)

    @MyFunction
    def forward_one(self, idx:int, state:List[torch.Tensor]):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, v_first = RWKV_x070_TMix_one(i, self.n_head, self.head_size, xx, state[0][i], v_first, state[1][i],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx = RWKV_x070_CMix_one(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x
        
    def forward_one_alt(self, x:torch.Tensor, state:List[torch.Tensor]):
        with torch.no_grad(): 
            z = self.z
            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, v_first = RWKV_x070_TMix_one(i, self.n_head, self.head_size, xx, state[0][i], v_first, state[1][i],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx = RWKV_x070_CMix_one(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x

    @MyFunction
    def forward_seq(self, idx:List[int], state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, v_first = RWKV_x070_TMix_seq(i, self.n_head, self.head_size, xx, state[0][i], v_first, state[1][i],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx = RWKV_x070_CMix_seq(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            if not full_output: x = x[-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x
        
    @MyFunction
    def forward_seq_batch(self, idxs:List[List[int]], state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][torch.tensor(idxs, device=z['emb.weight'].device)]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, v_first = RWKV_x070_TMix_seq_batch(i, self.n_head, self.head_size, xx, state[0][i], v_first, state[1][i],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx = RWKV_x070_CMix_seq_batch(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            if not full_output: x = x[:,-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x
    
    def print_kernel_stats(self):
        """打印 kernel 使用统计"""
        total = sum(self.kernel_stats.values())
        if total > 0:
            print(f"\n{'='*80}")
            print("CUDA Kernel Usage Statistics:")
            for kernel, count in self.kernel_stats.items():
                pct = count / total * 100
                print(f"  {kernel:15s}: {count:6d} calls ({pct:5.1f}%)")
            print(f"{'='*80}\n")

########################################################################################################
# TMix and CMix
########################################################################################################

@MyStatic
def RWKV_x070_TMix_one(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    xx = x_prev[0] - x
    x_prev[0] = x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2
    kk = F.normalize((k * k_k).view(H,N), dim=-1, p=2.0).view(H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    w = torch.sigmoid(w0 + w)
    xx = RWKV7_ONE_OP(state, r, w, k, v, -kk, kk*a)

    xx = F.group_norm(xx.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)    
    xx = xx + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
    return (xx * g) @ O_, v_first

@MyStatic
def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    T = x.shape[0]
    xx = torch.cat((x_prev[0].unsqueeze(0), x[:-1,:])) - x
    x_prev[0] = x[-1,:]
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = F.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    w = torch.sigmoid(w0 + w)
    xx = RWKV7_OP(state, r, w, k, v, -kk, kk*a)

    xx = F.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
    xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
    return (xx * g) @ O_, v_first

@MyStatic
def RWKV_x070_TMix_seq_batch(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    B,T,C = x.shape
    xx = torch.cat((x_prev[0].unsqueeze(1), x[:,:-1,:]), dim=1) - x
    x_prev[0] = x[:,-1,:]
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = F.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    w = torch.sigmoid(w0 + w)
    xx = RWKV7_BATCH_OP(state, r, w, k, v, -kk, kk*a)

    xx = F.group_norm(xx.view(B*T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(B,T,H*N)
    xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
    return (xx * g) @ O_, v_first

@MyStatic
def RWKV_x070_CMix_one(x, x_prev, x_k, K_, V_):
    xx = x_prev[1] - x
    x_prev[1] = x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_

@MyStatic
def RWKV_x070_CMix_seq(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev[1].unsqueeze(0), x[:-1,:])) - x
    x_prev[1] = x[-1,:]
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_

@MyStatic
def RWKV_x070_CMix_seq_batch(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev[1].unsqueeze(1), x[:,:-1,:]), dim=1) - x
    x_prev[1] = x[:,-1,:]
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_

