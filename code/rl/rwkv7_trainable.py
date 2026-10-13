########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from einops import rearrange
import os, math, gc, importlib
import shutil
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import deepspeed

########################################################################################################
# CUDA Kernel
########################################################################################################

HEAD_SIZE = int(os.environ.get("RWKV_HEAD_SIZE_A", "64"))
CHUNK_LEN = 16


def RUN_CUDA_RWKV7g():
    raise NotImplementedError('RUN_CUDA_RWKV7g not implemented')

def RUN_RWKV7_STATE():
    raise NotImplementedError('RUN_RWKV7_STATE not implemented')

# ========================================================================
# CUDA Backend
# ========================================================================
_SRC_DIR = Path(__file__).resolve().parent / "cuda"

def _ensure_ninja_in_path():
    if shutil.which("ninja"):
        return
    conda_prefix = os.environ.get("CONDA_PREFIX")
    candidates = []
    if conda_prefix:
        candidates.append(os.path.join(conda_prefix, "bin", "ninja"))
    candidates += ["/root/miniconda3/bin/ninja", "/usr/bin/ninja", "/bin/ninja"]
    for cand in candidates:
        if os.path.isfile(cand):
            os.environ["PATH"] = os.path.dirname(cand) + os.pathsep + os.environ.get("PATH", "")
            break

from torch.utils.cpp_extension import load
_ensure_ninja_in_path()
if 'x070' in os.environ.get("RWKV_MY_TESTING", ""):

    # 加载标准WKV7 kernel
    flags = [
    '-res-usage',
    f'-D_C_={HEAD_SIZE}', 
    f"-D_CHUNK_LEN_={CHUNK_LEN}", 
    "--use_fast_math", 
    "-Xcompiler", "-O2",      # <--- 关键修改：调用 MSVC 的优化器
    "-Xptxas", "-O3",         # <--- CUDA 内部优化保持不变
    "--extra-device-vectorization",
    "-allow-unsupported-compiler"]

    load(name="wind_backstepping", sources=[str(_SRC_DIR / 'wkv7_cuda.cu'), str(_SRC_DIR / 'wkv7_op.cpp')], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w,q,k,v,z,b):
            B,T,H,C = w.shape 
            assert T%CHUNK_LEN == 0
            assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
            assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
            y = torch.empty_like(v)
            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
            torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
            ctx.save_for_backward(w,q,k,v,z,b,s,sa)
            return y
        @staticmethod
        def backward(ctx, dy):
            assert all(i.dtype==torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            w,q,k,v,z,b,s,sa = ctx.saved_tensors
            dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
            torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
            return dw,dq,dk,dv,dz,db

    def RUN_CUDA_RWKV7g(q,w,k,v,a,b, HEAD_SIZE=64):
        B,T,HC = q.shape
        C = HEAD_SIZE
        H = HC // C

        # Padding
        orig_T = T
        if T % CHUNK_LEN != 0:
            pad_len = CHUNK_LEN - (T % CHUNK_LEN)
            q = F.pad(q, (0, 0, 0, pad_len))
            w = F.pad(w, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            a = F.pad(a, (0, 0, 0, pad_len))
            b = F.pad(b, (0, 0, 0, pad_len))
            T = T + pad_len

        q,w,k,v,a,b = [i.view(B,T,H,C) for i in [q,w,k,v,a,b]]
        y = WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)

        if T != orig_T:
            y = y[:, :orig_T, :].contiguous()
        return y

    # ====================================================================
    # WKV7 State - 支持初始state的CUDA实现
    # ====================================================================

    # 加载state版本kernel
    if os.environ.get("RWKV_TRAIN_TYPE") in ['state', 'fullstate']:
        print("Loading WKV7 State CUDA kernel for state-tuning...")
        load(name="wkv7_state", sources=[str(_SRC_DIR / 'wkv7state_cuda.cu'), str(_SRC_DIR / 'wkv7state_op.cpp')], 
             is_python_module=False, verbose=True, extra_cuda_cflags=flags)

        class WKV7StateFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, w, q, k, v, z, b, s0):
                """
                s0: (H, C, C) - 原始state参数（不是扩展后的）
                """
                B, T, H, C = w.shape
                assert T % CHUNK_LEN == 0
                assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b, s0])
                assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
                assert s0.shape == (H, C, C), f"s0 shape should be (H, C, C)={(H, C, C)}, got {s0.shape}"

                # 在function内部扩展state
                s0_expanded = s0.unsqueeze(0).expand(B, H, C, C).contiguous()

                y = torch.empty_like(v)
                s = torch.empty(B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
                sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
                sT = torch.empty(B, H, C, C, dtype=w.dtype, device=w.device)  # final state

                torch.ops.wkv7_state.forward(w, q, k, v, z, b, s0_expanded, y, s, sa, sT)
                ctx.save_for_backward(w, q, k, v, z, b, s, sa, s0_expanded)
                return y

            @staticmethod
            def backward(ctx, dy):
                assert dy.dtype == torch.bfloat16
                assert dy.is_contiguous()
                w, q, k, v, z, b, s, sa, s0_expanded = ctx.saved_tensors
                B, T, H, C = w.shape

                dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
                ds0 = torch.empty(B, H, C, C, dtype=torch.bfloat16, device=w.device)

                torch.ops.wkv7_state.backward(w, q, k, v, z, b, dy, s, sa, s0_expanded, dw, dq, dk, dv, dz, db, ds0)

                # 对batch维度求和，返回(H, C, C)形状 - 与输入s0形状一致
                ds0_sum = ds0.sum(dim=0)

                return dw, dq, dk, dv, dz, db, ds0_sum

        def RUN_RWKV7_STATE(r, k, v, w, a, b, s, HEAD_SIZE=64):
            """
            带初始state的RWKV7 CUDA实现

            Args:
                r, k, v, w, a, b: (B, T, HC) 输入张量
                s: (H, C, C) 初始state（可训练参数）
                HEAD_SIZE: head维度，默认64

            Returns:
                output: (B, T, HC)
                state: None（当前不返回最终state）
            """
            B, T, HC = r.shape
            C = HEAD_SIZE
            H = HC // C

            # Padding T to multiple of CHUNK_LEN
            orig_T = T
            if T % CHUNK_LEN != 0:
                pad_len = CHUNK_LEN - (T % CHUNK_LEN)
                r = F.pad(r, (0, 0, 0, pad_len))
                k = F.pad(k, (0, 0, 0, pad_len))
                v = F.pad(v, (0, 0, 0, pad_len))
                w = F.pad(w, (0, 0, 0, pad_len))
                a = F.pad(a, (0, 0, 0, pad_len))
                b = F.pad(b, (0, 0, 0, pad_len))
                T = T + pad_len

            # Reshape to (B, T, H, C)
            r_4d = r.view(B, T, H, C).contiguous()
            k_4d = k.view(B, T, H, C).contiguous()
            v_4d = v.view(B, T, H, C).contiguous()
            w_4d = w.view(B, T, H, C).contiguous()
            a_4d = a.view(B, T, H, C).contiguous()
            b_4d = b.view(B, T, H, C).contiguous()

            # state保持(H, C, C)形状，WKV7StateFunction内部会扩展
            s_3d = s.contiguous()

            # 确保dtype为bfloat16
            if r_4d.dtype != torch.bfloat16:
                r_4d = r_4d.to(torch.bfloat16)
                k_4d = k_4d.to(torch.bfloat16)
                v_4d = v_4d.to(torch.bfloat16)
                w_4d = w_4d.to(torch.bfloat16)
                a_4d = a_4d.to(torch.bfloat16)
                b_4d = b_4d.to(torch.bfloat16)
            if s_3d.dtype != torch.bfloat16:
                s_3d = s_3d.to(torch.bfloat16)

            # 调用CUDA kernel
            # 参数顺序: (w, q, k, v, z, b, s0) 其中 q=r, z=a
            y = WKV7StateFunction.apply(w_4d, r_4d, k_4d, v_4d, a_4d, b_4d, s_3d)

            # Reshape back
            y = y.view(B, T, HC)
            if T != orig_T:
                y = y[:, :orig_T, :].contiguous()

            return y, None

elif 'x060' in os.environ.get("RWKV_MY_TESTING", ""):
    # RWKV6 实现
    pass
else:
    # RWKV5 实现
    pass


########################################################################################################
# FFN (from ffn.py)
########################################################################################################

channel_mixing_rwkv7 = None

def RWKV_Cmix_v7(*args, **kwargs):
    if os.environ["RWKV_TRAIN_TYPE"] == 'fullstate':
        return RWKV_CMix_x070_FullState(*args, **kwargs)
    else:
        return RWKV_CMix_x070(*args, **kwargs)

class RWKV_CMix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        # self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        # self.value.weight.data.zero_()

    def forward(self, x, attention_mask=None):
        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])
        xx = self.time_shift(x) - x
        
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)


class RWKV_CMix_x070_FullState(RWKV_CMix_x070):
    def __init__(self, args, layer_id):
        super().__init__(args, layer_id)
        self.args = args
        self.layer_id = layer_id
        self.dim = args.n_embd
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.ts_state = nn.Parameter(torch.zeros(self.dim))

    def forward(self, x, attention_mask=None):
        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])
        
        xx = self.time_shift(x) - x

        xx[:,0,:] += self.ts_state
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)


########################################################################################################
# Attention (from att.py)
########################################################################################################

fused_addcmul_rwkv7 = None
FusedGroupNorm = None

def RWKV_Tmix_v7(*args, **kwargs):
    
    if os.environ["RWKV_TRAIN_TYPE"] == 'state':
        return RWKV_Tmix_x070_State(*args, **kwargs)
    elif os.environ["RWKV_TRAIN_TYPE"] == 'fullstate':
        return RWKV_Tmix_x070_FullState(*args, **kwargs)
    else:
        return RWKV_Tmix_x070(*args, **kwargs)
    
class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.my_testing = args.my_testing

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        self.addcmul_kernel = self.torch_addcmul

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            # D_DECAY_LORA = 64
            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.w0 = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            # D_AAA_LORA = 64
            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C))

            # D_MV_LORA = 32
            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+1.0)

            # D_GATE_LORA = 128
            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            if C==1024:
                D_GATE_LORA = 128
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.k_a = nn.Parameter(torch.ones(1,1,C))
            self.r_k = nn.Parameter(torch.zeros(H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(args.head_size_divisor**2))


            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            # self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            # self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            # self.output.weight.data.zero_()

    def torch_addcmul(self, x, xx):
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g
        return xr, xw, xk, xv, xa, xg
    
    def fused_addcmul(self, x, xx):
        return fused_addcmul_rwkv7(x, xx, self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g)

    #@torch.compile
    def forward(self, x, v_first, attention_mask=None):
        B, T, C = x.size()
        H = self.n_head

        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])
        xx = self.time_shift(x) - x

        xr, xw, xk, xv, xa, xg = self.addcmul_kernel(x, xx)

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        if attention_mask is not None:
            v = v * attention_mask[:, -v.shape[-2]:, None]
        
        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.reshape(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first
  

class RWKV_Tmix_x070_State(RWKV_Tmix_x070):
    def __init__(self, args, layer_id):
        super().__init__(args, layer_id)
        with torch.no_grad():
            #for State-tuning
            self.time_state = nn.Parameter(torch.zeros(self.n_head, self.head_size, self.head_size))


    #@torch.compile
    def forward(self, x, v_first, attention_mask=None):
        B, T, C = x.size()
        H = self.n_head

        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])
        xx = self.time_shift(x) - x

        xr, xw, xk, xv, xa, xg = self.addcmul_kernel(x, xx)

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x , _ = RUN_RWKV7_STATE(r,k,v,w,-kk, kk*a,self.time_state)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first
    


class RWKV_Tmix_x070_FullState(RWKV_Tmix_x070):
    def __init__(self, args, layer_id):
        super().__init__(args, layer_id)
        with torch.no_grad():
            #for State-tuning
            self.time_state = nn.Parameter(torch.zeros(self.n_head, self.head_size, self.head_size))
            self.ts_state = nn.Parameter(torch.zeros(args.n_embd))


    # @torch.compile
    def forward(self, x, v_first, attention_mask=None):
        B, T, C = x.size()
        H = self.n_head

        if attention_mask is not None:
            x = x.mul(attention_mask[:, -x.shape[-2]:, None])
        xx = self.time_shift(x) - x
        xx[:,0,:] += self.ts_state
        xr, xw, xk, xv, xa, xg = self.addcmul_kernel(x, xx)

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x , _ = RUN_RWKV7_STATE(r,k,v,w,-kk, kk*a,self.time_state)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first


########################################################################################################
# Block (from block.py)
########################################################################################################

class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_v7(args, layer_id)  
        self.ffn = RWKV_Cmix_v7(args, layer_id)


    def forward(self, *args, **kwargs):
        return self.forward_normal(*args, **kwargs)

    def forward_normal(self, x, v_first, attention_mask = None):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), v_first, attention_mask = attention_mask)
        x = x + x_attn

        x = x + self.ffn(self.ln2(x), attention_mask = attention_mask)
        return x, v_first


########################################################################################################
# Model (from model.py)
########################################################################################################

class RWKV7(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        兼容 transformers 的 generate() 接口.
        对 RWKV 来说，我们不需要做实际处理，直接返回原始输入即可。
        """
        return {"input_ids": input_ids, **kwargs}

    def get_input_embeddings(self):
        """为 PEFT 提供 Embedding 层引用"""
        return self.emb

    def set_input_embeddings(self, new_emb):
        """允许 PEFT 替换 Embedding 层（通常不会触发）"""
        self.emb = new_emb

    def get_output_embeddings(self):
        """为 PEFT 提供输出 head 层引用"""
        return self.head

    def set_output_embeddings(self, new_head):
        """允许 PEFT 替换输出 head 层"""
        self.head = new_head

    def forward(self, *args, **kwargs):
        return self.forward_normal(*args, **kwargs)

    def forward_normal(self, input_ids, inputs_embeds=None, attention_mask=None, **kwargs):
        args = self.args
        B, T = input_ids.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(input_ids)
        v_first = torch.empty_like(x)

        for block in self.blocks:
            if args.grad_cp == 1:
                if args.train_type == 'state' or args.peft !='none':
                    x, v_first = torch_checkpoint(block, x, v_first , attention_mask, use_reentrant=False)
                else:
                    x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first, attention_mask)
            else:
                x, v_first = block(x, v_first, attention_mask)

        x = self.ln_out(x)
        x = self.head(x)

        return x
