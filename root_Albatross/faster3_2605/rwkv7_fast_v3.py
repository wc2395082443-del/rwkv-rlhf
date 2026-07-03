#!/usr/bin/env python3
import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

HEAD_SIZE = 64
DTYPE = torch.float16
MODEL_PATH = "/dev/shm/rwkv7-g1f-7.2b-20260414-ctx8192.pth"
THIS_DIR = Path(__file__).resolve().parent
CUDA_DIR = THIS_DIR / "cuda"
L,C,H,N,V = 0,0,0,HEAD_SIZE,0
WKV_MODE = "fp16"
EMB_DEVICE = "gpu"
RKV_MODE = "auto"
CMIX_SPARSE = "auto"
CMIX_B1T1_SPARSE = "b1t1_sparse"
CMIX_ROWS2_SPARSE = "rows2_sparse"
CMIX_B1T1_NOFC = "b1t1_nofc"
CMIX_ROWS2_NOFC = "rows2_nofc"
CMIX_DENSE = "dense"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--cases", default="1x1,1x2,1x4,1x8,1x16,1x32,1x64,1x128,1x256,2x1,4x1,8x1,16x1,32x1,64x1,128x1,256x1,2x2,4x4,8x8,16x16") # try 1x1024 1024x1 32x32 for extreme tps
    parser.add_argument("--profile-range", action="store_true")
    parser.add_argument("--eval-json", default="")
    parser.add_argument("--eval-out", default="")
    parser.add_argument("--eval-all-logits-out", default="")
    parser.add_argument("--eval-paths", default="b1tn")
    parser.add_argument("--wkv", choices=("fp16", "fp32io16"), default="fp16") # fp32io16 is more accurate
    parser.add_argument("--emb", choices=("gpu", "cpu"), default="cpu") # cpu is fast too, and saves VRAM
    parser.add_argument("--batched-rkv", choices=("auto", "on", "off"), default="off") # auto is slightly faster but consumes lots of VRAM
    parser.add_argument("--cmix-sparse", choices=("auto", "no-fc", "off"), default="no-fc") # auto is slightly faster but consumes lots of VRAM
    args = parser.parse_args()

    global WKV_MODE, EMB_DEVICE, RKV_MODE, CMIX_SPARSE
    WKV_MODE = args.wkv
    EMB_DEVICE = args.emb
    RKV_MODE = args.batched_rkv
    CMIX_SPARSE = args.cmix_sparse
    log(f"start model={MODEL_PATH} wkv={WKV_MODE} emb={EMB_DEVICE} batched_rkv={RKV_MODE} cmix_sparse={CMIX_SPARSE}")
    load_extensions(WKV_MODE)
    model = RWKV7()
    if args.eval_json:
        run_eval(model, args.eval_json, args.eval_out, args.eval_all_logits_out, args.eval_paths)
        return
    print("csv_header,label,B,T,iters,p10_ms,p50_ms,p90_ms,tok_s_p50", flush=True)
    for item in args.cases.replace(",", " ").split():
        B, T = [int(x) for x in item.lower().split("x", 1)]
        bench_case(model, B, T, args.warmup, args.iters, args.profile_range)

def log(message: str) -> None:
    print(f"[rwkv7_fast_v3] {message}", flush=True)

def cuda_mem() -> str:
    if not torch.cuda.is_available():
        return "cuda=unavailable"
    free, total = torch.cuda.mem_get_info()
    used = total - free
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    return f"gpu_mem used={used/2**30:.2f}GiB allocated={allocated/2**30:.2f}GiB reserved={reserved/2**30:.2f}GiB total={total/2**30:.2f}GiB"

@dataclass(frozen=True)
class PathConfig:
    rows: int
    use_batched_rkv: bool
    cmix_mode: str

def select_path(B: int, T: int) -> PathConfig:
    """All B/T dependent fast-path choices live here."""
    rows = B*T
    if CMIX_SPARSE == "off":
        cmix_mode = CMIX_DENSE
    elif CMIX_SPARSE == "no-fc":
        cmix_mode = CMIX_B1T1_NOFC if rows == 1 else (CMIX_ROWS2_NOFC if rows == 2 else CMIX_DENSE)
    elif rows == 1:
        cmix_mode = CMIX_B1T1_SPARSE
    elif rows == 2:
        cmix_mode = CMIX_ROWS2_SPARSE
    else:
        cmix_mode = CMIX_DENSE
    if RKV_MODE == "auto":
        use_batched_rkv = 4 <= rows <= 64
    elif RKV_MODE == "on":
        use_batched_rkv = True
    else:
        use_batched_rkv = False
    return PathConfig(rows=rows, use_batched_rkv=use_batched_rkv, cmix_mode=cmix_mode)

def load_extensions(wkv_mode: str = "fp16") -> None:
    t0 = time.perf_counter()
    log(f"loading CUDA extensions fast_ops + wkv={wkv_mode}")
    cuda_flags = ["-O3", "--use_fast_math", "--extra-device-vectorization"] + ([] if os.name == "nt" else ["-Xptxas", "-O3"])
    load(name="rwkv7_fast_ops_fp16", sources=[str(CUDA_DIR / "rwkv7_fast_ops_fp16.cpp"), str(CUDA_DIR / "rwkv7_fast_ops_fp16.cu")], is_python_module=False, verbose=False, extra_cflags=["-O3"], extra_cuda_cflags=cuda_flags)
    if wkv_mode == "fp16":
        load(name="rwkv7_wkv_fp16_v2", sources=[str(CUDA_DIR / "rwkv7_wkv_fp16_v2.cpp"), str(CUDA_DIR / "rwkv7_wkv_fp16_v2.cu")], is_python_module=False, verbose=False, extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-res-usage", "--extra-device-vectorization", "-Xptxas", "-O3"])
    elif wkv_mode == "fp32io16":
        load(name="rwkv7_wkv_fp32_v2", sources=[str(CUDA_DIR / "rwkv7_wkv_fp32_v2.cpp"), str(CUDA_DIR / "rwkv7_wkv_fp32_v2.cu")], is_python_module=False, verbose=False, extra_cflags=["-O3", "-D_IO_FP16_"], extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas", "-O3", "-D_IO_FP16_"])
    else:
        raise ValueError(f"unknown wkv_mode: {wkv_mode}")
    log(f"CUDA extensions loaded in {time.perf_counter() - t0:.3f}s")

class RWKV7:
    def __init__(self) -> None:
        global L, C, H, N, V
        torch.set_grad_enabled(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch._C._jit_set_autocast_mode(False)

        t0 = time.perf_counter()
        log(f"loading weights from {MODEL_PATH}")
        z = torch.load(MODEL_PATH, map_location="cpu", mmap=True)
        log(f"weights mmap loaded in {time.perf_counter() - t0:.3f}s tensors={len(z)}")

        H, N = z["blocks.0.att.r_k"].shape
        C, V = H * N, z["emb.weight"].shape[0]
        assert N == HEAD_SIZE
        log(f"detected model C={C} H={H} N={N} V={V}")

        emb_cpu = z["emb.weight"].squeeze() if EMB_DEVICE == "cpu" else None
        max_layer = -1
        t0 = time.perf_counter()
        log(f"moving and preprocessing weights to CUDA emb={EMB_DEVICE}")
        for key in list(z.keys()):
            if key == "emb.weight" and emb_cpu is not None:
                continue
            value = z[key].squeeze()
            if ".ffn.key.weight" in key and CMIX_SPARSE == "auto":
                z[key + ".fc"] = value.to(device="cuda", dtype=DTYPE).contiguous()
            if (
                "key.weight" in key
                or "value.weight" in key
                or "receptance.weight" in key
                or "output.weight" in key
                or "head.weight" in key
            ):
                value = value.t()
            value = value.to(device="cuda", dtype=DTYPE).contiguous()
            if key.endswith("att.r_k"):
                value = value.flatten().contiguous()
            z[key] = value
            parts = key.split(".")
            if parts[0] == "blocks":
                max_layer = max(max_layer, int(parts[1]))

        L = max_layer + 1
        if emb_cpu is None:
            z["emb.weight"] = F.layer_norm(z["emb.weight"], (C,), weight=z["blocks.0.ln0.weight"], bias=z["blocks.0.ln0.bias"]).contiguous()
        else:
            emb = torch.empty((V,C), dtype=DTYPE, pin_memory=True)
            for start in range(0, V, 4096):
                end = min(start + 4096, V)
                chunk = emb_cpu[start:end].to(device="cuda", dtype=DTYPE)
                chunk = F.layer_norm(chunk, (C,), weight=z["blocks.0.ln0.weight"], bias=z["blocks.0.ln0.bias"])
                emb[start:end].copy_(chunk)
            z["emb.weight"] = emb
        if RKV_MODE != "off":
            for layer in range(L):
                p = f"blocks.{layer}.att."
                z[p+"rkv.weight"] = torch.stack((z[p+"receptance.weight"], z[p+"key.weight"], z[p+"value.weight"])).contiguous()
        self.z = z
        self.emb_cpu = EMB_DEVICE == "cpu"
        self.emb_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        torch.cuda.synchronize()
        log(f"model ready in {time.perf_counter() - t0:.3f}s L={L} C={C} H={H} N={N} V={V}")
        log(cuda_mem())

    def zero_state(self, B: int) -> list[torch.Tensor]:
        return [
            torch.zeros((L,2,B,C), dtype=DTYPE, device="cuda"),
            torch.zeros((L,B,H,N,N), dtype=torch.float32 if WKV_MODE == "fp32io16" else DTYPE, device="cuda"),
            torch.zeros((B,), dtype=torch.int32, device="cuda"),
        ]

    def forward(self, tokens: torch.Tensor, state: list[torch.Tensor]) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        path = select_path(B, T)
        x = self.embed(tokens)
        return self.forward_from_x(x, state, path)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.emb_cpu:
            return self.z["emb.weight"][tokens]
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        host, dev = self.emb_cache.get((B, T), (None, None))
        if host is None:
            host = torch.empty((B*T,C), dtype=DTYPE, pin_memory=True)
            dev = torch.empty((B,T,C), dtype=DTYPE, device="cuda")
            self.emb_cache[(B, T)] = (host, dev)
        flat = tokens.reshape(-1)
        if flat.device.type != "cpu":
            flat = flat.cpu()
        torch.index_select(self.z["emb.weight"], 0, flat, out=host)
        dev.copy_(host.view(B,T,C), non_blocking=True)
        return dev

    def forward_from_x(self, x: torch.Tensor, state: list[torch.Tensor], path: PathConfig, all_logits: bool = False) -> torch.Tensor:
        z = self.z
        B, T, _ = x.shape
        v_first = torch.empty_like(x)

        for layer in range(L):
            p = f"blocks.{layer}."
            xx = F.layer_norm(x, (C,), weight=z[p+"ln1.weight"], bias=z[p+"ln1.bias"])
            xx, v_first = self.tmix(layer, xx, state[0][layer], state[1][layer], state[2], v_first, p+"att.", path)
            x = x + xx
            xx = F.layer_norm(x, (C,), weight=z[p+"ln2.weight"], bias=z[p+"ln2.bias"])
            x = x + self.cmix(xx, state[0][layer], p+"ffn.", path)

        if not all_logits:
            x = x[:, -1, :]
        x = F.layer_norm(x, (C,), weight=z["ln_out.weight"], bias=z["ln_out.bias"])
        state[2].add_(T) # !!! IMPORTANT FOR WKV16 DITHERING !!!
        return x @ z["head.weight"]

    def forward_all_logits(self, tokens: torch.Tensor, state: list[torch.Tensor]) -> torch.Tensor:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        B, T = tokens.shape
        path = select_path(B, T)
        x = self.embed(tokens)
        return self.forward_from_x(x, state, path, all_logits=True)

    def tmix(self, layer: int, x: torch.Tensor, shift_state: torch.Tensor, wkv_state: torch.Tensor, elapsed_t: torch.Tensor, v_first: torch.Tensor, p: str, path: PathConfig) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B, T, _ = x.shape
        xr, xw, xk, xv, xa, xg = ops.tmix_mix6(B, T, C, x.contiguous(), shift_state[0], z[p+"x_r"], z[p+"x_w"], z[p+"x_k"], z[p+"x_v"], z[p+"x_a"], z[p+"x_g"])
        if path.use_batched_rkv:
            flat = torch.stack((xr.reshape(-1,C), xk.reshape(-1,C), xv.reshape(-1,C)))
            rkv = torch.bmm(flat, z[p+"rkv.weight"])
            r, k, v = [t.view(B,T,C) for t in rkv.unbind(0)]
        else:
            r = xr @ z[p+"receptance.weight"]
            k = xk @ z[p+"key.weight"]
            v = xv @ z[p+"value.weight"]

        w = ops.act_tanh((xw @ z[p+"w1"]).contiguous()) @ z[p+"w2"]
        a = (xa @ z[p+"a1"]) @ z[p+"a2"]
        g = ops.act_sigmoid((xg @ z[p+"g1"]).contiguous()) @ z[p+"g2"]
        k, neg_kk, kka = ops.tmix_kk_a_gate(B, T, C, H, k.contiguous(), z[p+"k_k"], z[p+"a0"], a.contiguous(), z[p+"k_a"])

        if layer == 0:
            v_first = v
        else:
            v12 = (xv @ z[p+"v1"]) @ z[p+"v2"]
            v = ops.tmix_vres_gate(B, T, C, v.contiguous(), v_first.contiguous(), z[p+"v0"], v12.contiguous())

        w_raw = ops.add_vec(C, w.contiguous(), z[p+"w0"])
        y = torch.empty_like(r)
        if WKV_MODE == "fp32io16":
            torch.ops.rwkv7_wkv_fp32_v2.forward(B, T, C, H, wkv_state, r.contiguous(), w_raw.contiguous(), k.contiguous(), v.contiguous(), neg_kk.contiguous(), kka.contiguous(), y)
        else:
            torch.ops.rwkv7_wkv_fp16_v2.wkv_seq(B, T, C, H, wkv_state, r.contiguous(), w_raw.contiguous(), k.contiguous(), v.contiguous(), neg_kk.contiguous(), kka.contiguous(), y, elapsed_t)
        y = ops.tmix_lnx_rkvres_xg(B, T, C, H, y.contiguous(), r.contiguous(), k.contiguous(), v.contiguous(), z[p+"r_k"], z[p+"ln_x.weight"], z[p+"ln_x.bias"], g.contiguous())
        return y @ z[p+"output.weight"], v_first

    def cmix(self, x: torch.Tensor, shift_state: torch.Tensor, p: str, path: PathConfig) -> torch.Tensor:
        z = self.z
        ops = torch.ops.rwkv7_fast_ops_fp16
        B, T, _ = x.shape

        if path.cmix_mode == CMIX_B1T1_SPARSE:
            return ops.cmix_sparse_one(C, z[p+"key.weight.fc"].size(0), x.contiguous(), shift_state[1], z[p+"x_k"], z[p+"key.weight.fc"], z[p+"value.weight"])
        if path.cmix_mode == CMIX_ROWS2_SPARSE:
            return ops.cmix_sparse_rows(B, T, C, z[p+"key.weight.fc"].size(0), x.contiguous(), shift_state[1], z[p+"x_k"], z[p+"key.weight.fc"], z[p+"value.weight"])

        mixed = ops.cmix_mix(B, T, C, x.contiguous(), shift_state[1], z[p+"x_k"])
        hid = mixed @ z[p+"key.weight"]
        if path.cmix_mode == CMIX_B1T1_NOFC:
            return ops.cmix_sparse_down_relu_one(C, z[p+"value.weight"].size(0), hid.view(-1).contiguous(), z[p+"value.weight"])
        if path.cmix_mode == CMIX_ROWS2_NOFC:
            return ops.cmix_sparse_down_relu_rows(B, T, C, z[p+"value.weight"].size(0), hid.contiguous(), z[p+"value.weight"])

        k = ops.relu_square(hid.contiguous())
        return k @ z[p+"value.weight"]

def bench_case(model: RWKV7, B: int, T: int, warmup: int, iters: int, profile_range: bool) -> None:
    def percentile(values: list[float], q: float) -> float:
        return float(torch.quantile(torch.tensor(values, dtype=torch.float64), q / 100.0).item())

    state = model.zero_state(B)
    token_device = "cpu" if model.emb_cpu else "cuda"
    tokens = torch.arange(B*T, dtype=torch.long, device=token_device).view(B,T)
    tokens = (tokens * 1103515245 + 12345) % V
    path = select_path(B, T)
    x = model.embed(tokens) if model.emb_cpu else None
    for _ in range(warmup):
        if x is None:
            model.forward(tokens, state)
        else:
            model.forward_from_x(x, state, path)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if x is None:
            model.forward(tokens, state)
        else:
            model.forward_from_x(x, state, path)
    torch.cuda.synchronize()

    times = []
    if profile_range:
        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    if profile_range:
        torch.cuda.cudart().cudaProfilerStop()

    p10 = percentile(times, 10)
    p50 = percentile(times, 50)
    p90 = percentile(times, 90)
    tok_s = B*T*1000.0 / p50
    print(f"RESULT B={B} T={T} iters={iters} p10_ms={p10:.4f} p50_ms={p50:.4f} p90_ms={p90:.4f} tok_s_p50={tok_s:.2f}", flush=True)
    print(f"csv,rwkv7_fast_v3,{B},{T},{iters},{p10:.6f},{p50:.6f},{p90:.6f},{tok_s:.6f}", flush=True)

def run_eval(model: RWKV7, eval_json: str, eval_out: str, logits_out: str, paths: str) -> None:
    with open(eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = data["tokens"]
    outputs = {}
    for path in paths.replace(",", " ").split():
        token_device = "cpu" if model.emb_cpu else "cuda"
        targets = torch.tensor(ids[1:], dtype=torch.long, device="cuda")
        state = model.zero_state(1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if path == "b1tn":
            tokens = torch.tensor(ids[:-1], dtype=torch.long, device=token_device).view(1, -1)
            logits = model.forward_all_logits(tokens, state).squeeze(0).float()
            loss = F.cross_entropy(logits, targets, reduction="none")
        elif path == "b1t1":
            losses = []
            for i, tok in enumerate(ids[:-1]):
                token = torch.tensor([[tok]], dtype=torch.long, device=token_device)
                logits = model.forward(token, state).float()
                losses.append(F.cross_entropy(logits, targets[i:i + 1], reduction="none"))
            loss = torch.cat(losses)
            logits = None
        else:
            raise ValueError(f"unknown eval path: {path}")
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        loss_cpu = loss.detach().cpu()
        p90 = torch.quantile(loss_cpu.float(), 0.90).item()
        p99 = torch.quantile(loss_cpu.float(), 0.99).item()
        tok_s = loss_cpu.numel() / dt
        print(
            f"EVAL label=rwkv7_fast_v3 path={path} positions={loss_cpu.numel()} "
            f"mean_loss={loss_cpu.mean().item():.8f} p90_loss={p90:.8f} p99_loss={p99:.8f} "
            f"max_loss={loss_cpu.max().item():.8f} min_loss={loss_cpu.min().item():.8f} "
            f"time_s={dt:.3f} tok_s={tok_s:.3f}",
            flush=True,
        )
        if logits_out and path == "b1tn":
            torch.save(logits.detach().cpu(), logits_out)
        outputs[path] = {
            "label": "rwkv7_fast_v3",
            "path": path,
            "tokens": ids,
            "loss": loss_cpu,
            "mean_loss": float(loss_cpu.mean().item()),
            "p90_loss": float(p90),
            "p99_loss": float(p99),
            "max_loss": float(loss_cpu.max().item()),
            "min_loss": float(loss_cpu.min().item()),
            "time_s": float(dt),
        }
    if eval_out:
        torch.save(outputs, eval_out)

if __name__ == "__main__":
    main()
