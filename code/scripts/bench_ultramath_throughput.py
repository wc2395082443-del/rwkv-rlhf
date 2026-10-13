import importlib.util
import json
import os
import signal
import sys
import time
import torch

sys.path.insert(0, '/root/eval_transfer/albatross/faster3a_2605')
import rwkv7_fast_v3a as backend

spec = importlib.util.spec_from_file_location('eval_probe', '/hy-tmp/evals/ultramath_step1000_pass16_20260910/eval_math500_flower.py')
evaluator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluator
spec.loader.exec_module(evaluator)
worker = int(sys.argv[1])
os.kill(worker, signal.SIGSTOP)
try:
    backend.MODEL_PATH = '/hy-tmp/runs/g1i_step800_to_dapo_math17k_officialprompt_q32x16_cand64_max6144_ob512_continue800to1000_eval384_ckpt50_20260829_1810/checkpoint_step_1000.pth'
    backend.WKV_MODE = 'fp32io16'
    backend.load_extensions(backend.WKV_MODE)
    model = backend.RWKV7()
    for batch in (1280, 1536):
        state = model.zero_state(batch)
        tokens = torch.ones((batch, 1), device='cpu', dtype=torch.long)
        next_tokens = [1] * batch
        for mode in ('scalar', 'bulk'):
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                for iteration in range(50):
                    if mode == 'scalar':
                        tokens.fill_(0)
                        for row in range(batch):
                            tokens[row, 0] = next_tokens[row]
                    else:
                        tokens.copy_(torch.tensor(next_tokens, dtype=torch.long).view(batch, 1))
                    logits = model.forward(tokens, state).view(batch, -1)
                    next_tokens = evaluator.sample_logits(logits, 0.8, 0.35, 40).cpu().tolist()
                    if iteration == 9:
                        torch.cuda.synchronize()
                        started = time.perf_counter()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            print('BENCH_RESULT ' + json.dumps(dict(batch=batch, mode=mode, token_per_sec=batch*40/elapsed,
                                                     allocated_gib=torch.cuda.memory_allocated()/2**30,
                                                     reserved_gib=torch.cuda.memory_reserved()/2**30)), flush=True)
        del state, tokens, logits
        torch.cuda.empty_cache()
finally:
    os.kill(worker, signal.SIGCONT)
