# Albatross

efficient RWKV inference engine

Please check this first: https://github.com/BlinkDL/Albatross/blob/main/benchmark.py

Faster fwd & bwd CUDA kernels: https://github.com/BlinkDL/RWKV-CUDA/tree/main/rwkv7_fast_fused

Faster sampling: https://github.com/Triang-jyed-driung/Rapid-Sampling

## Result @ 251201

145+ token/s RWKV-7 7.2B fp16 bsz1 @ RTX5090

11289 token/s RWKV-7 7.2B fp16 bsz1 prefill @ RTX5090

Code: https://github.com/Triang-jyed-driung/Albatross/tree/fp16

## Result @ 251103

10250+ token/s RWKV-7 7.2B fp16 bsz960 @ RTX5090

123+ token/s RWKV-7 7.2B fp16 bsz1 @ RTX5090 with CUDAGraph and sparse FFN (lossless)

Code: https://github.com/BlinkDL/Albatross/tree/main/faster_251101

## Result @ 251007

1.3x 7B decoding and 5x 0.1B decoding, with CUDAGraph.

## Result @ 250909

Now with batch inference. 7B fp16 bsz 320 = 5848 token/s decoding (const speed & vram because it's RNN) on 5090. I think 10000 token/s is achievable (full fp16 precision).

## Result @ 250904

Baseline performance for RWKV-7 7.2B bsz=1 @ RTX5090, simply abysmal lol

Let me know if you can find simple methods (such as tuning torch.compile etc.) to improve these a bit
```
Token/s = 75.1 (forward), 73.76 (full) || Bandwidth = 1041.2 GB/s || 3.722s

CTX_LEN 512 : avg loss 1.6548 || prefill 9163 token/s = 127.03 TFLOPS
CTX_LEN 1024 : avg loss 1.5689 || prefill 9742 token/s = 135.06 TFLOPS
CTX_LEN 2048 : avg loss 1.5141 || prefill 10081 token/s = 139.76 TFLOPS
CTX_LEN 4096 : avg loss 1.4824 || prefill 10427 token/s = 144.55 TFLOPS
```
