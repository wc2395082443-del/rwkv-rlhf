```bash
# rwkv7_fast_v3a default = fp16 WKV (faster at large bsz). Use `--wkv fp32io16` for the more accurate fp32 WKV state path.
python3 rwkv7_fast_v3a.py --model /dev/shm/rwkv7-g1f-7.2b-20260414-ctx8192.pth --warmup 3 --iters 10
python3 rwkv7_fast_v3a.py --model /dev/shm/rwkv7-g1d-0.1b-20260129-ctx8192.pth --warmup 3 --iters 10
python3 rwkv7_fast_v3a.py --model /dev/shm/rwkv7-g1d-0.4b-20260210-ctx8192.pth --warmup 3 --iters 10
python3 rwkv7_fast_v3a.py --model /dev/shm/rwkv7-g1f-1.5b-20260419-ctx8192.pth --warmup 3 --iters 10
python3 rwkv7_fast_v3a.py --model /dev/shm/rwkv7-g1f-2.9b-20260420-ctx8192.pth --warmup 3 --iters 10
python3 rwkv7_fast_v3a.py --model /dev/shm/rwkv7-g1f-13.3b-20260415-ctx8192.pth --warmup 3 --iters 10

python3 eval_mmlu.py --model /dev/shm/rwkv7-g1f-7.2b-20260414-ctx8192.pth --bsz 256
python3 eval_mmlu.py --model /dev/shm/rwkv7-g1d-0.1b-20260129-ctx8192.pth --bsz 256
python3 eval_mmlu.py --model /dev/shm/rwkv7-g1d-0.4b-20260210-ctx8192.pth --bsz 256
python3 eval_mmlu.py --model /dev/shm/rwkv7-g1f-1.5b-20260419-ctx8192.pth --bsz 256
python3 eval_mmlu.py --model /dev/shm/rwkv7-g1f-2.9b-20260420-ctx8192.pth --bsz 256
python3 eval_mmlu.py --model /dev/shm/rwkv7-g1f-13.3b-20260415-ctx8192.pth --bsz 256

python3 eval_math500.py --model /dev/shm/rwkv7-g1f-1.5b-20260419-ctx8192.pth --bsz 960 --gpus 0,1,2,3 --rollout 16
```
