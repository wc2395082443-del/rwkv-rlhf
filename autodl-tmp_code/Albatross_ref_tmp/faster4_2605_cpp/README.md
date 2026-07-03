```bash
cd faster4_2605_cpp
rm -rf bin
cmake -S . -B bin -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build bin -j

# default = fp16 WKV. Use `--wkv32` for the more accurate fp32 WKV state path.
bin/rwkv7_fast_v4 --model /dev/shm/rwkv7-g1f-7.2b-20260414-ctx8192.pth --model-forward --cases '1x1,1x2,1x4,1x8,1x16,1x32,1x64,1x128,1x256,2x1,4x1,8x1,16x1,32x1,64x1,128x1,256x1,2x2,4x4,8x8,16x16' --graph-bench --warmup 3 --iters 10
bin/rwkv7_fast_v4 --model /dev/shm/rwkv7-g1d-0.1b-20260129-ctx8192.pth --model-forward --cases '1x1,1x2,1x4,1x8,1x16,1x32,1x64,1x128,1x256,2x1,4x1,8x1,16x1,32x1,64x1,128x1,256x1,2x2,4x4,8x8,16x16' --graph-bench --warmup 3 --iters 10
bin/rwkv7_fast_v4 --model /dev/shm/rwkv7-g1d-0.4b-20260210-ctx8192.pth --model-forward --cases '1x1,1x2,1x4,1x8,1x16,1x32,1x64,1x128,1x256,2x1,4x1,8x1,16x1,32x1,64x1,128x1,256x1,2x2,4x4,8x8,16x16' --graph-bench --warmup 3 --iters 10
bin/rwkv7_fast_v4 --model /dev/shm/rwkv7-g1f-1.5b-20260419-ctx8192.pth --model-forward --cases '1x1,1x2,1x4,1x8,1x16,1x32,1x64,1x128,1x256,2x1,4x1,8x1,16x1,32x1,64x1,128x1,256x1,2x2,4x4,8x8,16x16' --graph-bench --warmup 3 --iters 10
bin/rwkv7_fast_v4 --model /dev/shm/rwkv7-g1f-2.9b-20260420-ctx8192.pth --model-forward --cases '1x1,1x2,1x4,1x8,1x16,1x32,1x64,1x128,1x256,2x1,4x1,8x1,16x1,32x1,64x1,128x1,256x1,2x2,4x4,8x8,16x16' --graph-bench --warmup 3 --iters 10
bin/rwkv7_fast_v4 --model /dev/shm/rwkv7-g1f-13.3b-20260415-ctx8192.pth --model-forward --cases '1x1,1x2,1x4,1x8,1x16,1x32,1x64,1x128,1x256,2x1,4x1,8x1,16x1,32x1,64x1,128x1,256x1,2x2,4x4,8x8,16x16' --graph-bench --warmup 3 --iters 10
```
