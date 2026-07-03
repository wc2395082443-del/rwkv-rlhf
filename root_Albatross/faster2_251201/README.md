# Albatross
efficient RWKV inference engine

## Usage

Reference environment:
- python 3.12.12
- torch 2.8.0+cu130

Then run the benchmark script:
```
python benchmark.py
```

## 251201 Faster prefill via `cp.async`
```
CTX_LEN 256 : avg loss 1.7936 || prefill 9267 token/s = 128.47 TFLOPS
CTX_LEN 512 : avg loss 1.6549 || prefill 10133 token/s = 140.47 TFLOPS
CTX_LEN 1024 : avg loss 1.5689 || prefill 10750 token/s = 149.04 TFLOPS
CTX_LEN 2048 : avg loss 1.5141 || prefill 10921 token/s = 151.4 TFLOPS
CTX_LEN 4096 : avg loss 1.4825 || prefill 11289 token/s = 156.51 TFLOPS
```

## result @ 251127, CUDA FFN
```
Decode (with compile):
Token/s = 123.53 (forward), 122.86 (full) || Bandwidth = 1712.55 GB/s || 4.1s
Decode (compile + CUDAGraph):
Token/s = 145.98 (forward), 145.25 (full) || Bandwidth = 2023.73 GB/s || 1.763s
```

## result @ 251111, discrete state
```
Decode (with compile):
Token/s = 122.81 (forward), 122.13 (full) || Bandwidth = 1702.55 GB/s || 9.072s
Decode (compile + CUDAGraph):
Token/s = 144.82 (forward), 144.05 (full) || Bandwidth = 2007.68 GB/s || 1.778s
```


## Result @ 251105
```
Decode (with compile):
Token/s = 118.82 (forward), 118.2 (full) || Bandwidth = 1647.26 GB/s || 9.493s
Decode (compile + CUDAGraph):
Token/s = 142.38 (forward), 141.74 (full) || Bandwidth = 1973.9 GB/s || 1.807s
```

## Result @ 251102

```
Decode (with torch.jit):
Token/s = 88.2 (forward), 87.85 (full) || Bandwidth = 1222.69 GB/s || 3.309s
Decode (torch.jit + CUDAGraph):
Token/s = 105.84 (forward), 104.57 (full) || Bandwidth = 1467.27 GB/s || 2.449s

Decode (with compile):
Token/s = 109.4 (forward), 108.86 (full) || Bandwidth = 1516.67 GB/s || 5.451s
Decode (compile + CUDAGraph):
Token/s = 123.87 (forward), 123.36 (full) || Bandwidth = 1717.27 GB/s || 2.075s
```



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


# Proposal for an FP16-Compatible State Evolution Kernel with Deterministic Dithering

I propose a novel approach to implementing a state evolution entirely in **FP16**, leveraging **deterministic dithering**. The goal is to address the numerical challenges inherent in FP16 arithmetic while maintaining near-FP32 accuracy. Below, I detail the reasoning behind each numerical range and the motivations for my design choices.

---

## 1. Reinterpreting the Decay Factor $w$

Traditionally, $w$ is derived from $e^{-z}$. To improve numerical stability, I reinterpret it as:
```math
w' = e^{-z} - 1,
```
and update the state evolution equation accordingly:
```diff
- s = s * w[j] + k[j] * vv + sa * b[j];
+ s += s * w[j] + k[j] * vv + sa * b[j];
```

This change leverages the **`expm1` function's higher accuracy** near $z \approx 0$.

Further, I parameterize $w'$ as:
```math
w' = e^{-0.606531 \cdot \text{Sigmoid}(w)} - 1,
```
where $0.606531 \approx e^{-1/2}$. To optimize for hardware (PTX instruction `ex2.approx.ftz.f32`), I replace $e^x$ with $2^{x \cdot \log_2(e)}$ and fuse coefficients:
```cpp
constexpr float nexp_half_log2_e = -0.8750387749145276f; // == -exp(-1/2) * log2(e)
```

---

## 2. Deterministic Dithering Mechanism

### 2.1 Design of the Rotator Function

The rotator function is defined as:
```cpp
constexpr float two_to_neg_41 = 4.547473508864641e-13f; // == 2^(-41)
constexpr int ro1 = (int)2654435769, ro2 = (int)1779033704, ro3 = (int)3144134277;
#define rotator(_A,_B,_C) (two_to_neg_41*float(ro1*(_A)+ro2*(_B)+ro3*(_C)))
```

#### Key Properties of the Rotator:
1. **Low-Discrepancy Sequences**:
   - The coefficients $\text{ro1}, \text{ro2}, \text{ro3}$ are carefully chosen to produce low-discrepancy sequences.
   - Specifically, $\text{ro1} = \lfloor 4294967296 \cdot \phi \rfloor$, where $\phi$ is the golden ratio ($\approx 1.618$).
   - This ensures that `rotator(t, _, _)` produces sequences with discrepancy bounds of $\Theta(\frac{\log t}{t})$ (L. Kuipers and H. Niederreiter, Uniform distribution of sequences), superior to traditional pseudo-random numbers ( $\Theta(\frac{1}{\sqrt{t \log \log t}})$, law of iterated logarithm).

2. **Range of Values**:
   - $\text{ro1} \cdot A + \text{ro2} \cdot B + \text{ro3} \cdot C \in [-2^{31}, 2^{31}]$.
   - $\texttt{rotator}(A, B, C) \in [-2^{-10}, 2^{-10}]$, equivalent to one or two ULP units of FP16 near 1 (the smallest number greater than 1 is $1.0009765625$).

### 2.2 Dithering Around `exp2f(nexp_half_log2_e * w0) - 1`

The final decay factor is computed as:
```cpp
exp2f(nexp_half_log2_e * w0) - 1 + rotator(t0+_t, i, (int)blockIdx.x)
```

- The term `exp2f(nexp_half_log2_e * w0) - 1` ensures $1 + w$ operates in range $[0.545, 1.0]$.
- The dithering term $\text{rotator}(...)$ introduces a $\pm 2 \text{ULP}$ perturbation with a low-discrepancy pattern, breaking quantization bias.

---

## 3. Performance Comparison

### 3.1 Context Length vs Loss

Performance comparison between FP16 and FP32 implementations is shown below:

| Context Length | FP32 Avg Loss | FP16 Avg Loss | Difference |
|----------------|---------------|---------------|------------|
| 512            | 1.6549        | 1.6550        | +0.0001    |
| 1024           | 1.5689        | 1.5689        | +0.0000    |
| 2048           | 1.5142        | 1.5143        | +0.0001    |
| 4096           | 1.4825        | 1.4826        | +0.0001    |

### 3.2 Key Observations

- The performance loss is minimal: approximately $7 \times 10^{-5}$.
- This demonstrates that the proposed FP16 implementation achieves **near-FP32 accuracy** while significantly reducing memory and computation costs.


# Result @ 251008
Now over 10000 tokens/s on RTX5090 (bsz960). Special thanks to [@blealtan](https://github.com/blealtan) for implementing swizzling for coalesced state r/w. There is still plenty of room for optimization.

# CUDA Functions for Sparse-Vector-Dense-Matrix Multiplication
Now with a even faster kernel for Sparse-Vector-Dense-Matrix Multiplication. 

> Mechanism: only rows `i` where `vector[i] != 0` are being read. This reduces down global memory access significantly.

Estimated around 18us on dimension 16384x4096, this kernel outperforms naive matrix-vector multiplication.

We sincerely thank [FlagOpen/FlagGems](https://github.com/FlagOpen/FlagGems) for first implementing this idea, and [tile-ai/tilelang](https://github.com/tile-ai/tilelang) for code templates for asynchronous copy operations.

```cpp
#define BLOCKDIM 128
#define MAXNPERBLOCK 64
__global__ void __launch_bounds__(BLOCKDIM, 1) spvecmatmul_noindices(
    const int C,
    const half* __restrict__ vec,
    const half* __restrict__ mat,
    half* __restrict__ out
){
    __shared__ __align__(256) half mat_row_smem[2][2*BLOCKDIM];
    __shared__ __align__(256) half vec_slice[MAXNPERBLOCK];
    __shared__ __align__(256) int nnz_ids[MAXNPERBLOCK];
    __shared__ int nnz_count;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int t = threadIdx.x;
    const int start_pos = bx * MAXNPERBLOCK;

    if (t < 32){
        *(half2*)(vec_slice + t*2) = *(const half2*)(vec + start_pos + t*2);
    }
    __syncthreads();
    if (t == 0){
        int cnt = 0;
        #pragma unroll
        for (int i=0; i<8; ++i) {
            common128 z;
            z.I = ((const int4*)vec_slice)[i];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                unsigned short bits = __half_as_ushort(z.h[j]);
                if (bits != 0x0000 && bits != 0x8000) {
                    int idx = i * 8 + j;
                    nnz_ids[cnt] = idx;
                    cnt++;
                }
            }
        }
        nnz_count = cnt;
    }
    __syncthreads();

    half2 out_frag;
    *(int*)(&out_frag) = 0;
    // init
    #pragma unroll
    for(int i = 0; i < 2; i++){
        if (i < nnz_count){
            int actual_pos = start_pos + nnz_ids[i];
            cp_async_gs_conditional<4>(mat_row_smem[i%2] + t*2, mat + actual_pos * C + by * (2*BLOCKDIM) + t*2, true);
            cp_async_commit();
        }
    }
    // main for
    for(int i = 0; i < nnz_count-2; i++){
        // take data
        cp_async_wait<1>();
        __syncthreads();

        half2 mat_row_frag = *(half2*) (mat_row_smem[i%2] + t*2);
        half vec_value = vec_slice[nnz_ids[i]];

        // store
        int actual_pos = start_pos + nnz_ids[i+2];
        cp_async_gs_conditional<4>(mat_row_smem[i%2] + t*2, mat + actual_pos * C + by * (2*BLOCKDIM) + t*2, true);
        cp_async_commit();

        // compute
        out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
    }

    // end
    if (nnz_count >= 2){
        cp_async_wait<1>();
        __syncthreads();

        half2 mat_row_frag = *(half2*) (mat_row_smem[nnz_count%2] + t*2);
        half vec_value = vec_slice[nnz_ids[nnz_count - 2]];

        out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
    }
    if (nnz_count >= 1){
        cp_async_wait<0>();
        __syncthreads();

        half2 mat_row_frag = *(half2*) (mat_row_smem[(nnz_count+1)%2] + t*2);
        half vec_value = vec_slice[nnz_ids[nnz_count - 1]];

        out_frag = __hfma2(__half2half2(vec_value), mat_row_frag, out_frag);
    }
    atomicAdd((half2*)(out + by*(2*BLOCKDIM) + t*2), out_frag);
}
```
