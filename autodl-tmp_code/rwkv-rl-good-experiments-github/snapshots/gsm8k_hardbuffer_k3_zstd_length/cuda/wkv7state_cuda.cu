// RWKV7 State-Tuning CUDA Kernel
// 基于 wkv7_cuda.cu 修改，添加初始state支持
// 参考 wkv6state_cuda.cu 的state处理方式

#include <cuda_bf16.h>
#include <assert.h>

using bf = __nv_bfloat16;
__device__ inline float to_float(const bf & u) { return __bfloat162float(u); }
__device__ inline bf to_bf(const float & u) { return __float2bfloat16_rn(u); }

typedef bf * __restrict__ F_;

// ============================================================================
// Forward Kernel - 支持初始state，输出final state
// ============================================================================
__global__ void forward_kernel(int T, int H, 
    F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, 
    F_ s0_,      // 初始state (B, H, C, C)
    bf* y_, 
    float* s_,   // 中间state用于backward
    float* sa_,
    bf* sT_)     // 输出final state (B, H, C, C)
{
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;

    // 从s0加载初始state，而不是初始化为0
    float state[C];
    #pragma unroll
    for (int j = 0; j < C; j++) {
        state[j] = to_float(s0_[bb*H*C*C + hh*C*C + i*C + j]);
    }
    
    __shared__ float q[C], k[C], w[C], a[C], b[C];

    for (int t = 0; t < T; t++) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        q[i] = to_float(q_[ind]);
        w[i] = __expf(-__expf(to_float(w_[ind])));
        k[i] = to_float(k_[ind]);
        a[i] = to_float(a_[ind]);
        b[i] = to_float(b_[ind]);
        __syncthreads();

        float sa = 0;
        #pragma unroll
        for (int j = 0; j < C; j++) {
            sa += a[j] * state[j];
        }
        sa_[ind] = sa;

        float v = to_float(v_[ind]);
        float y = 0;
        #pragma unroll
        for (int j = 0; j < C; j++) {
            float& s = state[j];
            s = s * w[j] + sa * b[j] + k[j] * v;
            y += s * q[j];
        }
        y_[ind] = to_bf(y);

        // 保存中间state用于backward
        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i;
            #pragma unroll
            for (int j = 0; j < C; j++) {
                s_[base + j*C] = state[j];
            }
        }
    }
    
    // 输出final state
    #pragma unroll
    for (int j = 0; j < C; j++) {
        sT_[bb*H*C*C + hh*C*C + i*C + j] = to_bf(state[j]);
    }
}

// ============================================================================
// Backward Kernel - 计算ds0梯度
// ============================================================================
__global__ void backward_kernel(int T, int H, 
    F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, 
    F_ dy_, 
    float * __restrict__ s_, 
    float * __restrict__ sa_, 
    F_ s0_,      // 初始state，用于重计算
    bf* dw_, bf* dq_, bf* dk_, bf* dv_, bf* da_, bf* db_,
    bf* ds0_)    // state梯度输出 (B, H, C, C)
{
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;

    float stateT[C] = {0}, dstate[C] = {0}, dstateT[C] = {0};
    __shared__ float w[C], q[C], k[C], v[C], a[C], b[C], dy[C], sa[C], dSb_shared[C];
    float qi, wi, ki, ai, bi, dyi;

    for (int t = T-1; t >= 0; t--) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        q[i] = qi = to_float(q_[ind]);
        float wi_fac = -__expf(to_float(w_[ind]));
        w[i] = wi = __expf(wi_fac);
        k[i] = ki = to_float(k_[ind]);
        a[i] = ai = to_float(a_[ind]);
        b[i] = bi = to_float(b_[ind]);
        v[i] = to_float(v_[ind]);
        dy[i] = dyi = to_float(dy_[ind]);
        sa[i] = sa_[ind];
        __syncthreads();

        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i*C;
            #pragma unroll
            for (int j = 0; j < C; j++) {
                stateT[j] = s_[base + j];
            }
        }

        float dq = 0;
        #pragma unroll
        for (int j = 0; j < C; j++) {
            dq += stateT[j]*dy[j];
        }
        dq_[ind] = to_bf(dq);

        float iwi = 1.0f/wi;
        #pragma unroll        
        for (int j = 0; j < C; j++) {
            stateT[j] = (stateT[j] - ki*v[j] - bi*sa[j]) * iwi;
            dstate[j] += dyi * q[j];
            dstateT[j] += qi * dy[j];
        }

        float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0;
        #pragma unroll
        for (int j = 0; j < C; j++) {
            dw += dstateT[j]*stateT[j];
            dk += dstateT[j]*v[j];
            dv += dstate[j]*k[j];
            dSb += dstate[j]*b[j];
            db += dstateT[j]*sa[j];
        }
        dw_[ind] = to_bf(dw * wi * wi_fac);
        dk_[ind] = to_bf(dk);
        dv_[ind] = to_bf(dv);
        db_[ind] = to_bf(db);

        __syncthreads();
        dSb_shared[i] = dSb;
        __syncthreads();

        float da = 0;
        #pragma unroll
        for (int j = 0; j < C; j++) {
            da += stateT[j]*dSb_shared[j];
        }
        da_[ind] = to_bf(da);

        #pragma unroll        
        for (int j = 0; j < C; j++) {
            dstate[j] = dstate[j]*w[j] + dSb * a[j];
            dstateT[j] = dstateT[j]*wi + ai * dSb_shared[j];
        }
    }
    
    // 输出初始state的梯度
    #pragma unroll
    for (int j = 0; j < C; j++) {
        ds0_[bb*H*C*C + hh*C*C + i*C + j] = to_bf(dstate[j]);
    }
}

// ============================================================================
// C++ Interface
// ============================================================================
void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*s0, bf*y, float*s, float*sa, bf*sT) {
    forward_kernel<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,s0,y,s,sa,sT);
}

void cuda_backward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy, float*s, float*sa, bf*s0, bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da, bf*ds0) {
    assert(T%_CHUNK_LEN_ == 0);
    backward_kernel<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,dy,s,sa,s0,dw,dq,dk,dv,dz,da,ds0);
}
