// PyTorch C++ binding for WKV7 State CUDA kernel
#include <torch/extension.h>
#include <cuda_bf16.h>
using bf = __nv_bfloat16;

void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*s0, bf*y, float*s, float*sa, bf*sT);
void cuda_backward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy, float*s, float*sa, bf*s0, bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da, bf*ds0);

void forward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, 
             torch::Tensor &s0, torch::Tensor &y, torch::Tensor &s, torch::Tensor &sa, torch::Tensor &sT) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_forward(B, T, H, 
        (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)z.data_ptr(), (bf*)a.data_ptr(), 
        (bf*)s0.data_ptr(), (bf*)y.data_ptr(), (float*)s.data_ptr(), (float*)sa.data_ptr(), (bf*)sT.data_ptr());
}

void backward(torch::Tensor &w, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &z, torch::Tensor &a, 
              torch::Tensor &dy, torch::Tensor &s, torch::Tensor &sa, torch::Tensor &s0,
              torch::Tensor &dw, torch::Tensor &dq, torch::Tensor &dk, torch::Tensor &dv, torch::Tensor &dz, torch::Tensor &da, torch::Tensor &ds0) {
    int B = w.sizes()[0], T = w.sizes()[1], H = w.sizes()[2];
    cuda_backward(B, T, H, 
        (bf*)w.data_ptr(), (bf*)q.data_ptr(), (bf*)k.data_ptr(), (bf*)v.data_ptr(), (bf*)z.data_ptr(), (bf*)a.data_ptr(), 
        (bf*)dy.data_ptr(), (float*)s.data_ptr(), (float*)sa.data_ptr(), (bf*)s0.data_ptr(),
        (bf*)dw.data_ptr(), (bf*)dq.data_ptr(), (bf*)dk.data_ptr(), (bf*)dv.data_ptr(), (bf*)dz.data_ptr(), (bf*)da.data_ptr(), (bf*)ds0.data_ptr());
}

TORCH_LIBRARY(wkv7_state, m) {
    m.def("forward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor s0, Tensor(a!) y, Tensor(b!) s, Tensor(c!) sa, Tensor(d!) sT) -> ()");
    m.def("backward(Tensor w, Tensor q, Tensor k, Tensor v, Tensor z, Tensor a, Tensor dy, Tensor s, Tensor sa, Tensor s0, Tensor(a!) dw, Tensor(b!) dq, Tensor(c!) dk, Tensor(d!) dv, Tensor(e!) dz, Tensor(f!) da, Tensor(g!) ds0) -> ()");
}

TORCH_LIBRARY_IMPL(wkv7_state, CUDA, m) {
    m.impl("forward", &forward);
    m.impl("backward", &backward);
}
