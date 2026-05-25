#include <torch/extension.h>
#include "ATen/ATen.h"

typedef at::BFloat16 bf16;

void cuda_tmix_mix6_infer(
    int T, int C,
    bf16 *x, bf16 *x_prev,
    bf16 *x_r, bf16 *x_w, bf16 *x_k, bf16 *x_v, bf16 *x_a, bf16 *x_g,
    bf16 *xr, bf16 *xw, bf16 *xk, bf16 *xv, bf16 *xa, bf16 *xg);

void forward(int64_t T, int64_t C,
             torch::Tensor &x, torch::Tensor &x_prev,
             torch::Tensor &x_r, torch::Tensor &x_w, torch::Tensor &x_k,
             torch::Tensor &x_v, torch::Tensor &x_a, torch::Tensor &x_g,
             torch::Tensor &xr, torch::Tensor &xw, torch::Tensor &xk,
             torch::Tensor &xv, torch::Tensor &xa, torch::Tensor &xg) {
    cuda_tmix_mix6_infer(T, C,
        x.data_ptr<bf16>(), x_prev.data_ptr<bf16>(),
        x_r.data_ptr<bf16>(), x_w.data_ptr<bf16>(), x_k.data_ptr<bf16>(),
        x_v.data_ptr<bf16>(), x_a.data_ptr<bf16>(), x_g.data_ptr<bf16>(),
        xr.data_ptr<bf16>(), xw.data_ptr<bf16>(), xk.data_ptr<bf16>(),
        xv.data_ptr<bf16>(), xa.data_ptr<bf16>(), xg.data_ptr<bf16>());
}

// Schema string with explicit Tensor(X!) for each of the 6 mutable
// output buffers. Required for CUDA Graph capture — see wkv7s_op.cpp
// for the full rationale (without the schema PyTorch silently drops
// the kernel from the captured graph).
TORCH_LIBRARY(tmix_mix6_infer, m) {
    m.def(
        "forward("
        "int T, int C, "
        "Tensor x, Tensor x_prev, "
        "Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, "
        "Tensor(a!) xr, Tensor(b!) xw, Tensor(c!) xk, "
        "Tensor(d!) xv, Tensor(e!) xa, Tensor(f!) xg"
        ") -> ()",
        forward);
}
