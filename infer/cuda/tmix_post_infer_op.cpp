#include <torch/extension.h>
#include "ATen/ATen.h"

typedef at::BFloat16 bf16;

void cuda_tmix_post_infer(
    int T, int H,
    bf16 *x, bf16 *r, bf16 *k, bf16 *v, bf16 *r_k,
    bf16 *weight, bf16 *bias, bf16 *g, bf16 *out);

void forward(int64_t T, int64_t H,
             torch::Tensor &x, torch::Tensor &r, torch::Tensor &k,
             torch::Tensor &v, torch::Tensor &r_k,
             torch::Tensor &weight, torch::Tensor &bias,
             torch::Tensor &g, torch::Tensor &out) {
    cuda_tmix_post_infer(T, H,
        x.data_ptr<bf16>(), r.data_ptr<bf16>(), k.data_ptr<bf16>(),
        v.data_ptr<bf16>(), r_k.data_ptr<bf16>(),
        weight.data_ptr<bf16>(), bias.data_ptr<bf16>(),
        g.data_ptr<bf16>(), out.data_ptr<bf16>());
}

// Schema string with Tensor(a!) for the mutable output. Required for
// CUDA Graph capture — see wkv7s_op.cpp.
TORCH_LIBRARY(tmix_post_infer, m) {
    m.def(
        "forward("
        "int T, int H, "
        "Tensor x, Tensor r, Tensor k, Tensor v, Tensor r_k, "
        "Tensor weight, Tensor bias, Tensor g, "
        "Tensor(a!) out"
        ") -> ()",
        forward);
}
