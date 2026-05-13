#include <torch/extension.h>
#include "ATen/ATen.h"

typedef at::BFloat16 bf16;

void cuda_forward_seqv2(int B, int T, int C, int H, float *state,
                        bf16 *r, bf16 *w, bf16 *k, bf16 *v, bf16 *a, bf16 *b,
                        bf16 *y);

void forward(int64_t B, int64_t T, int64_t C, int64_t H,
             torch::Tensor &state,
             torch::Tensor &r, torch::Tensor &w, torch::Tensor &k,
             torch::Tensor &v, torch::Tensor &a, torch::Tensor &b,
             torch::Tensor &y) {
    cuda_forward_seqv2(B, T, C, H,
                       state.data_ptr<float>(),
                       r.data_ptr<bf16>(), w.data_ptr<bf16>(), k.data_ptr<bf16>(),
                       v.data_ptr<bf16>(), a.data_ptr<bf16>(), b.data_ptr<bf16>(),
                       y.data_ptr<bf16>());
}

// Schema string is REQUIRED — see wkv7s_op.cpp for the why.
TORCH_LIBRARY(wkv7s_seqv2, m) {
    m.def(
        "forward("
        "int B, int T, int C, int H, "
        "Tensor(a!) state, "
        "Tensor r, Tensor w, Tensor k, Tensor v, Tensor a, Tensor b, "
        "Tensor(b!) y"
        ") -> ()",
        forward);
}
