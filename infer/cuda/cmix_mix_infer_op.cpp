#include <torch/extension.h>
#include "ATen/ATen.h"

typedef at::BFloat16 bf16;

void cuda_cmix_mix_infer(int T, int C,
    bf16 *x, bf16 *shift_state, bf16 *x_k, bf16 *out);

void forward(int64_t T, int64_t C,
             torch::Tensor &x,
             torch::Tensor &shift_state,
             torch::Tensor &x_k,
             torch::Tensor &out) {
    cuda_cmix_mix_infer(T, C,
        x.data_ptr<bf16>(),
        shift_state.data_ptr<bf16>(),
        x_k.data_ptr<bf16>(),
        out.data_ptr<bf16>());
}

// shift_state and out are both mutated; x and x_k are read-only.
TORCH_LIBRARY(cmix_mix_infer, m) {
    m.def(
        "forward("
        "int T, int C, "
        "Tensor x, Tensor(a!) shift_state, Tensor x_k, Tensor(b!) out"
        ") -> ()",
        forward);
}
