// Fused CMix-mix inference kernel — bf16 port of Albatross
// faster3a_2605/cuda/rwkv7_fast_ops_fp16.cu :: cmix_mix_kernel
// (~lines 441-471).
//
// Replaces the 3-step PyTorch chain at the start of _cmix_seq:
//   shifted = torch.cat((x_prev.unsqueeze(0), x[:-1, :]), dim=0)   # cat
//   xx = shifted - x                                                 # sub
//   k = x + xx * ffn.x_k.view(-1)                                    # mix (broadcast)
// with a single packed-bf162 kernel that:
//   (a) reads x[t,c] and prev (= shift_state[c] if t==0, else x[t-1,c]),
//   (b) writes out[t,c] = cur + (prev - cur) * x_k[c],
//   (c) on t == T-1, updates shift_state[c] = x[T-1,c] in place
//       (saves the caller's separate ``x_prev.copy_(x[-1,:])`` step).
//
// Inference-only, B==1, bf16. Saves 3 kernel launches per layer
// + the (T,C) intermediate `xx` and `shifted` HBM round-trips.

#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

typedef at::BFloat16 bf16;

namespace cmix_mix_infer_detail {

__device__ inline __nv_bfloat162 load_b2(const __nv_bfloat16* ptr) {
    return *reinterpret_cast<const __nv_bfloat162*>(ptr);
}

__device__ inline void store_b2(__nv_bfloat16* ptr, float x0, float x1) {
    *reinterpret_cast<__nv_bfloat162*>(ptr) =
        __halves2bfloat162(__float2bfloat16_rn(x0), __float2bfloat16_rn(x1));
}

__device__ inline void store_b2_pass(__nv_bfloat16* ptr, __nv_bfloat162 v) {
    *reinterpret_cast<__nv_bfloat162*>(ptr) = v;
}

}  // namespace cmix_mix_infer_detail

__global__ void __launch_bounds__(128, 8) cmix_mix_infer_kernel(
    const int T, const int C,
    const __nv_bfloat16 * __restrict__ x,        // (T, C)
    __nv_bfloat16 * __restrict__ shift_state,    // (C,) — mutated in place on last token
    const __nv_bfloat16 * __restrict__ x_k,      // (C,)
    __nv_bfloat16 * __restrict__ out,            // (T, C)
    const int64_t total_pairs)                   // T * (C/2)
{
    using namespace cmix_mix_infer_detail;

    const int64_t pair_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) return;

    const int c_pairs = C >> 1;
    const int t = static_cast<int>(pair_idx / c_pairs);
    const int c = static_cast<int>(pair_idx - static_cast<int64_t>(t) * c_pairs) << 1;
    const int64_t idx = static_cast<int64_t>(t) * C + c;

    const __nv_bfloat162 cur2 = load_b2(x + idx);
    const __nv_bfloat162 prev2 = (t == 0) ? load_b2(shift_state + c)
                                          : load_b2(x + idx - C);
    const float2 cur = __bfloat1622float2(cur2);
    const float2 prev = __bfloat1622float2(prev2);
    const float2 mix = __bfloat1622float2(load_b2(x_k + c));
    store_b2(out + idx,
             cur.x + (prev.x - cur.x) * mix.x,
             cur.y + (prev.y - cur.y) * mix.y);

    // Last token's threads update shift_state in place to x[T-1, c]
    // for the caller's next call. Cheap: 1 b16 write per thread.
    if (t == T - 1) {
        store_b2_pass(shift_state + c, cur2);
    }
}

void cuda_cmix_mix_infer(int T, int C,
    bf16 *x, bf16 *shift_state, bf16 *x_k, bf16 *out)
{
    constexpr int TPB = 128;
    const int64_t total_pairs = static_cast<int64_t>(T) * (C / 2);
    const int64_t blocks = (total_pairs + TPB - 1) / TPB;
    auto stream = at::cuda::getCurrentCUDAStream();
    cmix_mix_infer_kernel<<<dim3(blocks), dim3(TPB), 0, stream>>>(
        T, C,
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(shift_state),
        reinterpret_cast<const __nv_bfloat16*>(x_k),
        reinterpret_cast<__nv_bfloat16*>(out),
        total_pairs);
}
