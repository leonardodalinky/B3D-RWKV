// Fused TMix-mix6 inference kernel: replaces the 8 PyTorch ops
//
//   shifted = torch.cat((x_prev.unsqueeze(0), x[:-1, :]), dim=0)
//   xx = shifted - x
//   xr = x + xx * x_r;  xw = x + xx * x_w;  ...  xg = x + xx * x_g
//
// with a single kernel that produces all 6 (xr, xw, xk, xv, xa, xg)
// in one HBM read pass of x. Saves 8 kernel launches/layer * 32 layers
// = ~256 launches/step, plus eliminates the (T,C) intermediate `xx`
// and `shifted` writes (was ~2 * T * C HBM writes wasted per layer).
//
// Inference-only (no autograd, no train coupling). Each thread owns
// one (t, c) output position; reads x[t,c] + neighbor (x_prev[c] if
// t==0 else x[t-1,c]) + 6 mix scalars from constant LUT in registers.
//
// Launch on PyTorch's current CUDA stream — required for CUDA Graph
// capture (see wkv7s.cu for why).

#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

typedef at::BFloat16 bf16;

__global__ void __launch_bounds__(256, 4) tmix_mix6_infer_kernel(
    const int T, const int C,
    const __nv_bfloat16 * __restrict__ x,         // (T, C)
    const __nv_bfloat16 * __restrict__ x_prev,    // (C,)
    const __nv_bfloat16 * __restrict__ x_r,       // (C,)
    const __nv_bfloat16 * __restrict__ x_w,
    const __nv_bfloat16 * __restrict__ x_k,
    const __nv_bfloat16 * __restrict__ x_v,
    const __nv_bfloat16 * __restrict__ x_a,
    const __nv_bfloat16 * __restrict__ x_g,
    __nv_bfloat16 * __restrict__ xr,              // (T, C)
    __nv_bfloat16 * __restrict__ xw,
    __nv_bfloat16 * __restrict__ xk,
    __nv_bfloat16 * __restrict__ xv,
    __nv_bfloat16 * __restrict__ xa,
    __nv_bfloat16 * __restrict__ xg)
{
    // Grid: (T, ceil(C / blockDim.x)).
    const int t = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= C) return;

    const int idx = t * C + c;
    const float my_x = __bfloat162float(x[idx]);
    const float prev = (t == 0)
        ? __bfloat162float(x_prev[c])
        : __bfloat162float(x[idx - C]);  // x[t-1, c]
    const float xx = prev - my_x;

    // Per-channel mix scalars (loaded once each).
    const float mr = __bfloat162float(x_r[c]);
    const float mw = __bfloat162float(x_w[c]);
    const float mk = __bfloat162float(x_k[c]);
    const float mv = __bfloat162float(x_v[c]);
    const float ma = __bfloat162float(x_a[c]);
    const float mg = __bfloat162float(x_g[c]);

    // Same xx, same my_x, just different mix scalar per output.
    xr[idx] = __float2bfloat16_rn(my_x + xx * mr);
    xw[idx] = __float2bfloat16_rn(my_x + xx * mw);
    xk[idx] = __float2bfloat16_rn(my_x + xx * mk);
    xv[idx] = __float2bfloat16_rn(my_x + xx * mv);
    xa[idx] = __float2bfloat16_rn(my_x + xx * ma);
    xg[idx] = __float2bfloat16_rn(my_x + xx * mg);
}

void cuda_tmix_mix6_infer(
    int T, int C,
    bf16 *x, bf16 *x_prev,
    bf16 *x_r, bf16 *x_w, bf16 *x_k, bf16 *x_v, bf16 *x_a, bf16 *x_g,
    bf16 *xr, bf16 *xw, bf16 *xk, bf16 *xv, bf16 *xa, bf16 *xg)
{
    constexpr int TPB = 256;
    const int blocks_c = (C + TPB - 1) / TPB;
    dim3 grid(T, blocks_c);
    dim3 block(TPB);
    auto stream = at::cuda::getCurrentCUDAStream();
    tmix_mix6_infer_kernel<<<grid, block, 0, stream>>>(
        T, C,
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(x_prev),
        reinterpret_cast<const __nv_bfloat16*>(x_r),
        reinterpret_cast<const __nv_bfloat16*>(x_w),
        reinterpret_cast<const __nv_bfloat16*>(x_k),
        reinterpret_cast<const __nv_bfloat16*>(x_v),
        reinterpret_cast<const __nv_bfloat16*>(x_a),
        reinterpret_cast<const __nv_bfloat16*>(x_g),
        reinterpret_cast<__nv_bfloat16*>(xr),
        reinterpret_cast<__nv_bfloat16*>(xw),
        reinterpret_cast<__nv_bfloat16*>(xk),
        reinterpret_cast<__nv_bfloat16*>(xv),
        reinterpret_cast<__nv_bfloat16*>(xa),
        reinterpret_cast<__nv_bfloat16*>(xg));
}
