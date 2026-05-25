// Fused post-WKV inference kernel — bf16 port of Albatross
// faster3a_2605/cuda/rwkv7_fast_ops_fp16.cu :: tmix_lnx_rkvres_xg_kernel
// (~lines 222-285).
//
// Replaces the 3-step PyTorch chain that runs after the WKV custom op:
//   out = F.group_norm(out, num_groups=H, weight=ln_x.weight, bias=ln_x.bias, eps=64e-5)
//   out = out + ((r * k * r_k).view(T,H,N).sum(-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
//   out = out * g
// with a single per-(t,h) kernel that does:
//   mean_th = sum_n(out[t, hN+n]) / N
//   var_th  = sum_n((out[t, hN+n] - mean_th)**2) / N
//   norm    = (out[t, hN+n] - mean_th) * rsqrt(var_th + eps) * ln_x.weight[hN+n] + ln_x.bias[hN+n]
//   rkrk_th = sum_n(r[t, hN+n] * k[t, hN+n] * r_k[hN+n])     // scalar per (t,h)
//   final[t, hN+n] = (norm[t, hN+n] + rkrk_th * v[t, hN+n]) * g[t, hN+n]
//
// Saves 3-5 kernel launches per layer × 32 layers per denoise step.
//
// Each block owns one (t, h). HEAD_SIZE=64 = blockDim.x = 2 warps. The
// 3 reductions (mean, var, rkrk) each use a warp_sum shfl + 2-slot
// shared-memory cross-warp combine. ~5 sync points per (t,h).
//
// Launch on PyTorch's current CUDA stream (CUDA Graph capture).

#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

typedef at::BFloat16 bf16;

namespace tmix_post_infer_detail {

__device__ inline float load_b1(const __nv_bfloat16* ptr) {
    return __bfloat162float(*ptr);
}

__device__ inline void store_b1(__nv_bfloat16* ptr, float value) {
    *ptr = __float2bfloat16_rn(value);
}

__device__ inline float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

}  // namespace tmix_post_infer_detail

__global__ void __launch_bounds__(64, 8) tmix_post_infer_kernel(
    const int H,
    const __nv_bfloat16 * __restrict__ x,        // (T, H*N) — WKV out
    const __nv_bfloat16 * __restrict__ r,        // (T, H*N)
    const __nv_bfloat16 * __restrict__ k,        // (T, H*N)
    const __nv_bfloat16 * __restrict__ v,        // (T, H*N)
    const __nv_bfloat16 * __restrict__ r_k,      // (H*N)
    const __nv_bfloat16 * __restrict__ weight,   // (H*N) — ln_x.weight
    const __nv_bfloat16 * __restrict__ bias,     // (H*N) — ln_x.bias
    const __nv_bfloat16 * __restrict__ g,        // (T, H*N)
    __nv_bfloat16 * __restrict__ out)            // (T, H*N)
{
    using namespace tmix_post_infer_detail;

    constexpr int N = _N_;            // HEAD_SIZE, set by build flag
    static_assert(N == 64, "kernel assumes N=64 (2 warps/block)");

    __shared__ float partial[2];
    const int th = blockIdx.x;        // (t * H + h) — flat (t,h)
    const int lane = threadIdx.x;     // [0, N)
    const int warp = lane >> 5;
    const int warp_lane = lane & 31;
    const int h = th % H;
    const int64_t base = static_cast<int64_t>(th) * N;     // (t,h)-row start in (T, H*N)
    const int64_t cbase = static_cast<int64_t>(h) * N;     // h-row start in (H*N)
    const int64_t idx = base + lane;
    const int64_t c = cbase + lane;

    // Load this thread's x value once; use 3 times (mean, var, final).
    const float xv = load_b1(x + idx);

    // ---- mean over N elements per (t,h) ----
    float s_mean = warp_sum(xv);
    if (warp_lane == 0) partial[warp] = s_mean;
    __syncthreads();
    const float mean = (partial[0] + partial[1]) * (1.0f / static_cast<float>(N));
    __syncthreads();

    // ---- var over N elements ----
    const float d = xv - mean;
    float s_var = warp_sum(d * d);
    if (warp_lane == 0) partial[warp] = s_var;
    __syncthreads();
    const float var = (partial[0] + partial[1]) * (1.0f / static_cast<float>(N));
    const float rstd = rsqrtf(var + 64.0e-5f);  // eps matches F.group_norm(eps=64e-5)
    __syncthreads();

    // ---- rkrk = sum_n(r * k * r_k) per (t,h) ----
    const float rv = load_b1(r + idx);
    const float kv = load_b1(k + idx);
    const float vv = load_b1(v + idx);
    float dot = rv * kv * load_b1(r_k + c);
    dot = warp_sum(dot);
    if (warp_lane == 0) partial[warp] = dot;
    __syncthreads();
    const float rkv = partial[0] + partial[1];
    __syncthreads();

    // ---- final: (norm * weight + bias + rkv * v) * g ----
    const float y = (d * rstd * load_b1(weight + c) + load_b1(bias + c) + rkv * vv)
                    * load_b1(g + idx);
    store_b1(out + idx, y);
}

void cuda_tmix_post_infer(
    int T, int H,
    bf16 *x, bf16 *r, bf16 *k, bf16 *v, bf16 *r_k,
    bf16 *weight, bf16 *bias, bf16 *g, bf16 *out)
{
    const int64_t bth_size = static_cast<int64_t>(T) * H;   // grid = T*H blocks
    auto stream = at::cuda::getCurrentCUDAStream();
    tmix_post_infer_kernel<<<dim3(bth_size), dim3(_N_), 0, stream>>>(
        H,
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(r),
        reinterpret_cast<const __nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v),
        reinterpret_cast<const __nv_bfloat16*>(r_k),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<const __nv_bfloat16*>(bias),
        reinterpret_cast<const __nv_bfloat16*>(g),
        reinterpret_cast<__nv_bfloat16*>(out));
}
