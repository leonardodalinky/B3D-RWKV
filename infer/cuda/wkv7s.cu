// wkv7s.cu — RWKV-7 state-mode WKV kernel (baseline, no cp.async).
//
// **In-place** on ``state``: kernel reads each (i,j) entry, updates,
// and writes back to the same address.
//
// CUDA Graph capture correctness depends on TWO non-obvious things:
//   1. The kernel launch must specify PyTorch's current CUDA stream
//      via at::cuda::getCurrentCUDAStream(); a naked launch goes to
//      the CUDA default stream which is NOT captured, so the kernel
//      silently drops out of the recorded graph on replay.
//   2. The caller passes a slice of a big backing state tensor (see
//      RWKV.init_state in train/src/model.py), so PyTorch's
//      functionalization keeps the parent storage live for the whole
//      graph — the caching allocator can't reuse a per-layer slice
//      for an unrelated transient tensor mid-graph.
// Same pattern as Albatross's rwkv7_wkv_fp16_v2.cu, plus the explicit
// schema string in wkv7s_op.cpp (which Albatross omits — they get away
// with it because their bench measures timing, not correctness).

#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
#include <c10/cuda/CUDAStream.h>

typedef at::BFloat16 bf16;

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               float * __restrict__ _state,
                               const F * __restrict__ const _r,
                               const F * __restrict__ const _w,
                               const F * __restrict__ const _k,
                               const F * __restrict__ const _v,
                               const F * __restrict__ const _a,
                               const F * __restrict__ const _b,
                               F * __restrict__ const _y)
{
    const int e = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _state += h*_N_*_N_ + i*_N_;     // B==1 asserted below

    float state[_N_];
    #pragma unroll
    for (int j = 0; j < _N_; j++)
        state[j] = _state[j];

    __shared__ float r[_N_], k[_N_], w[_N_], a[_N_], b[_N_];

    for (int _t = 0; _t < T; _t++)
    {
        const int t = e*T*C + h*_N_ + i + _t * C;
        __syncthreads();
        r[i] = float(_r[t]);
        w[i] = __expf(-__expf(float(_w[t])));
        k[i] = float(_k[t]);
        a[i] = float(_a[t]);
        b[i] = float(_b[t]);
        __syncthreads();

        float sa = 0;
        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            sa += a[j] * state[j];
        }

        float vv = float(_v[t]);
        float y = 0;
        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = state[j];
            s = s * w[j] + k[j] * vv + sa * b[j];
            y += s * r[j];
        }
        _y[t] = F(y);
    }
    #pragma unroll
    for (int j = 0; j < _N_; j++)
        _state[j] = state[j];
}

void cuda_forward(int B, int T, int C, int H,
                  float *state,
                  bf16 *r, bf16 *w, bf16 *k, bf16 *v, bf16 *a, bf16 *b,
                  bf16 *y)
{
    assert(H*_N_ == C);
    assert(B == 1); // only for B=1
    // CRITICAL: must launch on PyTorch's current CUDA stream, not the
    // default stream. CUDA Graph capture only records ops on the capture
    // stream; kernels launched on the default stream are silently dropped
    // from the captured graph, which manifests as "state mutation visible
    // during capture but not during replay". (Albatross does this; we
    // didn't until we hit the bug.)
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel_forward<<<dim3(B * H), dim3(_N_), 0, stream>>>(
        B, T, C, H, state, r, w, k, v, a, b, y);
}
