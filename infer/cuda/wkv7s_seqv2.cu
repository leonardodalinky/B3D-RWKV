// DiffuRWKV WKV-7 stateful inference kernel (seq_v2) — bf16 port of
// BlinkDL/Albatross wkv_fp16_seq_v2_kernel
// (faster3a_2605/cuda/rwkv7_wkv_fp16_v2.cu, ~lines 312-407).
//
// Same ABI as wkv7s.cu: B=1, fp32 (H, N, N) state mutated in place,
// bf16 (T, C) r/w/k/v/a/b, bf16 (T, C) y output. State is read+written
// at the same address (kernel internally caches it in registers across
// the T-loop). See wkv7s.cu for the full CUDA-Graph-correctness notes
// (current-stream launch + big-storage state views + schema string).
// Caller pre-bakes w_pre = -softplus(-(w0 + lora)) - 0.5; the kernel
// finishes the transform inline as `w_delta = exp(-exp(w_pre)) - 1`
// (delta form, matching Albatross's FMA chain semantics — the +s
// residual in the chain recovers the s*w_eff multiplication).
//
// Speed vs wkv7s.cu at (B=1, T>=8) comes from:
//  (a) cp.async double-buffered ping-pong prefetch of next-token
//      r/w/k/a/b overlapping current-token compute,
//  (b) packed-bf16 __hfma2 on __nv_bfloat162, cutting FLOP issue count
//      ~2x for the inner state-update loop.
//
// Numerics: state lives in registers as 32 packed bf162 pairs across
// the entire T-loop. bf16's 7-bit mantissa drifts ~1% relative over
// T=64 in realistic-scale inputs (post-RWKV7-norm distributions where
// state magnitudes stay O(0.1)). On synthetic IID random tensors with
// std=0.5, baseline wkv7s itself enters unbounded-feedback runaway and
// either kernel produces explosively large state; the absolute Δ between
// kernels then looks huge but is dominated by ~1% relative drift on the
// blown-up magnitudes. For trained-model inference state stays bounded
// and the precision drift is invisible.
//
// Requires sm_80+ (cp.async + bf16 __hfma2). Built per-arch by model.py
// with --gencode=arch=compute_80,code=sm_80 and compute_90/sm_90.

#undef __CUDA_NO_BFLOAT16_OPERATORS__
#undef __CUDA_NO_BFLOAT162_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__

#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

typedef at::BFloat16 bf16;

namespace wkv7s_seqv2_detail {

constexpr int HALF2_N = _N_ / 2;

template <int Bytes>
__device__ __forceinline__ void cp_async(void* smem, const void* global, bool pred) {
    static_assert(Bytes == 16 || Bytes == 8 || Bytes == 4);
    int bytes = pred ? Bytes : 0;
    unsigned addr = __cvta_generic_to_shared(smem);
    if constexpr (Bytes == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;"
                     ::"r"(addr), "l"(global), "n"(Bytes), "r"(bytes));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;"
                     ::"r"(addr), "l"(global), "n"(Bytes), "r"(bytes));
    }
}

__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int NWait>
__device__ __forceinline__ void cp_wait() {
    if constexpr (NWait == 0) {
        asm volatile("cp.async.wait_all;\n" ::);
    } else {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(NWait));
    }
}

// Issue the 5 cp.async loads for one token's r/w/k/a/b. Each load is 4
// bytes (1 bf162 = 2 bf16). Threads 0-31 stream w/r/b, threads 32-63
// stream a/k (with b's pred=false, suppressed on those threads). Total
// bytes per token: 5 * 32 * 4 = 640 B (covers 5 vectors of N=64 bf16).
__device__ __forceinline__ void prefetch_token(
    int tid, int lane, int token,
    __nv_bfloat162* r, __nv_bfloat162* w,
    __nv_bfloat162* k, __nv_bfloat162* a, __nv_bfloat162* b,
    const __nv_bfloat16* r_ptr, const __nv_bfloat16* w_ptr,
    const __nv_bfloat16* k_ptr, const __nv_bfloat16* a_ptr,
    const __nv_bfloat16* b_ptr)
{
    cp_async<4>((tid < 32 ? w : a) + lane,
                (const __nv_bfloat162*)(tid < 32 ? w_ptr + token : a_ptr + token) + lane,
                true);
    cp_commit();
    cp_async<4>((tid < 32 ? r : k) + lane,
                (const __nv_bfloat162*)(tid < 32 ? r_ptr + token : k_ptr + token) + lane,
                true);
    cp_async<4>(b + lane,
                (const __nv_bfloat162*)(b_ptr + token) + lane,
                tid < 32);
    cp_commit();
}

}  // namespace wkv7s_seqv2_detail

__global__ void __launch_bounds__(_N_, 2) kernel_forward_seqv2(
    const int T, const int C, const int H,
    float * __restrict__ _state,
    const __nv_bfloat16 * __restrict__ const r_ptr,
    const __nv_bfloat16 * __restrict__ const w_ptr,
    const __nv_bfloat16 * __restrict__ const k_ptr,
    const __nv_bfloat16 * __restrict__ const v_ptr,
    const __nv_bfloat16 * __restrict__ const a_ptr,
    const __nv_bfloat16 * __restrict__ const b_ptr,
    __nv_bfloat16 * __restrict__ const y_ptr)
{
    using namespace wkv7s_seqv2_detail;

    // B==1: blockIdx.x indexes head directly.
    const int h = blockIdx.x;
    const int i = threadIdx.x;
    const int lane = i & 31;

    // Thread i owns row i of state. Load fp32 row from _state, pack
    // into bf162 register pairs. State lives in registers throughout
    // the T-loop, then we write back to the same fp32 address at the
    // end. The caller passes a view of a big backing state tensor —
    // PyTorch sees the slice as a view of the parent storage, so the
    // parent stays live across the full captured graph and the
    // caching allocator never reuses this address for other transient
    // tensors (the pattern Albatross's wkv kernels rely on).
    _state += h * _N_ * _N_ + i * _N_;
    __nv_bfloat162 state[HALF2_N];
    #pragma unroll
    for (int j = 0; j < HALF2_N; j++) {
        state[j] = __halves2bfloat162(
            __float2bfloat16_rn(_state[j*2]),
            __float2bfloat16_rn(_state[j*2 + 1]));
    }

    // Ping-pong shared buffers for next-token prefetch.
    __shared__ __align__(128) __nv_bfloat162
        r[2][HALF2_N], w[2][HALF2_N], k[2][HALF2_N],
        a[2][HALF2_N], bvec[2][HALF2_N];

    // Pre-issue token 0 before the main loop.
    int token = h * _N_;
    prefetch_token(i, lane, token, r[0], w[0], k[0], a[0], bvec[0],
                   r_ptr, w_ptr, k_ptr, a_ptr, b_ptr);

    for (int tt = 0; tt < T; tt++) {
        const int cur = tt & 1;
        cp_wait<0>();
        __syncthreads();

        // sa = a · state  (per-thread reduction over packed pairs).
        __nv_bfloat162 sa2 = __float2bfloat162_rn(0.0f);
        #pragma unroll
        for (int j = 0; j < HALF2_N; j++) {
            sa2 = __hfma2(a[cur][j], state[j], sa2);
        }
        __nv_bfloat16 sa = __hadd(sa2.x, sa2.y);
        sa2 = __halves2bfloat162(sa, sa);

        // Apply w-transform in place: store w_delta = exp(-exp(w_pre))-1.
        // The FMA chain below produces s*w + k*v + sa*b + s, which with
        // w = w_eff - 1 algebraically recovers s*w_eff + k*v + sa*b
        // (the math wkv7s.cu computes).
        ((__nv_bfloat16*)w[cur])[i] = __float2bfloat16_rn(
            __expf(-__expf(__bfloat162float(((__nv_bfloat16*)w[cur])[i]))) - 1.0f);
        __syncthreads();

        // Kick off NEXT token's prefetch while we crunch the current.
        // Writes the OTHER ping-pong slot (cur^1); no conflict with the
        // FMA loop below that reads slot `cur`.
        if (tt + 1 < T) {
            const int next_token = token + C;
            prefetch_token(i, lane, next_token,
                           r[cur ^ 1], w[cur ^ 1], k[cur ^ 1],
                           a[cur ^ 1], bvec[cur ^ 1],
                           r_ptr, w_ptr, k_ptr, a_ptr, b_ptr);
        }

        // v is per-thread; direct ldg + broadcast.
        __nv_bfloat16 vv = v_ptr[token + i];
        __nv_bfloat162 vv2 = __halves2bfloat162(vv, vv);

        // State update + y accumulation. 32 packed __hfma2 = 64 FMA-equiv
        // per t-step, ~2x throughput vs scalar bf16.
        __nv_bfloat162 y2 = __float2bfloat162_rn(0.0f);
        #pragma unroll
        for (int j = 0; j < HALF2_N; j++) {
            __nv_bfloat162 s = state[j];
            s = __hfma2(s, w[cur][j],
                        __hfma2(k[cur][j], vv2,
                                __hfma2(sa2, bvec[cur][j], s)));
            state[j] = s;
            y2 = __hfma2(s, r[cur][j], y2);
        }
        y_ptr[token + i] = __hadd(y2.x, y2.y);
        token += C;
    }

    // Unpack state back to fp32 and write back in place.
    #pragma unroll
    for (int j = 0; j < HALF2_N; j++) {
        float2 s = __bfloat1622float2(state[j]);
        _state[j*2]     = s.x;
        _state[j*2 + 1] = s.y;
    }
}

void cuda_forward_seqv2(int B, int T, int C, int H, float *state,
    bf16 *r, bf16 *w, bf16 *k, bf16 *v, bf16 *a, bf16 *b, bf16 *y)
{
    assert(H * _N_ == C);
    assert(B == 1);
    // Launch on PyTorch's current CUDA stream — required for CUDA Graph
    // capture to pick up this kernel. See wkv7s.cu for the why.
    auto stream = at::cuda::getCurrentCUDAStream();
    kernel_forward_seqv2<<<dim3(H), dim3(_N_), 0, stream>>>(
        T, C, H, state,
        reinterpret_cast<const __nv_bfloat16*>(r),
        reinterpret_cast<const __nv_bfloat16*>(w),
        reinterpret_cast<const __nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v),
        reinterpret_cast<const __nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(b),
        reinterpret_cast<__nv_bfloat16*>(y));
}
