########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import gc
import importlib
import math
import os
import sys

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from torch.nn import functional as F

if importlib.util.find_spec("deepspeed"):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

try:
    print("RWKV_MY_TESTING", os.environ["RWKV_MY_TESTING"])
except:
    os.environ["RWKV_MY_TESTING"] = ""


def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method

# os.environ["RWKV_HEAD_L2WRAP_CE_CHUNK"] = '4096' # saves 80% VRAM, slower
# os.environ["RWKV_HEAD_L2WRAP_CE_CHUNK"] = '65536' # saves 70% VRAM, sometimes faster than '4096'
os.environ["RWKV_HEAD_L2WRAP_CE_CHUNK"] = "0"  # fast, takes more VRAM

########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE"])

# When `RWKV_INFERENCE_ONLY=1`, skip compiling all training-only CUDA
# kernels (autograd-aware fused TMix/CMix, clampw, L2-wrap CE). This
# cuts module-import time from minutes (first cold build of ~7 train
# kernels) down to seconds, since inference paths (RWKV.forward_fast)
# only need the small inference-only kernels under infer/cuda/.
#
# Set this in inference entry points (infer/diffusion_sample.py's
# build_model) before importing src.model. Leave unset for training
# scripts so all train kernels load normally.
_INFERENCE_ONLY = os.environ.get("RWKV_INFERENCE_ONLY", "0") == "1"


# Compute-capability detection used to pick arch-specific kernel
# variants (e.g. clampw has a v3_for_h100 fast path that uses
# constructs only well-supported from sm_80+).
def _sm_at_least(major: int, minor: int = 0) -> bool:
    if not torch.cuda.is_available():
        return False
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    return (cc_major, cc_minor) >= (major, minor)


# ---- Conditional-compile helper ------------------------------------
# Each training-only kernel call site uses ``conditional_load(cond, ...)``
# with ``cond = not _INFERENCE_ONLY``: the build is skipped (returns
# None) in inference mode. Inference kernels stay on plain ``load(...)``.
# Two ancillary stubs are needed in inference mode so the training-side
# class definitions that follow each skipped build don't blow up:
#   * ``torch.library.register_autograd`` → no-op, since no op was
#     registered to attach an autograd to.
#   * ``torch.jit.script`` → identity, since the decorated function's
#     body references the missing ``torch.ops.X.forward`` symbol at
#     decoration time.
# Both stubs are local to this import scope; nothing else in the
# codebase relies on these symbols.


def conditional_load(condition, name, *args, **kwargs):
    """Build the CUDA extension only when ``condition`` is True;
    otherwise return ``None`` and skip the build. Used to gate
    training-only kernel compiles when running inference."""
    if not condition:
        return None
    return load(name, *args, **kwargs)


if _INFERENCE_ONLY:

    def _stub_register_autograd(*args, **kwargs):
        return None

    torch.library.register_autograd = _stub_register_autograd

    def _stub_jit_script(fn=None, *args, **kwargs):
        # Used as both a decorator (`@torch.jit.script`) and as a
        # function (`torch.jit.script(fn)`). Identity in both cases.
        if fn is None:
            return lambda f: f
        return fn

    torch.jit.script = _stub_jit_script

if "x070" in os.environ["RWKV_MY_TESTING"]:
    CHUNK_LEN = 16
    assert HEAD_SIZE == 64  # can change 64 to your HEAD_SIZE

    # check https://github.com/BlinkDL/RWKV-CUDA/blob/main/rwkv7_fast_fused/rwkv7_cuda_benchmark.py
    #
    # use rwkv7_clampw_v3.cpp and rwkv7_clampw_v3_for_h100.cu for 20% faster fwd & bwd kernel on H100s

    # ---- Training-only: WKV-7 fwd+bwd fused kernel (clampw) ----
    # The v3_for_h100 variant is faster on sm_90 but should compile fine
    # back to sm_80; the baseline rwkv7_clampw.cu is the conservative
    # fallback if compute capability is older. In inference mode the
    # `load` wrapper below skips the compile; the autograd Function and
    # wrapper still get defined (they're never called).
    flags = [
        "-res-usage",
        f"-D_N_={HEAD_SIZE}",
        f"-D_CHUNK_LEN_={CHUNK_LEN}",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
    ]
    if _sm_at_least(8, 0):
        _clampw_sources = ["cuda/rwkv7_clampw_v3_for_h100.cu", "cuda/rwkv7_clampw_v3.cpp"]
    else:
        _clampw_sources = ["cuda/rwkv7_clampw.cu", "cuda/rwkv7_clampw.cpp"]
    conditional_load(
        not _INFERENCE_ONLY,
        name="rwkv7_clampw",
        sources=_clampw_sources,
        is_python_module=False,
        verbose=True,
        extra_cuda_cflags=flags,
    )

    class RWKV7_CLAMPW_CUDA_OP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, r, w, k, v, a, b):
            B, T, H, N = r.shape
            assert (
                T % CHUNK_LEN == 0
            )  # if T%CHUNK_LEN != 0: pad your input to T%CHUNK_LEN == 0, or change CHUNK_LEN (will be slower)
            assert all(i.dtype == torch.bfloat16 for i in [r, w, k, v, a, b])
            assert all(i.is_contiguous() for i in [r, w, k, v, a, b])
            y = torch.empty_like(v)
            s = torch.empty(B, H, T // CHUNK_LEN, N, N, dtype=torch.float32, device=w.device)
            sa = torch.empty(B, T, H, N, dtype=torch.float32, device=w.device)
            torch.ops.rwkv7_clampw.forward(r, w, k, v, a, b, y, s, sa)
            ctx.save_for_backward(r, w, k, v, a, b, s, sa)
            return y

        @staticmethod
        def backward(ctx, dy):
            assert all(i.dtype == torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            r, w, k, v, a, b, s, sa = ctx.saved_tensors
            dr, dw, dk, dv, da, db = [torch.empty_like(x) for x in [r, w, k, v, a, b]]
            torch.ops.rwkv7_clampw.backward(r, w, k, v, a, b, dy, s, sa, dr, dw, dk, dv, da, db)
            return dr, dw, dk, dv, da, db

    def RWKV7_CLAMPW_CUDA(r, w, k, v, a, b):
        B, T, HN = r.shape
        r, w, k, v, a, b = [
            i.view(B, T, HN // 64, 64) for i in [r, w, k, v, a, b]
        ]  # can change 64 to your HEAD_SIZE. have to hard-code the number here, or pytorch will complain
        return RWKV7_CLAMPW_CUDA_OP.apply(r, w, k, v, a, b).view(B, T, HN)

    # Stateful (RNN-mode) WKV kernel from upstream rwkv_v7_demo_fast.py — used by
    # RWKV.forward_fast for state-aware inference (no autograd, eval only).
    # Source lives under DiffuRWKV/infer/cuda/ (inference-only kernels are
    # isolated there); resolved via absolute path so build_model's
    # chdir(train/) for the other training-only kernels still works.
    _INFER_CUDA_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "infer", "cuda")
    )
    load(
        name="wkv7s",
        sources=[
            os.path.join(_INFER_CUDA_DIR, "wkv7s_op.cpp"),
            os.path.join(_INFER_CUDA_DIR, "wkv7s.cu"),
        ],
        is_python_module=False,
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-D_N_={HEAD_SIZE}",
        ],
    )

    def RWKV7S_OP(state, r, w, k, v, a, b):
        """Stateful WKV-7 op. Inputs are (T, C); state is (H, N, N) fp32.

        In-place on ``state``. Two non-obvious things are required for
        the op to behave correctly inside ``torch.cuda.graph()``:

          1. The kernel (see wkv7s.cu) launches on PyTorch's current
             stream via ``at::cuda::getCurrentCUDAStream()``. A naked
             ``<<<grid, block>>>`` launch goes to the CUDA default
             stream, which is NOT captured; the kernel runs during
             capture but is silently dropped from the recorded graph,
             producing the classic "eager matches, replay misses the
             state mutation" symptom.

          2. The schema string in wkv7s_op.cpp marks ``state`` and
             ``y`` as ``Tensor(a!)`` / ``Tensor(b!)`` (mutable).
             Without the schema, PyTorch infers a side-effect-free op
             and the dispatcher elides the call. Required, not
             optional.
        """
        T, C = r.shape
        H = C // HEAD_SIZE
        y = torch.empty(
            (T, C),
            device=r.device,
            dtype=r.dtype,
            requires_grad=False,
            memory_format=torch.contiguous_format,
        )
        torch.ops.wkv7s.forward(1, T, C, H, state, r, w, k, v, a, b, y)
        return y

    # seq_v2 variant — same ABI, ported from BlinkDL/Albatross
    # wkv_fp16_seq_v2_kernel. Uses cp.async ping-pong prefetch + packed
    # bf16 __hfma2 for ~1.4-1.8x speedup vs wkv7s at (B=1, T>=8). Falls
    # back to wkv7s when T < 8 (decode-style; ping-pong setup is wasted).
    # Requires sm_80+; build flags pin both 8.0 and 9.0.
    load(
        name="wkv7s_seqv2",
        sources=[
            os.path.join(_INFER_CUDA_DIR, "wkv7s_seqv2_op.cpp"),
            os.path.join(_INFER_CUDA_DIR, "wkv7s_seqv2.cu"),
        ],
        is_python_module=False,
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-D_N_={HEAD_SIZE}",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_90,code=sm_90",
        ],
    )

    def RWKV7S_OP_SEQV2(state, r, w, k, v, a, b):
        """Same contract as RWKV7S_OP, dispatched to the seq_v2 kernel.
        See RWKV7S_OP for the CUDA-Graph-correctness requirements
        (current-stream launch + schema string)."""
        T, C = r.shape
        H = C // HEAD_SIZE
        y = torch.empty(
            (T, C),
            device=r.device,
            dtype=r.dtype,
            requires_grad=False,
            memory_format=torch.contiguous_format,
        )
        torch.ops.wkv7s_seqv2.forward(1, T, C, H, state, r, w, k, v, a, b, y)
        return y

    # Fused TMix-mix6 kernel for inference. Replaces 8 PyTorch ops (cat
    # + sub + 6 elementwise) in _tmix_seq's pre-WKV block with a single
    # kernel launch. Inference-only (no autograd), bf16. ~0.5 ms /step
    # speedup at 7.2B / T=64 by removing the intermediate `xx` and
    # `shifted` HBM round-trips.
    load(
        name="tmix_mix6_infer",
        sources=[
            os.path.join(_INFER_CUDA_DIR, "tmix_mix6_infer_op.cpp"),
            os.path.join(_INFER_CUDA_DIR, "tmix_mix6_infer.cu"),
        ],
        is_python_module=False,
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
        ],
    )

    def TMIX_MIX6_INFER(x, x_prev, x_r, x_w, x_k, x_v, x_a, x_g):
        """Fused mix6 op. ``x`` is (T, C) bf16; ``x_prev`` is (C,) bf16;
        ``x_*`` are (C,) bf16. Returns 6 (T, C) bf16 tensors:
        (xr, xw, xk, xv, xa, xg)."""
        T, C = x.shape
        xr = torch.empty_like(x)
        xw = torch.empty_like(x)
        xk = torch.empty_like(x)
        xv = torch.empty_like(x)
        xa = torch.empty_like(x)
        xg = torch.empty_like(x)
        torch.ops.tmix_mix6_infer.forward(
            T, C, x, x_prev, x_r, x_w, x_k, x_v, x_a, x_g, xr, xw, xk, xv, xa, xg
        )
        return xr, xw, xk, xv, xa, xg

    # Fused post-WKV inference kernel. Combines group_norm (per-head over
    # N=64), r·k·r_k reduction + v residual, and the gate (`xg`) multiply
    # — all the work between WKV's output and the final att.output linear.
    # Saves ~3-5 launches per layer × 32 layers / step. bf16, B==1.
    load(
        name="tmix_post_infer",
        sources=[
            os.path.join(_INFER_CUDA_DIR, "tmix_post_infer_op.cpp"),
            os.path.join(_INFER_CUDA_DIR, "tmix_post_infer.cu"),
        ],
        is_python_module=False,
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-D_N_={HEAD_SIZE}",
            "-gencode=arch=compute_80,code=sm_80",
            "-gencode=arch=compute_90,code=sm_90",
        ],
    )

    def TMIX_POST_INFER(out_wkv, r, k, v, r_k, ln_x_weight, ln_x_bias, g):
        """Fused post-WKV op. All inputs bf16. ``out_wkv``/r/k/v/g are
        (T, H*N); r_k/weight/bias are (H*N,). Returns the (T, H*N)
        bf16 tensor that gets fed to ``att.output`` linear."""
        T, C = out_wkv.shape
        H = C // HEAD_SIZE
        out = torch.empty_like(out_wkv)
        torch.ops.tmix_post_infer.forward(
            T, H, out_wkv, r, k, v, r_k, ln_x_weight, ln_x_bias, g, out
        )
        return out

    # Fused CMix-mix kernel. Combines (cat + sub + mix) for the cmix
    # token-shift and ALSO updates the caller's ``x_prev`` in place to
    # x[T-1, :] (saves the separate ``x_prev.copy_(x[-1, :])``).
    load(
        name="cmix_mix_infer",
        sources=[
            os.path.join(_INFER_CUDA_DIR, "cmix_mix_infer_op.cpp"),
            os.path.join(_INFER_CUDA_DIR, "cmix_mix_infer.cu"),
        ],
        is_python_module=False,
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
        ],
    )

    def CMIX_MIX_INFER(x, x_prev, x_k):
        """Fused cmix mix-1 op. ``x`` is (T, C) bf16; ``x_prev`` (C,)
        bf16 — MUTATED in place to x[T-1, :]; ``x_k`` is (C,). Returns
        the mixed (T, C) tensor."""
        T, C = x.shape
        out = torch.empty_like(x)
        torch.ops.cmix_mix_infer.forward(T, C, x, x_prev, x_k, out)
        return out


########################################################################################################

conditional_load(
    not _INFERENCE_ONLY,
    name="rwkv7_cmix_bf16_v5",
    sources=["cuda/rwkv7_cmix_bf16_v5.cpp", "cuda/rwkv7_cmix_bf16_v5.cu"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-res-usage",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
    ],
    is_python_module=False,
    verbose=True,
)


class _CmixLayerV2Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_k, key_weight, value_weight):
        out, mixed, act = torch.ops.rwkv7_cmix_bf16_v5.forward(
            x.contiguous(),
            x_k.contiguous(),
            key_weight.contiguous(),
            value_weight.contiguous(),
        )
        ctx.save_for_backward(x, x_k, key_weight, value_weight, mixed, act)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, x_k, key_weight, value_weight, mixed, act = ctx.saved_tensors
        grad_x, grad_x_k, grad_key_weight, grad_value_weight = (
            torch.ops.rwkv7_cmix_bf16_v5.backward(
                grad_out.contiguous(),
                x,
                x_k,
                key_weight,
                value_weight,
                mixed,
                act,
            )
        )
        return grad_x, grad_x_k, grad_key_weight, grad_value_weight


########################################################################################################

conditional_load(
    not _INFERENCE_ONLY,
    name="rwkv7_tmix_mix6_bf16_v5",
    sources=["cuda/rwkv7_tmix_mix6_bf16_v5.cpp", "cuda/rwkv7_tmix_mix6_bf16_v5.cu"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-res-usage",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
    ],
    is_python_module=False,
    verbose=True,
)

from typing import Tuple


def _setup_context(ctx, inputs, output):
    del output
    ctx.save_for_backward(*inputs)


def _backward(ctx, grads):
    return tuple(
        torch.ops.rwkv7_tmix_mix6_bf16_v5.backward(
            grads[0].contiguous(),
            grads[1].contiguous(),
            grads[2].contiguous(),
            grads[3].contiguous(),
            grads[4].contiguous(),
            grads[5].contiguous(),
            *ctx.saved_tensors,
        )
    )


torch.library.register_autograd(
    "rwkv7_tmix_mix6_bf16_v5::forward",
    _backward,
    setup_context=_setup_context,
)


def _forward_op(x, x_r, x_w, x_k, x_v, x_a, x_g):
    return torch.ops.rwkv7_tmix_mix6_bf16_v5.forward(
        x.contiguous(),
        x_r.contiguous(),
        x_w.contiguous(),
        x_k.contiguous(),
        x_v.contiguous(),
        x_a.contiguous(),
        x_g.contiguous(),
    )


@torch.jit.script
def _tmix_mix6_bf16_v5_jit(
    x: torch.Tensor,
    x_r: torch.Tensor,
    x_w: torch.Tensor,
    x_k: torch.Tensor,
    x_v: torch.Tensor,
    x_a: torch.Tensor,
    x_g: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outs = torch.ops.rwkv7_tmix_mix6_bf16_v5.forward(
        x.contiguous(),
        x_r.contiguous(),
        x_w.contiguous(),
        x_k.contiguous(),
        x_v.contiguous(),
        x_a.contiguous(),
        x_g.contiguous(),
    )
    return outs[0], outs[1], outs[2], outs[3], outs[4], outs[5]


if os.environ.get("RWKV_JIT_ON") == "1":

    def tmix_mix6_bf16_v5(x, x_r, x_w, x_k, x_v, x_a, x_g):
        return _tmix_mix6_bf16_v5_jit(x, x_r, x_w, x_k, x_v, x_a, x_g)

else:
    # _forward_op gets re-defined 4 more times below; capture the current
    # binding via default arg to defeat Python late binding.
    def tmix_mix6_bf16_v5(x, x_r, x_w, x_k, x_v, x_a, x_g, _op=_forward_op):
        return tuple(_op(x, x_r, x_w, x_k, x_v, x_a, x_g))


########################################################################################################

conditional_load(
    not _INFERENCE_ONLY,
    name="rwkv7_tmix_kk_pre_bf16_v5",
    sources=["cuda/rwkv7_tmix_kk_pre_bf16_v5.cpp", "cuda/rwkv7_tmix_kk_pre_bf16_v5.cu"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-res-usage",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
    ],
    is_python_module=False,
    verbose=True,
)

assert HEAD_SIZE == 64


def _setup_context(ctx, inputs, output):
    k, k_k, a, k_a, _head_size = inputs
    inv_d = output[3]
    ctx.save_for_backward(k, k_k, a, k_a, inv_d)


def _backward(ctx, grads):
    k, k_k, a, k_a, inv_d = ctx.saved_tensors
    grad_new_k = grads[0].contiguous()
    grad_neg_kk = grads[1].contiguous()
    grad_kka = grads[2].contiguous()

    return tuple(
        torch.ops.rwkv7_tmix_kk_pre_bf16_v5.backward(
            grad_new_k,
            grad_neg_kk,
            grad_kka,
            k,
            k_k,
            a,
            k_a,
            inv_d,
            64,
        )
    ) + (None,)


torch.library.register_autograd(
    "rwkv7_tmix_kk_pre_bf16_v5::forward",
    _backward,
    setup_context=_setup_context,
)


def _forward_op(k, k_k, a, k_a):
    outs = torch.ops.rwkv7_tmix_kk_pre_bf16_v5.forward(
        k.contiguous(),
        k_k.contiguous(),
        a.contiguous(),
        k_a.contiguous(),
        64,
    )
    return outs[0], outs[1], outs[2]


@torch.jit.script
def _tmix_kk_pre_bf16_v5_jit(
    k: torch.Tensor,
    k_k: torch.Tensor,
    a: torch.Tensor,
    k_a: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    outs = torch.ops.rwkv7_tmix_kk_pre_bf16_v5.forward(
        k.contiguous(),
        k_k.contiguous(),
        a.contiguous(),
        k_a.contiguous(),
        64,
    )
    return outs[0], outs[1], outs[2]


if os.environ.get("RWKV_JIT_ON") == "1":

    def tmix_kk_pre_bf16_v5(k, k_k, a, k_a):
        return _tmix_kk_pre_bf16_v5_jit(k, k_k, a, k_a)

else:

    def tmix_kk_pre_bf16_v5(k, k_k, a, k_a, _op=_forward_op):
        return tuple(_op(k, k_k, a, k_a))


########################################################################################################

conditional_load(
    not _INFERENCE_ONLY,
    name="rwkv7_tmix_lnx_rkvres_xg_bf16_v1",
    sources=[
        "cuda/rwkv7_tmix_lnx_rkvres_xg_bf16_v1.cpp",
        "cuda/rwkv7_tmix_lnx_rkvres_xg_bf16_v1.cu",
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-res-usage",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
    ],
    is_python_module=False,
    verbose=True,
)


def _setup_context(ctx, inputs, output):
    x, r, k, v, r_k, weight, bias, g = inputs
    mean = output[1]
    rstd = output[2]
    ctx.save_for_backward(x, r, k, v, r_k, weight, bias, g, mean, rstd)


def _backward(ctx, grads):
    x, r, k, v, r_k, weight, bias, g, mean, rstd = ctx.saved_tensors
    grad_xg = grads[0].contiguous()
    return tuple(
        torch.ops.rwkv7_tmix_lnx_rkvres_xg_bf16_v1.backward(
            grad_xg,
            x,
            r,
            k,
            v,
            r_k,
            weight,
            bias,
            g,
            mean,
            rstd,
        )
    )


torch.library.register_autograd(
    "rwkv7_tmix_lnx_rkvres_xg_bf16_v1::forward",
    _backward,
    setup_context=_setup_context,
)


def _forward_op(x, r, k, v, r_k, weight, bias, g):
    outs = torch.ops.rwkv7_tmix_lnx_rkvres_xg_bf16_v1.forward(
        x.contiguous(),
        r.contiguous(),
        k.contiguous(),
        v.contiguous(),
        r_k.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
        g.contiguous(),
    )
    return outs[0]


@torch.jit.script
def _tmix_lnx_rkvres_xg_bf16_v1_jit(
    x: torch.Tensor,
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r_k: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    outs = torch.ops.rwkv7_tmix_lnx_rkvres_xg_bf16_v1.forward(
        x.contiguous(),
        r.contiguous(),
        k.contiguous(),
        v.contiguous(),
        r_k.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
        g.contiguous(),
    )
    return outs[0]


if os.environ.get("RWKV_JIT_ON") == "1":

    def tmix_lnx_rkvres_xg_bf16_v1(x, r, k, v, r_k, weight, bias, g):
        return _tmix_lnx_rkvres_xg_bf16_v1_jit(x, r, k, v, r_k, weight, bias, g)

else:

    def tmix_lnx_rkvres_xg_bf16_v1(x, r, k, v, r_k, weight, bias, g, _op=_forward_op):
        return _op(x, r, k, v, r_k, weight, bias, g)


########################################################################################################

conditional_load(
    not _INFERENCE_ONLY,
    name="rwkv7_tmix_a_gate_bf16",
    sources=["cuda/rwkv7_tmix_a_gate_bf16.cpp", "cuda/rwkv7_tmix_a_gate_bf16.cu"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-res-usage",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
    ],
    is_python_module=False,
    verbose=True,
)


def _setup_context(ctx, inputs, output):
    del output
    a0, a12 = inputs
    ctx.save_for_backward(a0, a12)


def _backward(ctx, grad_out):
    a0, a12 = ctx.saved_tensors
    return tuple(
        torch.ops.rwkv7_tmix_a_gate_bf16.backward(
            grad_out.contiguous(),
            a0,
            a12,
        )
    )


torch.library.register_autograd(
    "rwkv7_tmix_a_gate_bf16::forward",
    _backward,
    setup_context=_setup_context,
)


def _forward_op(a0, a12):
    return torch.ops.rwkv7_tmix_a_gate_bf16.forward(
        a0.contiguous(),
        a12.contiguous(),
    )


@torch.jit.script
def _tmix_a_gate_bf16_jit(
    a0: torch.Tensor,
    a12: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.rwkv7_tmix_a_gate_bf16.forward(
        a0.contiguous(),
        a12.contiguous(),
    )


if os.environ.get("RWKV_JIT_ON") == "1":

    def tmix_a_gate_bf16(a0, a12):
        return _tmix_a_gate_bf16_jit(a0, a12)

else:

    def tmix_a_gate_bf16(a0, a12, _op=_forward_op):
        return _op(a0, a12)


########################################################################################################

conditional_load(
    not _INFERENCE_ONLY,
    name="rwkv7_tmix_vres_gate_bf16_v1",
    sources=["cuda/rwkv7_tmix_vres_gate_bf16_v1.cpp", "cuda/rwkv7_tmix_vres_gate_bf16_v1.cu"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-res-usage",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
    ],
    is_python_module=False,
    verbose=True,
)


def _setup_context(ctx, inputs, output):
    del output
    v, v_first, v0, v12 = inputs
    ctx.save_for_backward(v, v_first, v0, v12)


def _backward(ctx, grad_out):
    v, v_first, v0, v12 = ctx.saved_tensors
    grad_v, grad_v_first, grad_pre = torch.ops.rwkv7_tmix_vres_gate_bf16_v1.backward(
        grad_out.contiguous(),
        v,
        v_first,
        v0,
        v12,
    )
    grad_v0 = grad_pre.sum(dim=(0, 1), keepdim=True)
    return grad_v, grad_v_first, grad_v0.to(v0.dtype), grad_pre.to(v12.dtype)


torch.library.register_autograd(
    "rwkv7_tmix_vres_gate_bf16_v1::forward",
    _backward,
    setup_context=_setup_context,
)


def _forward_op(v, v_first, v0, v12):
    return torch.ops.rwkv7_tmix_vres_gate_bf16_v1.forward(
        v.contiguous(),
        v_first.contiguous(),
        v0.contiguous(),
        v12.contiguous(),
    )


@torch.jit.script
def _tmix_vres_gate_bf16_v1_jit(
    v: torch.Tensor,
    v_first: torch.Tensor,
    v0: torch.Tensor,
    v12: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.rwkv7_tmix_vres_gate_bf16_v1.forward(
        v.contiguous(),
        v_first.contiguous(),
        v0.contiguous(),
        v12.contiguous(),
    )


if os.environ.get("RWKV_JIT_ON") == "1":

    def tmix_vres_gate_bf16_v1(v, v_first, v0, v12):
        return _tmix_vres_gate_bf16_v1_jit(v, v_first, v0, v12)

else:

    def tmix_vres_gate_bf16_v1(v, v_first, v0, v12, _op=_forward_op):
        return _op(v, v_first, v0, v12)


########################################################################################################

L2WRAP_CE_CUDA_V2 = conditional_load(
    not _INFERENCE_ONLY,
    name="rwkv7_l2wrap_ce_bf16_v2",
    sources=["cuda/rwkv7_l2wrap_ce_bf16_v2.cpp", "cuda/rwkv7_l2wrap_ce_bf16_v2.cu"],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-res-usage",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
    ],
    verbose=True,
)


class L2WrapCrossEntropyCUDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        logits = logits.contiguous()
        targets = targets.contiguous()
        loss, lse, max_vals, argmax = L2WRAP_CE_CUDA_V2.forward(logits, targets)
        ctx.save_for_backward(logits, targets.view(-1), lse, max_vals, argmax)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, lse, max_vals, argmax = ctx.saved_tensors
        grad_logits = L2WRAP_CE_CUDA_V2.backward(
            grad_output.contiguous().float(),
            logits,
            targets,
            lse,
            max_vals,
            argmax,
        )
        return grad_logits, None


def l2wrap_cross_entropy(logits, targets):
    return L2WrapCrossEntropyCUDA.apply(logits, targets)


def _diffusion_confidence_loss(logits, targets):
    """LLaDA-2.0 Confidence-Aware Parallel Training (CAP) auxiliary loss.

    Minimizes the entropy of the predicted distribution H(p_theta(. | x_t, c))
    ONLY at positions that (a) carry CE supervision (targets != -100, i.e. a
    masked b2 ∩ lossable position) AND (b) are *already correctly* predicted
    by the current model (argmax(logits) == target). Sharpens the model's
    confidence on already-correct tokens so parallel decoders can commit more
    of them per step. Selective gating prevents the entropy term from fighting
    CE on still-wrong tokens.

    Memory note: we INDEX-SELECT the (correct & valid) subset BEFORE running
    log_softmax. Otherwise we'd materialize three (B*T, V) fp32 tensors which
    OOMs at training shapes (B=16, T=6144, V=65536 -> ~25 GB each). After
    selection the dimension shrinks to ~hundreds-to-thousands of rows.

    Args:
        logits:  (B, T, V) raw scores
        targets: (B, T)    int64; -100 marks "skip"

    Returns:
        scalar mean entropy over the (correct & valid) positions, or 0 when
        no such positions exist in the batch.
    """
    flat_logits = logits.reshape(-1, logits.size(-1))  # (N, V)
    flat_targets = targets.reshape(-1)  # (N,)

    # All gating decisions are no_grad: argmax + boolean ops produce no gradient.
    with torch.no_grad():
        valid = flat_targets != -100
        if not valid.any():
            return logits.new_zeros(())
        argmax = flat_logits.argmax(dim=-1)
        correct_mask = (argmax == flat_targets) & valid
        if not correct_mask.any():
            return logits.new_zeros(())

    # Subset first: now we only materialize (M, V), M = #correct (typically
    # << B*T). Gradient still flows through the selected rows.
    sub_logits = flat_logits[correct_mask]  # (M, V)

    # Compute entropy via the logsumexp identity to avoid the 0*(-inf)=NaN
    # trap. The naive form `-(softmax(z) * log_softmax(z)).sum()` blows up
    # whenever some logit is so much smaller than max that softmax(z_i) = 0
    # exactly while log_softmax(z_i) = -inf. With bf16 + a 65k-vocab head and
    # peaky predictions, this is common, not rare.
    #   H(p) = -sum_i p_i * log p_i
    #        = -sum_i p_i * (z_i - lse)            [since log p_i = z_i - lse]
    #        =  lse - sum_i p_i * z_i              [since sum p_i = 1]
    # `z_i` is a finite logit (never -inf), so the multiplication is well-defined.
    # Cast to fp32 for stable softmax / logsumexp (bf16 sums can drift).
    sub_logits_f = sub_logits.float()
    lse = torch.logsumexp(sub_logits_f, dim=-1)  # (M,)
    probs = F.softmax(sub_logits_f, dim=-1)  # (M, V)
    entropy = lse - (probs * sub_logits_f).sum(dim=-1)  # (M,)
    return entropy.mean()


########################################################################################################

if int(os.environ["RWKV_HEAD_L2WRAP_CE_CHUNK"]) > 0:
    HEAD_L2WRAP_CE_CHUNK = int(os.environ["RWKV_HEAD_L2WRAP_CE_CHUNK"])
    HEAD_L2WRAP_CE_CUDA_V4 = conditional_load(
        not _INFERENCE_ONLY,
        name="rwkv7_head_l2wrap_ce_bf16_v4",
        sources=["cuda/rwkv7_head_l2wrap_ce_bf16_v4.cpp", "cuda/rwkv7_head_l2wrap_ce_bf16_v4.cu"],
        extra_cflags=["-O3", f"-DHEAD_CE_CHUNK={HEAD_L2WRAP_CE_CHUNK}"],
        extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            f"-DHEAD_CE_CHUNK={HEAD_L2WRAP_CE_CHUNK}",
        ],
        verbose=True,
    )

    class HeadL2WrapCrossEntropyCUDAV4(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, weight, targets):
            hidden = hidden.contiguous()
            weight = weight.contiguous()
            targets = targets.contiguous()
            loss, grad_hidden, grad_weight = HEAD_L2WRAP_CE_CUDA_V4.forward(
                hidden,
                weight,
                targets,
                HEAD_L2WRAP_CE_CHUNK,
            )
            ctx.save_for_backward(grad_hidden, grad_weight)
            return loss

        @staticmethod
        def backward(ctx, grad_output):
            grad_hidden, grad_weight = ctx.saved_tensors
            if grad_output.numel() == 1 and float(grad_output.detach()) == 1.0:
                return grad_hidden, grad_weight, None
            scale = grad_output.to(dtype=torch.float32)
            return (
                grad_hidden * scale.to(grad_hidden.dtype),
                grad_weight * scale.to(grad_weight.dtype),
                None,
            )

    def head_l2wrap_cross_entropy(hidden, weight, targets):
        return HeadL2WrapCrossEntropyCUDAV4.apply(hidden, weight, targets)


########################################################################################################


class RWKV_Tmix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.my_testing = args.my_testing

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C - 1) - 0.5
                zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1**0.3)

            # The four LoRA-style ranks below are chosen by an "n_embd-scaled" heuristic
            # by default, but BlinkDL's released ckpts (e.g. RWKV7-G1f-7.2B) use
            # hand-picked values that don't match those formulas. Allow args overrides
            # so you can load any official ckpt by passing the right CLI flags.
            D_DECAY_LORA = getattr(args, "d_decay_lora", 0) or max(
                32, int(round((2.5 * (C**0.5)) / 32) * 32)
            )
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1, 1, C) + 0.5 + zigzag * 2.5)

            D_AAA_LORA = getattr(args, "d_aaa_lora", 0) or max(
                32, int(round((2.5 * (C**0.5)) / 32) * 32)
            )
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4)

            D_MV_LORA = getattr(args, "d_mv_lora", 0) or max(
                32, int(round((1.7 * (C**0.5)) / 32) * 32)
            )
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 0.73 - linear * 0.4)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            D_GATE_LORA = getattr(args, "d_gate_lora", 0) or max(
                32, int(round((5 * (C**0.5)) / 32) * 32)
            )
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1, 1, C) + 0.71 - linear * 0.1)
            self.k_a = nn.Parameter(torch.zeros(1, 1, C) + 1.02)
            self.r_k = nn.Parameter(torch.zeros(H, N) - 0.04)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5)  # !!! notice eps value !!!

            self.receptance.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.key.weight.data.uniform_(-0.05 / (C**0.5), 0.05 / (C**0.5))
            self.value.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.output.weight.data.zero_()

    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head

        ############################################################
        # slow pytorch version
        # xx = self.time_shift(x) - x
        # xr = x + xx * self.x_r
        # xw = x + xx * self.x_w
        # xk = x + xx * self.x_k
        # xv = x + xx * self.x_v
        # xa = x + xx * self.x_a
        # xg = x + xx * self.x_g
        ############################################################
        # much faster CUDA version
        xr, xw, xk, xv, xa, xg = tmix_mix6_bf16_v5(
            x,
            self.x_r.view(-1),
            self.x_w.view(-1),
            self.x_k.view(-1),
            self.x_v.view(-1),
            self.x_a.view(-1),
            self.x_g.view(-1),
        )
        ############################################################

        r = self.receptance(xr)
        w = (
            self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        )  # will be soft-clamped to (-inf, -0.5) and exp(-exp(w)) in RWKV7_CLAMPW_CUDA kernel
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v  # store the v of the first layer
        else:
            ############################################################
            # slow pytorch version
            # v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
            ############################################################
            # much faster CUDA version
            v12 = (xv @ self.v1) @ self.v2
            v = tmix_vres_gate_bf16_v1(v, v_first, self.v0, v12)  # add value residual
            ############################################################

        ############################################################
        # slow pytorch version
        # a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        ############################################################
        # much faster CUDA version
        a = tmix_a_gate_bf16(self.a0, (xa @ self.a1) @ self.a2)  # a is "in-context learning rate"
        ############################################################

        g = torch.sigmoid(xg @ self.g1) @ self.g2

        ############################################################
        # slow pytorch version
        # kk = k * self.k_k
        # kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        # k = k * (1 + (a-1) * self.k_a)
        # x = RWKV7_CLAMPW_CUDA(r, w, k, v, -kk, kk*a)
        ############################################################
        # much faster CUDA version (!!! fixed eps=1e-12 same as pytorch !!!)
        k, neg_kk, kka = tmix_kk_pre_bf16_v5(
            k,
            self.k_k.view(-1),
            a,
            self.k_a.view(-1),
        )
        x = RWKV7_CLAMPW_CUDA(r, w, k, v, neg_kk, kka)
        ############################################################

        ############################################################
        # slow pytorch version
        # x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        # x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        # x = self.output(x * g)
        ############################################################
        # much faster CUDA version (!!! fixed eps=64e-5 and H=64, also fused x*g !!!)
        x = tmix_lnx_rkvres_xg_bf16_v1(
            x,
            r,
            k,
            v,
            self.r_k,
            self.ln_x.weight,
            self.ln_x.bias,
            g,
        )
        x = self.output(x)
        ############################################################

        return x, v_first


########################################################################################################

# class RWKV_CMix_x070(MyModule): # slow pytorch version
#     def __init__(self, args, layer_id):
#         super().__init__()
#         self.args = args
#         self.layer_id = layer_id
#         self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

#         with torch.no_grad():
#             ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
#             ddd = torch.ones(1, 1, args.n_embd)
#             for i in range(args.n_embd):
#                 ddd[0, 0, i] = i / args.n_embd
#             self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

#         self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
#         self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

#         self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
#         self.value.weight.data.zero_()

#     @MyFunction
#     def forward(self, x):
#         xx = self.time_shift(x) - x

#         k = x + xx * self.x_k
#         k = torch.relu(self.key(k)) ** 2

#         return self.value(k)


class RWKV_CMix_x070(nn.Module):  # fast CUDA version
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        self.key.weight.data.uniform_(-0.5 / (args.n_embd**0.5), 0.5 / (args.n_embd**0.5))
        self.value.weight.data.zero_()

    def forward(self, x):
        return _CmixLayerV2Fn.apply(x, self.x_k.view(-1), self.key.weight, self.value.weight)


########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn

        x = x + self.ffn(self.ln2(x))
        return x, v_first


# class L2Wrap(torch.autograd.Function): # avoid: very slow and takes lots of vram
#     @staticmethod
#     def forward(ctx, loss, y):
#         ctx.save_for_backward(y)
#         return loss

#     @staticmethod
#     def backward(ctx, grad_output):
#         y = ctx.saved_tensors[0]
#         factor = 1e-4 / (y.shape[0] * y.shape[1])
#         maxx, ids = torch.max(y, -1, keepdim=True)
#         gy = torch.zeros_like(y)
#         gy.scatter_(-1, ids, maxx * factor)
#         return (grad_output, grad_output * gy) # original (grad_output, gy) is buggy when grad_output != 1 !!!


########################################################################################################
# State-mode (RNN) helpers used by RWKV.forward_fast.
# Direct port of the demo_fast.py TMix/CMix one+seq routines, but reading weights
# from our existing nn.Module attributes (no flat-dict transposes).
########################################################################################################


@torch.no_grad()
def _tmix_one(att, layer_id, x, x_prev, kv_state, v_first):
    """Single-token TMix in state mode. Shapes: x (C,), x_prev (C,), kv_state (H,N,N) fp32."""
    H = att.n_head
    N = att.head_size
    xx = x_prev - x
    xr = x + xx * att.x_r.view(-1)
    xw = x + xx * att.x_w.view(-1)
    xk = x + xx * att.x_k.view(-1)
    xv = x + xx * att.x_v.view(-1)
    xa = x + xx * att.x_a.view(-1)
    xg = x + xx * att.x_g.view(-1)

    r = att.receptance(xr)
    w_lora = torch.tanh(xw @ att.w1) @ att.w2  # (C,)
    k = att.key(xk)
    v = att.value(xv)
    a = torch.sigmoid(att.a0.view(-1) + (xa @ att.a1) @ att.a2)
    g = torch.sigmoid(xg @ att.g1) @ att.g2

    kk = F.normalize((k * att.k_k.view(-1)).view(H, N), dim=-1, p=2.0).view(H * N)
    k = k * (1 + (a - 1) * att.k_a.view(-1))
    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(att.v0.view(-1) + (xv @ att.v1) @ att.v2)

    # Single-token decay form (from demo_fast forward_one)
    w = torch.exp(
        -0.606531 * torch.sigmoid((att.w0.view(-1) + w_lora).float())
    )  # 0.606531 = exp(-0.5)

    vk = v.view(H, N, 1) @ k.view(H, 1, N)
    ab = (-kk).view(H, N, 1) @ (kk * a).view(H, 1, N)
    kv_state = kv_state * w.view(H, 1, N) + kv_state @ ab.float() + vk.float()
    out = (kv_state.to(dtype=x.dtype) @ r.view(H, N, 1)).view(H * N)

    out = F.group_norm(
        out.view(1, H * N), num_groups=H, weight=att.ln_x.weight, bias=att.ln_x.bias, eps=64e-5
    ).view(H * N)
    out = out + (
        (r * k * att.r_k.view(-1)).view(H, N).sum(dim=-1, keepdim=True) * v.view(H, N)
    ).view(H * N)
    return att.output(out * g), x, kv_state, v_first


@torch.no_grad()
def _tmix_seq(att, layer_id, x, x_prev, kv_state, v_first):
    """Multi-token TMix in state mode. Shapes: x (T,C), x_prev (C,), kv_state (H,N,N) fp32."""
    T = x.shape[0]
    H = att.n_head
    N = att.head_size
    # Fused mix6: replaces cat + sub + 6 elementwise scalar broadcasts
    # with a single kernel launch that reads x once and emits all 6
    # mixed outputs. ~0.5 ms/step speedup on the 32-layer 7.2B path.
    xr, xw, xk, xv, xa, xg = TMIX_MIX6_INFER(
        x,
        x_prev,
        att.x_r.view(-1),
        att.x_w.view(-1),
        att.x_k.view(-1),
        att.x_v.view(-1),
        att.x_a.view(-1),
        att.x_g.view(-1),
    )

    r = att.receptance(xr)
    w_lora = torch.tanh(xw @ att.w1) @ att.w2  # (T, C)
    k = att.key(xk)
    v = att.value(xv)
    a = torch.sigmoid(att.a0.view(-1) + (xa @ att.a1) @ att.a2)
    g = torch.sigmoid(xg @ att.g1) @ att.g2

    kk = F.normalize((k * att.k_k.view(-1)).view(T, H, N), dim=-1, p=2.0).view(T, H * N)
    k = k * (1 + (a - 1) * att.k_a.view(-1))
    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(att.v0.view(-1) + (xv @ att.v1) @ att.v2)

    # Multi-token decay form (from demo_fast forward_seq) — log-domain for the wkv7s kernel.
    # T >= 8: seqv2 kernel (cp.async ping-pong + bf16 __hfma2) is a clean
    # win. T < 8 (incl. T==1 decode): pong setup is wasted; old kernel
    # is faster and bit-equivalent. Disable seqv2 via RWKV_DISABLE_SEQV2=1.
    w_for_kernel = -F.softplus(-(att.w0.view(-1) + w_lora)) - 0.5
    _disable_seqv2 = os.environ.get("RWKV_DISABLE_SEQV2", "0") == "1"
    op = RWKV7S_OP_SEQV2 if (T >= 8 and not _disable_seqv2) else RWKV7S_OP
    # op mutates kv_state in place; caller's state list slot is the same
    # object that gets returned, so the (no-op) rebind keeps semantics.
    out = op(
        kv_state,
        r.contiguous(),
        w_for_kernel.contiguous(),
        k.contiguous(),
        v.contiguous(),
        (-kk).contiguous(),
        (kk * a).contiguous(),
    )

    # Fused post-WKV: group_norm + r·k·r_k residual + gate mul, all in
    # one kernel. Replaces ~3-5 launches with 1.
    out = TMIX_POST_INFER(
        out,
        r.contiguous(),
        k.contiguous(),
        v.contiguous(),
        att.r_k.view(-1),
        att.ln_x.weight,
        att.ln_x.bias,
        g.contiguous(),
    )
    return att.output(out), x[-1, :], kv_state, v_first


@torch.no_grad()
def _cmix_one(ffn, x, x_prev):
    """Single-token CMix in state mode. x (C,), x_prev (C,)."""
    xx = x_prev - x
    k = x + xx * ffn.x_k.view(-1)
    k = torch.relu(ffn.key(k)) ** 2
    return ffn.value(k), x


@torch.no_grad()
def _cmix_seq(ffn, x, x_prev):
    """Multi-token CMix in state mode. x (T,C), x_prev (C,).

    Fused mix-1 kernel mutates ``x_prev`` in place to x[T-1, :] (so
    the caller's state slot for next call already holds the updated
    last position); returns ``x_prev`` so the caller's
    ``state[...] = ...`` rebind is a no-op identity assignment.
    """
    k = CMIX_MIX_INFER(x, x_prev, ffn.x_k.view(-1))
    k = torch.relu(ffn.key(k)) ** 2
    return ffn.value(k), x_prev


########################################################################################################


class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, "dim_att"):
            args.dim_att = args.n_embd
        if not hasattr(args, "dim_ffn"):
            args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)  # default = 3.5x emb size
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def configure_optimizers(self):
        args = self.args

        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        for n, p in self.named_parameters():
            if "att.w0" in n:
                lr_2x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0) and (".weight" in n):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))

        if self.trainer.is_global_zero:
            print("decay", lr_decay, "\n")
            print("1x", lr_1x, "\n")
            print("2x", lr_2x, "\n")

        param_dict = {n: p for n, p in self.named_parameters()}

        optim_groups = [
            {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
            {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
        ]

        # Optimizer choice via --optim:
        #   "adam"      (default): FusedAdam / DeepSpeedCPUAdam — 12 bytes/param Adam state.
        #   "8bit"      : bitsandbytes AdamW8bit — ~3 bytes/param, but unstable for some
        #                  setups (RWKV-7 SFT here has flown out). Skipped if offload is on.
        #   "adafactor" : HF Adafactor — 8 bytes/param (no fp32 v; uses row/col factorization).
        #                  Stays on GPU. Most memory-efficient stable choice for this model.
        # Back-compat: --optim_8bit 1 still selects "8bit".
        optim_choice = str(getattr(args, "optim", "adam") or "adam").lower()
        if bool(getattr(args, "optim_8bit", 0)):
            optim_choice = "8bit"

        if optim_choice == "adafactor":
            # ~33% memory savings vs Adam by storing variance as row+col factors instead
            # of full per-param fp32. No CPU offload needed; sit on GPU. Prefer the HF
            # implementation because its `relative_step=False, scale_parameter=False`
            # mode behaves like a drop-in Adam (just lighter).
            try:
                from transformers.optimization import Adafactor
            except Exception as e:
                raise RuntimeError("optim=adafactor needs `transformers` installed") from e
            if self.deepspeed_offload:
                rank_zero_info(
                    "########## --optim adafactor set but DeepSpeed offload is on; offload will be ignored (Adafactor stays on GPU) ##########"
                )
            if args.weight_decay > 0:
                optim_groups += [
                    {
                        "params": [param_dict[n] for n in lr_decay],
                        "weight_decay": args.weight_decay,
                        "my_lr_scale": 1.0,
                    }
                ]
            return Adafactor(
                optim_groups,
                lr=self.args.lr_init,
                eps=(1e-30, self.args.adam_eps),  # (eps_grad², eps_param)
                clip_threshold=1.0,
                beta1=self.args.beta1,  # use β1 for momentum (otherwise pure RMSProp-like)
                weight_decay=0.0,  # already in optim_groups
                scale_parameter=False,  # Adam-style LR semantics
                relative_step=False,  # use our cosine schedule, not Adafactor's
                warmup_init=False,
            )

        if optim_choice == "8bit":
            # bitsandbytes 8-bit AdamW: m/v stored as int8 with per-block scales
            # (~4x smaller than fp32). Stays on GPU; replaces FusedAdam.
            # When deepspeed_offload is also on, prefer the CPU path instead since
            # bnb's GPU optimizer can't be efficiently offloaded.
            if self.deepspeed_offload:
                rank_zero_info(
                    "########## --optim 8bit set but DeepSpeed offload is on; falling back to DeepSpeedCPUAdam ##########"
                )
            else:
                import bitsandbytes as bnb

                if args.weight_decay > 0:
                    optim_groups += [
                        {
                            "params": [param_dict[n] for n in lr_decay],
                            "weight_decay": args.weight_decay,
                            "my_lr_scale": 1.0,
                        }
                    ]
                    return bnb.optim.AdamW8bit(
                        optim_groups,
                        lr=self.args.lr_init,
                        betas=self.args.betas,
                        eps=self.args.adam_eps,
                    )
                return bnb.optim.Adam8bit(
                    optim_groups,
                    lr=self.args.lr_init,
                    betas=self.args.betas,
                    eps=self.args.adam_eps,
                )

        if args.weight_decay > 0:
            optim_groups += [
                {
                    "params": [param_dict[n] for n in lr_decay],
                    "weight_decay": args.weight_decay,
                    "my_lr_scale": 1.0,
                }
            ]
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(
                    optim_groups,
                    lr=self.args.lr_init,
                    betas=self.args.betas,
                    eps=self.args.adam_eps,
                    bias_correction=True,
                    adamw_mode=True,
                    amsgrad=False,
                )
            return FusedAdam(
                optim_groups,
                lr=self.args.lr_init,
                betas=self.args.betas,
                eps=self.args.adam_eps,
                bias_correction=True,
                adam_w_mode=True,
                amsgrad=False,
            )
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(
                    optim_groups,
                    lr=self.args.lr_init,
                    betas=self.args.betas,
                    eps=self.args.adam_eps,
                    bias_correction=True,
                    adamw_mode=False,
                    weight_decay=0,
                    amsgrad=False,
                )
            return FusedAdam(
                optim_groups,
                lr=self.args.lr_init,
                betas=self.args.betas,
                eps=self.args.adam_eps,
                bias_correction=True,
                adam_w_mode=False,
                weight_decay=0,
                amsgrad=False,
            )

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def _forward_features(self, idx):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        v_first = torch.empty_like(x)
        for block in self.blocks:
            if args.grad_cp == 1:
                x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
            else:
                x, v_first = block(x, v_first)

        x = self.ln_out(x)
        return x

    # ------------------------------------------------------------------
    # State-mode (RNN-style) inference path. Mirrors RWKV-v7 demo_fast.py.
    # No autograd, eval only. Use this for fast sampling / diffusion denoising
    # where the same prefix state is reused across many forwards.
    # ------------------------------------------------------------------

    def init_state(self, device=None, dtype=None):
        """Build a fresh zero state for forward_fast. Returns a list of length n_layer*3."""
        args = self.args
        if device is None:
            device = self.emb.weight.device
        if dtype is None:
            dtype = self.emb.weight.dtype
        H = args.dim_att // args.head_size
        N = args.head_size
        state = []
        for _ in range(args.n_layer):
            state.append(torch.zeros(args.n_embd, dtype=dtype, device=device))  # att_x_prev
            state.append(
                torch.zeros((H, N, N), dtype=torch.float32, device=device)
            )  # att_kv (fp32)
            state.append(torch.zeros(args.n_embd, dtype=dtype, device=device))  # ffn_x_prev
        return state

    @torch.no_grad()
    def forward_fast(self, idx, state=None, full_output=False):
        """State-mode forward (RNN-style). Equivalent in math to ``forward(idx)`` but
        carries an explicit RNN state, so consecutive calls don't redo earlier work.

        Args:
            idx: ``int`` (single token), or 1-D ``LongTensor`` on cuda (a sequence).
            state: list of ``n_layer * 3`` tensors, or ``None`` to start from zero.
            full_output: if False (default) and idx is a sequence, return logits for
                the last position only. If True, return logits for every position.

        Returns: ``(logits, state)``. ``logits`` shape:
            - single-token idx: ``(vocab,)``
            - sequence idx, full_output=False: ``(vocab,)`` (last position)
            - sequence idx, full_output=True:  ``(T, vocab)``
        """
        if state is None:
            state = self.init_state()

        if isinstance(idx, int):
            x = self.emb.weight[idx]  # (C,)
            is_seq = False
        else:
            assert idx.dim() == 1, "forward_fast expects a 1-D token tensor or an int"
            if idx.shape[0] == 1:
                x = self.emb(idx).squeeze(0)
                is_seq = False
            else:
                x = self.emb(idx)  # (T, C)
                is_seq = True

        # Layer 0 ln0 (only first block has it)
        x = self.blocks[0].ln0(x)

        v_first = torch.empty_like(x)
        for i, block in enumerate(self.blocks):
            ln1, ln2 = block.ln1, block.ln2
            att = block.att
            ffn = block.ffn

            xx = ln1(x)
            if is_seq:
                xx, state[i * 3 + 0], state[i * 3 + 1], v_first = _tmix_seq(
                    att, i, xx, state[i * 3 + 0], state[i * 3 + 1], v_first
                )
            else:
                xx, state[i * 3 + 0], state[i * 3 + 1], v_first = _tmix_one(
                    att, i, xx, state[i * 3 + 0], state[i * 3 + 1], v_first
                )
            x = x + xx

            xx = ln2(x)
            if is_seq:
                xx, state[i * 3 + 2] = _cmix_seq(ffn, xx, state[i * 3 + 2])
            else:
                xx, state[i * 3 + 2] = _cmix_one(ffn, xx, state[i * 3 + 2])
            x = x + xx

        if is_seq and not full_output:
            x = x[-1]
        x = self.ln_out(x)
        logits = self.head(x)
        return logits, state

    if int(os.environ["RWKV_HEAD_L2WRAP_CE_CHUNK"]) > 0:  # saves 70~80% VRAM

        def forward(self, idx):
            return self._forward_features(idx)

        def training_step(self, batch, batch_idx):
            idx, targets = batch
            if getattr(self.args, "diffusion_mode", 0) == 1:
                # Diffusion mode needs full logits + ignore_index, which the chunked
                # CE kernel doesn't support; apply head explicitly and use F.cross_entropy.
                hidden = self(idx)
                logits = self.head(hidden)
                # Manual sum / n_valid avoids NaN when a batch has zero valid
                # (non -100) targets — F.cross_entropy(reduction='mean', ignore_index=-100)
                # silently returns NaN in that case (0/0 in the per-sample average),
                # and one NaN forever poisons Adam's m/v.
                n_valid = (targets != -100).sum().clamp_min(1)
                ce_sum = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                loss = ce_sum / n_valid
                conf_lambda = float(getattr(self.args, "diff_conf_lambda", 0.0))
                if conf_lambda > 0.0:
                    loss = loss + conf_lambda * _diffusion_confidence_loss(logits, targets)
                # Early warning: if loss / logits go non-finite, surface it loudly
                # in the rank-0 log so we can catch it on the first occurrence
                # rather than discover wrecked weights tens of steps later.
                if not torch.isfinite(loss):
                    rank_zero_info(
                        f"[NaN-guard] non-finite loss at step {self.global_step}: "
                        f"loss={loss.item()}, logits min={logits.min().item():.3g}, "
                        f"max={logits.max().item():.3g}, n_valid={n_valid.item()}"
                    )
                with torch.no_grad():
                    valid = targets != -100
                    n_masked = valid.sum().clamp_min(1)
                    correct = ((logits.argmax(-1) == targets) & valid).sum()
                    self.trainer.my_diff_acc = (correct.float() / n_masked.float()).item()
                    self.trainer.my_diff_n_masked = n_masked.item()
                return loss
            hidden = self(idx)
            return head_l2wrap_cross_entropy(hidden, self.head.weight, targets)

    else:

        def forward(self, idx):
            x = self._forward_features(idx)
            x = self.head(x)
            return x

        def training_step(self, batch, batch_idx):
            idx, targets = batch
            logits = self(idx)
            if getattr(self.args, "diffusion_mode", 0) == 1:
                # Diffusion mode: per-token CE with ignore_index for non-b2 / non-masked
                # positions and tail padding. The L2-wrap CE CUDA kernel doesn't support
                # ignore_index, so fall back to PyTorch's F.cross_entropy.
                # Manual sum / n_valid avoids NaN when a batch has zero valid
                # (non -100) targets — F.cross_entropy(reduction='mean', ignore_index=-100)
                # silently returns NaN in that case (0/0), poisoning Adam state forever.
                n_valid = (targets != -100).sum().clamp_min(1)
                ce_sum = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                loss = ce_sum / n_valid
                conf_lambda = float(getattr(self.args, "diff_conf_lambda", 0.0))
                if conf_lambda > 0.0:
                    loss = loss + conf_lambda * _diffusion_confidence_loss(logits, targets)
                # Early warning: if loss / logits go non-finite, surface it loudly
                # in the rank-0 log so we can catch it on the first occurrence
                # rather than discover wrecked weights tens of steps later.
                if not torch.isfinite(loss):
                    rank_zero_info(
                        f"[NaN-guard] non-finite loss at step {self.global_step}: "
                        f"loss={loss.item()}, logits min={logits.min().item():.3g}, "
                        f"max={logits.max().item():.3g}, n_valid={n_valid.item()}"
                    )
                with torch.no_grad():
                    valid = targets != -100
                    n_masked = valid.sum().clamp_min(1)
                    correct = ((logits.argmax(-1) == targets) & valid).sum()
                    self.trainer.my_diff_acc = (correct.float() / n_masked.float()).item()
                    self.trainer.my_diff_n_masked = n_masked.item()
                return loss

            ############################################################
            # slow pytorch version (!!! SLOW AND TAKES 40% MORE VRAM !!!)
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # return L2Wrap.apply(loss, logits)
            ############################################################
            # much faster CUDA version (!!! fixed 1e-4 factor !!!)
            return l2wrap_cross_entropy(logits, targets)

    def training_step_end(self, batch_parts):
        all = self.all_gather(batch_parts)
        if self.trainer.is_global_zero:
            self.trainer.my_loss_all = all

    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        n_params = 0
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            s3 = str(shape[3]) if len(shape) > 3 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}", end="")

            scale = 1.0
            if (
                "ln_" in n
                or ".ln" in n
                or "time_" in n
                or "_mask" in n
                or "pos_emb" in n
                or ".mask." in n
                or n.endswith("_w")
                or n.endswith("_w1")
                or n.endswith("_w2")
                or n.endswith("_bias")
                or (".weight" not in n)
            ):
                if "ln_x.weight" in n:
                    layer_scale = (1 + int(n.split(".")[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale**0.7)
                else:
                    m[n] = p
                print()
            elif n == "emb.weight":
                m[n] = p
                scale = -1e-4
                nn.init.uniform_(m[n], a=scale, b=-scale)
                print(f" [scale {scale}]")
            elif n == "head.weight":
                m[n] = p
                if self.args.vocab_size > self.args.n_embd:
                    scale = 0.5 * math.sqrt(self.args.vocab_size / self.args.n_embd)
                else:
                    scale = 0.5
                nn.init.orthogonal_(m[n], gain=scale)
                print(f" [scale {scale}]")
            else:
                assert n.endswith(".weight")  # should always be true

                zero = [
                    ".att.output.",
                    ".ffn.value.",
                    ".ffn.receptance.",
                    ".ffnPre.value.",
                    ".ffnPre.receptance.",
                    "head_q.",
                    ".oo.",
                    ".rr.",
                ]

                for kk in zero:
                    if kk in n:
                        scale = 0

                for kk in [".att.key."]:
                    if kk in n:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in n:
                        scale = 0.1

                print(f" [scale {scale}]")

                if self.args.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=scale)

            m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()
            n_params += m[n].numel()

        print("model params", n_params)
        gc.collect()
        torch.cuda.empty_cache()
        return m
