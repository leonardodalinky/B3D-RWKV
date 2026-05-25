"""Profile a single denoise step end-to-end to find the actual bottleneck.

Loads the real ckpt, runs forward_fast at the inference shape (T=64),
measures: clone state, forward_fast, sampling logic. Also enables PyTorch
profiler for one step to dump per-op breakdown.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

_REPO = Path(__file__).resolve().parent.parent
_INFER = _REPO / "infer"
for p in (str(_REPO), str(_INFER)):
    if p not in sys.path:
        sys.path.insert(0, p)

import diffusion_sample as ds  # noqa


def build_args(ckpt: str):
    return SimpleNamespace(
        n_layer=32,
        n_embd=4096,
        dim_att=4096,
        dim_ffn=int((4096 * 3.5) // 32 * 32),
        head_size=64,
        vocab_size=65536,
        ctx_len=4096,
        my_testing="x070",
        grad_cp=0,
        weight_decay=0.0,
        lr_init=0.0,
        lr_final=0.0,
        betas=(0.9, 0.99),
        adam_eps=1e-18,
        layerwise_lr=0,
        my_pile_stage=0,
        train_stage=0,
        diffusion_mode=0,
        d_decay_lora=128,
        d_aaa_lora=128,
        d_mv_lora=96,
        d_gate_lora=480,
    )


def main():
    ckpt = os.environ.get(
        "CKPT",
        "/data/rsync/RWKV/DiffuRWKV/train/out/diff-L32-D4096-x070-blk32-ctx6144/rwkv-21.pth",
    )
    BLOCK = 32
    print(f"[profile] loading model from {ckpt}", flush=True)
    t0 = time.time()
    model = ds.build_model(ckpt, build_args(ckpt))
    print(f"[profile] loaded in {time.time()-t0:.1f}s", flush=True)

    mask_id = 65535
    cur = torch.full((BLOCK,), mask_id, dtype=torch.long, device="cuda")
    ctx_state = model.init_state()

    # Warmup
    for _ in range(3):
        step_state = [s.clone() for s in ctx_state]
        inp = torch.cat([cur, cur], dim=0)
        _ = model.forward_fast(inp, step_state, full_output=True)
        torch.cuda.synchronize()

    # ---- Coarse timing (1 denoise step = 1 clone + 1 fwd_fast at T=64) ----
    iters = 50
    print(f"\n[profile] coarse: {iters} denoise-step iters at T={2*BLOCK}")
    torch.cuda.synchronize()
    t_clone, t_fwd = 0.0, 0.0
    for _ in range(iters):
        t0 = time.time()
        step_state = [s.clone() for s in ctx_state]
        torch.cuda.synchronize()
        t_clone += time.time() - t0

        t0 = time.time()
        inp = torch.cat([cur, cur], dim=0)
        _ = model.forward_fast(inp, step_state, full_output=True)
        torch.cuda.synchronize()
        t_fwd += time.time() - t0
    print(f"  clone state:   {1000*t_clone/iters:.2f} ms/iter")
    print(f"  forward_fast:  {1000*t_fwd/iters:.2f} ms/iter")
    print(f"  total/step:    {1000*(t_clone+t_fwd)/iters:.2f} ms/iter")
    print(f"  -> at 32 steps/block, 64 blocks (2048 tok): " f"{32 * 64 * (t_clone + t_fwd):.1f}s")

    # ---- PyTorch profiler for one step: WHERE does the time go? ----
    print("\n[profile] torch.profiler breakdown of one forward_fast(T=64)")
    step_state = [s.clone() for s in ctx_state]
    inp = torch.cat([cur, cur], dim=0)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        _ = model.forward_fast(inp, step_state, full_output=True)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))


if __name__ == "__main__":
    main()
