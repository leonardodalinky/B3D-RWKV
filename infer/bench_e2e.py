"""End-to-end tok/s bench: with vs without CUDA graph.

Loads the model once, runs forward_fast in a tight denoise-step loop
for a fixed number of iterations under each backend, reports ms/iter
and effective tok/s assuming 32 tokens / step (block_size=32, 1 token
committed per denoise step in steady state — see threshold strategy).

This measures the inner-loop throughput, NOT including ckpt load,
prompt prefill, or post-block ctx_state advance.
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


def build_args():
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
    ITERS = int(os.environ.get("ITERS", "100"))
    WARMUP = 5

    print(f"[bench_e2e] loading model from {ckpt}", flush=True)
    t0 = time.time()
    model = ds.build_model(ckpt, build_args())
    print(f"[bench_e2e] loaded in {time.time()-t0:.1f}s", flush=True)

    mask_id = 65535
    cur = torch.full((BLOCK,), mask_id, dtype=torch.long, device="cuda")
    ctx_state = model.init_state()

    # ---- Path A: eager (current production path) ----
    print(f"\n[bench_e2e] EAGER path ({ITERS} iters)")
    for _ in range(WARMUP):
        step_state = [s.clone() for s in ctx_state]
        inp = torch.cat([cur, cur], dim=0)
        _ = model.forward_fast(inp, step_state, full_output=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        step_state = [s.clone() for s in ctx_state]
        inp = torch.cat([cur, cur], dim=0)
        _ = model.forward_fast(inp, step_state, full_output=True)
    torch.cuda.synchronize()
    eager_ms = 1000 * (time.time() - t0) / ITERS

    # ---- Path B: CUDA graph ----
    print(f"\n[bench_e2e] CUDA-GRAPH path ({ITERS} iters)")
    runner = ds.GraphStepRunner(model, BLOCK, mask_id)
    print(f"  capturing...", flush=True)
    t_cap = time.time()
    runner.warmup_and_capture(n_warmup=3)
    runner.set_ctx_state(ctx_state)
    print(f"  capture done in {time.time()-t_cap:.1f}s", flush=True)

    for _ in range(WARMUP):
        _ = runner.step(cur)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        _ = runner.step(cur)
    torch.cuda.synchronize()
    graph_ms = 1000 * (time.time() - t0) / ITERS

    # Report
    speedup = eager_ms / graph_ms
    print(f"\n[bench_e2e] === Results ===")
    print(f"  Eager     :  {eager_ms:7.2f} ms / denoise step")
    print(f"  CUDA Graph:  {graph_ms:7.2f} ms / denoise step")
    print(f"  Speedup   :  {speedup:.2f}x")

    # Translate to tok/s, assuming threshold strategy commits ~1
    # token per step in steady state (conservative). With block_size
    # committed tokens per `block_size`-step block, tok/s = (block /
    # block_steps) / ms_per_step / 1000. For a model that commits more
    # tokens per step the effective tok/s is higher; this is a floor.
    eager_toks = (BLOCK / BLOCK) / (eager_ms / 1000)  # 1 tok/step
    graph_toks = (BLOCK / BLOCK) / (graph_ms / 1000)
    print(f"\n  At 1 commit/step (worst case threshold):")
    print(f"    Eager:      {eager_toks:.1f} tok/s")
    print(f"    Graph:      {graph_toks:.1f} tok/s")
    # If the model is confident, it commits MORE per step. With B=32 and
    # ~steps=32 → average commits/step = 1 if always pure threshold;
    # but in practice high-conf positions cluster early -> earlier blocks
    # finish in fewer denoise iters. End-to-end effective tok/s usually
    # lands at ~2-4x the worst-case floor.


if __name__ == "__main__":
    main()
