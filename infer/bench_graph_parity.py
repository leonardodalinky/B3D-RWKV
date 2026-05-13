"""Verify CUDA Graph forward_fast produces same logits as eager."""
from __future__ import annotations
import os, sys, time
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
        n_layer=32, n_embd=4096, dim_att=4096,
        dim_ffn=int((4096 * 3.5) // 32 * 32),
        head_size=64, vocab_size=65536, ctx_len=4096,
        my_testing="x070", grad_cp=0, weight_decay=0.0,
        lr_init=0.0, lr_final=0.0, betas=(0.9, 0.99), adam_eps=1e-18,
        layerwise_lr=0, my_pile_stage=0, train_stage=0, diffusion_mode=0,
        d_decay_lora=128, d_aaa_lora=128, d_mv_lora=96, d_gate_lora=480,
    )


ckpt = os.environ.get(
    "CKPT",
    "/data/rsync/RWKV/DiffuRWKV/train/out/diff-L32-D4096-x070-blk32-ctx6144/rwkv-21.pth",
)
print(f"[parity] loading {ckpt}", flush=True)
model = ds.build_model(ckpt, build_args())
print(f"[parity] loaded", flush=True)

BLOCK = 32
MASK = 65535
torch.manual_seed(0)
cur = torch.randint(1, 65000, (BLOCK,), dtype=torch.long, device="cuda")

# Build a non-trivial ctx_state (run prompt through forward_fast)
prompt = torch.randint(1, 65000, (50,), dtype=torch.long, device="cuda")
_, ctx_state = model.forward_fast(prompt, None, full_output=False)
print(f"[parity] ctx_state[1] (kv_state layer0) |max|={ctx_state[1].abs().max().item():.4f}")

# ---- Eager forward ----
step_state_eager = [s.clone() for s in ctx_state]
inp = torch.cat([cur, cur], dim=0)
logits_eager, _ = model.forward_fast(inp, step_state_eager, full_output=True)
torch.cuda.synchronize()
print(f"[parity] eager logits  |max|={logits_eager.abs().max().item():.4f}  argmax[0]={logits_eager[0].argmax().item()}")

# Eager-vs-eager determinism check (RIGHT here, before any graph stuff).
step_state_eager2 = [s.clone() for s in ctx_state]
logits_eager2, _ = model.forward_fast(inp, step_state_eager2, full_output=True)
torch.cuda.synchronize()
deager = (logits_eager.float() - logits_eager2.float()).abs().max().item()
print(f"[parity] eager-vs-eager max|Δ|={deager:.4e} (should be 0)")

# ---- Pure CUDAGraph test (no make_graphed_callables) ----
# Try the SIMPLEST possible capture: 1 forward_fast call, no copy_, no state reset.
# State init OUTSIDE the graph. Run twice with reset between to see if it's
# deterministic at all.
print("\n[parity] Minimal-capture test: ONE forward_fast in graph, no state reset inside")
inp_g = torch.cat([cur, cur], dim=0)
state_g = [s.clone() for s in ctx_state]

# Warmup
for _ in range(3):
    state_g_w = [s.clone() for s in ctx_state]
    _ = model.forward_fast(inp_g, state_g_w, full_output=True)
torch.cuda.synchronize()

# Capture
g = torch.cuda.CUDAGraph()
state_g_static = [s.clone() for s in ctx_state]
with torch.cuda.graph(g):
    logits_static, _ = model.forward_fast(inp_g, state_g_static, full_output=True)
    logits_out = logits_static.clone()

# Replay 1
# Reset state externally
for dst, src in zip(state_g_static, ctx_state):
    dst.copy_(src)
g.replay()
torch.cuda.synchronize()
r1 = logits_out.clone()

# Replay 2
for dst, src in zip(state_g_static, ctx_state):
    dst.copy_(src)
g.replay()
torch.cuda.synchronize()
r2 = logits_out.clone()
print(f"  Replay1 |max|={r1.abs().max().item():.4f}  argmax[0]={r1[0].argmax().item()}")
print(f"  Replay2 |max|={r2.abs().max().item():.4f}  argmax[0]={r2[0].argmax().item()}")
print(f"  Replay1-vs-Replay2 max|Δ|={(r1.float()-r2.float()).abs().max().item():.4e}")
deager_r1 = (logits_eager.float() - r1.float()).abs().max().item()
print(f"  Eager-vs-Replay1 max|Δ|={deager_r1:.4e}")

# ---- Full GraphStepRunner test ----
print("\n[parity] === GraphStepRunner test ===")
runner = ds.GraphStepRunner(model, BLOCK, MASK)
runner.warmup_and_capture(n_warmup=3)
runner.set_ctx_state(ctx_state)
logits_graph = runner.step(cur).clone()
torch.cuda.synchronize()
print(f"[parity] graph logits  |max|={logits_graph.abs().max().item():.4f}  argmax[0]={logits_graph[0].argmax().item()}")

# Eager determinism check: 2 calls with same input should match.
step_state_eager2 = [s.clone() for s in ctx_state]
logits_eager2, _ = model.forward_fast(inp, step_state_eager2, full_output=True)
torch.cuda.synchronize()
deager = (logits_eager.float() - logits_eager2.float()).abs()
print(f"[parity] eager-vs-eager max|Δ|={deager.max().item():.4e} (should be 0)")

# Compare
dlog = (logits_eager.float() - logits_graph.float()).abs()
print(f"[parity] max|Δlogits|={dlog.max().item():.4e}  mean|Δlogits|={dlog.mean().item():.4e}")
agree = (logits_eager.argmax(dim=-1) == logits_graph.argmax(dim=-1)).float().mean().item()
print(f"[parity] argmax agreement: {agree*100:.2f}%")

# Run a SECOND step to check for state leakage between replays
logits_graph2 = runner.step(cur).clone()
torch.cuda.synchronize()
print(f"[parity] graph 2nd replay |max|={logits_graph2.abs().max().item():.4f}")
dlog2 = (logits_graph.float() - logits_graph2.float()).abs()
print(f"[parity] 1st vs 2nd replay max|Δ|={dlog2.max().item():.4e} (should be 0 — same input!)")
