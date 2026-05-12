"""Iterative denoising sampler for a DiffuRWKV checkpoint, using ``forward_fast``
(RNN/state mode) so the prompt + previously-committed blocks are processed at
most once each, regardless of how many denoising steps the current block runs.

Layout per logical block, mirroring training:

    [b1_masked] [b2_masked == b1] [b3_clean]

At inference we don't have b3 yet, so each denoising step feeds [b1, b2] from
the cloned ctx_state and reads logits at the b2 (last B) positions:

    ctx_state   = process(prompt) once            # state after the prompt
    for block k in 0..n_blocks-1:
        cur = [MASK]*B
        for step t in 1..T:
            step_state = clone(ctx_state)
            logits, _  = forward_fast(cur ++ cur, step_state, full_output=True)
            commit floor(B*t/T) most-confident b2 positions
        ctx_state = forward_fast(cur, ctx_state)  # advance through committed clean block

Notes:
- bf16 / GPU only. cuda/wkv7s.{cu,op.cpp} have been switched from at::Half
  to at::BFloat16, so the state kernel matches training precision exactly.
- forward_fast uses the wkv7s state kernel; no CHUNK_LEN alignment needed.

Usage:
  python train/diffusion_sample.py \
      --ckpt out/diff-.../rwkv-9.pth \
      --n_layer 12 --n_embd 768 \
      --block_size 128 --gen_len 256 --steps 16 \
      --prompt "User: explain RWKV in two sentences.\n\nAssistant:"
"""
import argparse
import math
import os
import sys
from types import SimpleNamespace

import torch

# Add repo root (parent of train/) to sys.path so `import tokenizer` works.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from tokenizer import RWKVTokenizer  # noqa: E402


def build_model(ckpt: str, args_for_model):
    os.environ["RWKV_HEAD_L2WRAP_CE_CHUNK"] = "0"
    os.environ["RWKV_JIT_ON"] = "1"
    # The wkv7s state kernel (cuda/wkv7s.{cu,op.cpp}) is built for bf16 here
    # (we toggled the typedef from at::Half to at::BFloat16). Numerics match
    # training exactly. If you ever revert the typedef back to at::Half,
    # change this to "fp16" and cast the model below to torch.float16.
    os.environ["RWKV_FLOAT_MODE"] = "bf16"
    os.environ["RWKV_MY_TESTING"] = args_for_model.my_testing
    os.environ["RWKV_CTXLEN"] = str(args_for_model.ctx_len)
    os.environ["RWKV_HEAD_SIZE"] = str(args_for_model.head_size)

    # model.py JIT-loads "cuda/*.cu" with relative paths, so we must cwd into
    # train/ before importing it. Save the original cwd so --ckpt etc. resolve.
    train_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, train_dir)
    _orig_cwd = os.getcwd()
    os.chdir(train_dir)
    try:
        from src.model import RWKV  # noqa: E402
    finally:
        os.chdir(_orig_cwd)

    model = RWKV(args_for_model)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = {k.replace("_forward_module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model = model.to(dtype=torch.bfloat16, device="cuda").eval()

    # ---- DEBUG: confirm ckpt actually has trained content ----
    with torch.no_grad():
        emb = model.emb.weight
        head = model.head.weight
        mask_id = args_for_model.vocab_size - 1
        print(f"[DBG] ckpt           = {ckpt}")
        print(f"[DBG] emb.weight     dtype={emb.dtype} shape={tuple(emb.shape)} "
              f"abs_mean={emb.float().abs().mean().item():.4f} "
              f"max={emb.float().abs().max().item():.4f}")
        print(f"[DBG] head.weight    dtype={head.dtype} shape={tuple(head.shape)} "
              f"abs_mean={head.float().abs().mean().item():.4f}")
        print(f"[DBG] emb[MASK={mask_id}] abs_mean={emb[mask_id].float().abs().mean().item():.4f}")
        print(f"[DBG] emb[EOS=0]      abs_mean={emb[0].float().abs().mean().item():.4f}")
        print(f"[DBG] head[EOS=0]     abs_mean={head[0].float().abs().mean().item():.4f}")
        # First block weights
        b0 = model.blocks[0].att
        print(f"[DBG] blocks.0.att.w0 abs_mean={b0.w0.float().abs().mean().item():.4f}")
        print(f"[DBG] blocks.0.att.w1 abs_mean={b0.w1.float().abs().mean().item():.4f}")
        print(f"[DBG] blocks.0.att.receptance.weight abs_mean="
              f"{b0.receptance.weight.float().abs().mean().item():.4f}")
    # ---- end DEBUG ----
    return model


def _clone_state(state):
    """Deep-copy a state list so caller can branch off without polluting the original."""
    return [s.clone() for s in state]


# Tokens reserved for training-only roles, never emitted during real generation.
# 65534 = PAD, 65535 = MASK. Strip them before decoding so a single bad token
# doesn't blow up the whole utf-8 decode pass.
_NON_TEXT_IDS = {65534, 65535}


def _decode_one_segment(tok, buf: list[int]) -> str:
    """Decode a contiguous run of real-text token ids. If the whole run fails
    (typically a mid-byte split inside a multi-byte utf-8 char produced by the
    sampler filling noise after EOS), progressively shrink the tail until what
    remains decodes, then mark the dropped tail. This guarantees we never lose
    the prefix to a single bad trailing token.
    """
    try:
        return tok.decode(buf)
    except Exception:
        pass
    # Progressive tail-trim: keep dropping the last token until the prefix
    # decodes cleanly. Cheap because utf-8 mis-splits resolve within a few
    # tokens of the boundary.
    n = len(buf)
    for cut in range(n - 1, -1, -1):
        head = buf[:cut]
        try:
            decoded = tok.decode(head) if head else ""
            return decoded + f"<undecodable tail x{n - cut}>"
        except Exception:
            continue
    return f"<undecodable x{n}>"


def _safe_decode(tok, ids: list[int]) -> str:
    """Decode a list of token ids to text, gracefully handling tokens that the
    RWKV-world tokenizer can't represent (PAD/MASK) and mid-byte utf-8 splits.

    Strategy: try the whole sequence with non-text ids stripped first; if that
    fails, segment by PAD/MASK boundaries and decode each segment with the
    progressive-trim helper above.
    """
    clean = [t for t in ids if t not in _NON_TEXT_IDS]
    try:
        return tok.decode(clean)
    except Exception:
        pass
    out, buf = [], []
    for t in ids:
        if t in _NON_TEXT_IDS:
            if buf:
                out.append(_decode_one_segment(tok, buf))
                buf = []
            out.append(f"<{t}>")
        else:
            buf.append(t)
    if buf:
        out.append(_decode_one_segment(tok, buf))
    return "".join(out)


@torch.no_grad()
def denoise_block_fast(model, ctx_state, block_size: int, mask_id: int,
                       steps: int, temperature: float, top_k: int,
                       top_p: float = 1.0,
                       penalty: torch.Tensor | None = None,
                       strategy: str = "threshold",
                       conf_threshold: float = 0.95,
                       min_per_step: int = 0) -> torch.Tensor:
    """Iterative denoising of one block, starting from ``ctx_state`` (not mutated).

    Two commit strategies:

    * ``linear``    — old behavior: at iteration t commit the top-confidence
      positions until exactly ``floor(block_size * t / steps)`` are clean.
      Always runs the full ``steps`` iterations.

    * ``threshold`` — LLaDA-2.0 threshold + fallback:
        Phase 1: accept ALL still-masked positions whose token probability
                 exceeds ``conf_threshold`` (0.95 in the paper).
        Phase 2: if Phase 1 yielded fewer than ``min_per_step`` commits,
                 fallback: commit the top-``min_per_step`` most-confident
                 positions instead (regardless of threshold).
        This adapts to model confidence: confident steps commit many tokens
        in parallel (fewer iterations), uncertain steps still make
        guaranteed progress (no stalling). ``steps`` becomes a safety cap.

    Returns the committed clean block as a 1-D LongTensor on cuda.
    """
    cur = torch.full((block_size,), mask_id, dtype=torch.long, device="cuda")
    is_masked = torch.ones(block_size, dtype=torch.bool, device="cuda")
    pred = torch.zeros(block_size, dtype=torch.long, device="cuda")

    if min_per_step <= 0:
        # Default the fallback floor to the linear schedule's pace, so a fully
        # uncertain run is no worse than the old behavior.
        min_per_step = max(1, block_size // max(steps, 1))

    for t in range(1, steps + 1):
        if not is_masked.any():
            break  # threshold mode may finish early
        # Branch off ctx_state — wkv7s mutates state in place, so we must clone.
        step_state = _clone_state(ctx_state)
        # b1 + b2 (both = current best guess). Length 2B, decoupled from ctx length.
        inp = torch.cat([cur, cur], dim=0)
        logits, _ = model.forward_fast(inp, step_state, full_output=True)
        b2_logits = logits[block_size:].float()         # [B, V]
        # Always suppress PAD (65534) and MASK (65535): these are training-only
        # tokens and there's no valid byte-level decoding for them. The model
        # now learns to predict PAD after EOS (because b3 carries pad in
        # training); without suppression that leaks into generation and breaks
        # the tokenizer's decoder. EOS (id=0) is intentionally NOT suppressed
        # so the early-stop logic in run_one can detect it and truncate.
        pad_id = 65534
        b2_logits[:, pad_id] = float("-inf")
        b2_logits[:, mask_id] = float("-inf")

        # ---- DEBUG: dump logits stats on the very first denoise step ----
        if t == 1 and not getattr(denoise_block_fast, "_dbg_done", False):
            denoise_block_fast._dbg_done = True
            full_lg = logits.float()
            print(f"[DBG] full logits: shape={tuple(full_lg.shape)} "
                  f"min={full_lg.min().item():.3f} max={full_lg.max().item():.3f} "
                  f"mean={full_lg.mean().item():.3f} "
                  f"finite={bool(torch.isfinite(full_lg).all())}")
            print(f"[DBG] b2 logits:   shape={tuple(b2_logits.shape)} "
                  f"min={b2_logits.min().item():.3f} max={b2_logits.max().item():.3f}")
            top5_vals, top5_idx = b2_logits[0].topk(5)
            print(f"[DBG] b2[pos=0] top5 ids:  {top5_idx.tolist()}")
            print(f"[DBG] b2[pos=0] top5 vals: {[round(v, 3) for v in top5_vals.tolist()]}")
            print(f"[DBG] b2[pos=0] logit[0]={b2_logits[0, 0].item():.3f} "
                  f"logit[mask_id={mask_id}]={b2_logits[0, mask_id].item():.3f}")
            # Also check b1 (first half) - should be similar to b2 in magnitude
            b1_logits = logits[:block_size].float()
            b1_top5_vals, b1_top5_idx = b1_logits[0].topk(5)
            print(f"[DBG] b1[pos=0] top5 ids:  {b1_top5_idx.tolist()}")
            print(f"[DBG] b1[pos=0] top5 vals: {[round(v, 3) for v in b1_top5_vals.tolist()]}")
        # ---- end DEBUG ----

        # Subtract presence + frequency penalty (broadcast across all B
        # positions). `penalty[v]` is the precomputed per-vocab adjustment
        # built from history. Applied BEFORE temperature/top_k/top_p so the
        # penalty's effect propagates through all subsequent shaping.
        if penalty is not None:
            b2_logits = b2_logits - penalty.unsqueeze(0)

        if temperature != 1.0:
            b2_logits = b2_logits / max(temperature, 1e-6)
        if top_k > 0:
            v, _ = torch.topk(b2_logits, k=top_k, dim=-1)
            b2_logits = b2_logits.masked_fill(b2_logits < v[:, -1:], float("-inf"))
        if 0.0 < top_p < 1.0:
            # Nucleus sampling: per-position keep the smallest set of tokens
            # whose cumulative prob exceeds top_p. Vectorized across B.
            sorted_logits, sorted_idx = torch.sort(b2_logits, descending=True, dim=-1)
            cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # mask of tokens to remove (those AFTER the first one that pushed
            # cumulative prob over top_p)
            remove_sorted = cumprobs > top_p
            # always keep the very first one (highest-prob)
            remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
            remove_sorted[..., 0] = False
            # scatter back to original vocab order, set those logits to -inf
            remove_mask = torch.zeros_like(b2_logits, dtype=torch.bool)
            remove_mask.scatter_(-1, sorted_idx, remove_sorted)
            b2_logits = b2_logits.masked_fill(remove_mask, float("-inf"))
        probs = b2_logits.softmax(dim=-1)
        # When temperature > 0 we sample to break out of argmax-induced
        # repetition loops (esp. relevant for early-training models that
        # have a sharp prior on common phrases). Confidence is then the
        # probability of the *sampled* token, not the argmax probability,
        # so that commit ordering still favors high-confidence positions.
        if temperature > 0:
            pred = torch.multinomial(probs, num_samples=1).squeeze(-1)
            confidence = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
        else:
            confidence, pred = probs.max(dim=-1)
        # Already-committed positions don't compete in this round's argsort.
        confidence = confidence.masked_fill(~is_masked, float("-inf"))

        if strategy == "linear":
            target_clean = min(int(math.floor(block_size * t / steps)), block_size)
            n_commit = max(0, target_clean - int((~is_masked).sum().item()))
            if n_commit > 0:
                _, idx = torch.topk(confidence, k=n_commit)
            else:
                idx = torch.empty(0, dtype=torch.long, device="cuda")
        elif strategy == "threshold":
            # Phase 1: every still-masked position above conf_threshold commits in parallel.
            phase1 = (confidence > conf_threshold) & is_masked
            n_phase1 = int(phase1.sum().item())
            if n_phase1 >= min_per_step:
                idx = torch.nonzero(phase1, as_tuple=False).squeeze(-1)
            else:
                # Phase 2 fallback: take the top-min_per_step (or all remaining
                # if fewer are still masked) by confidence, regardless of threshold.
                n_remaining = int(is_masked.sum().item())
                k = min(min_per_step, n_remaining)
                _, idx = torch.topk(confidence, k=k)
        else:
            raise ValueError(f"unknown strategy: {strategy!r}")

        if idx.numel() > 0:
            n_pred_eos = int((pred == 0).sum().item())
            n_pred_mask = int((pred == mask_id).sum().item())
            print(f"[DBG step {t}] strategy={strategy} commits={idx.numel()}  "
                  f"#pred==0: {n_pred_eos}/{block_size}  "
                  f"#pred==MASK: {n_pred_mask}/{block_size}")
            print(f"[DBG step {t}] commit idx={sorted(idx.tolist())[:8]}...  "
                  f"  vals={pred[idx].tolist()[:8]}")
            cur[idx] = pred[idx]
            is_masked[idx] = False
            print(f"[DBG step {t}] cur[:8]={cur[:8].tolist()}  "
                  f"is_masked.sum()={int(is_masked.sum().item())}")

    # Final cleanup: any still-masked position -> argmax from the last step's logits.
    if is_masked.any():
        cur[is_masked] = pred[is_masked]
    return cur


def run_one(model, tok, mask_id: int, vocab_size: int,
            prompt_text: str, gen_len: int, steps: int, block_size: int,
            temperature: float, top_k: int, top_p: float,
            strategy: str, conf_threshold: float, min_per_step: int,
            presence_penalty: float, count_penalty: float, penalty_decay: float,
            penalize_prompt: bool):
    """Run one generation with a pre-loaded model and tokenizer.

    Lifted out of ``main`` so callers (e.g. test/sweep_inference.py) can keep
    the model resident in GPU memory and only vary the sampling knobs.
    Prints per-block partials and the final generation to stdout.
    """
    prompt_ids = tok.encode(prompt_text) if prompt_text else []
    print(f"prompt: {prompt_text!r}  ({len(prompt_ids)} tokens)")
    print(f"generating {gen_len} tokens in blocks of {block_size} x {steps} steps each")

    if prompt_ids:
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device="cuda")
        _, ctx_state = model.forward_fast(prompt_tensor, None, full_output=False)
    else:
        ctx_state = model.init_state()

    # Token-count history for presence + count penalties. Lives on GPU so
    # the per-step penalty broadcast inside denoise_block_fast is fast.
    # ChatRWKV's pattern is `occurrence: dict[int, float]`; we use a dense
    # vocab-sized tensor so the penalty subtraction is one fused op.
    token_count = torch.zeros(vocab_size, dtype=torch.float32, device="cuda")
    if penalize_prompt and prompt_ids:
        for tid in prompt_ids:
            token_count[tid] += 1.0
    use_penalty = (presence_penalty != 0.0) or (count_penalty != 0.0)

    committed: list[torch.Tensor] = []
    n_blocks = math.ceil(gen_len / block_size)
    eos_id = 0    # RWKV-world tokenizer's EOS
    eos_pos_in_concat = None
    for bi in range(n_blocks):
        # Build the per-vocab penalty tensor for this block from current
        # token_count. Recomputed each block so decay shows up.
        # ChatRWKV's penalty formula, one token at a time:
        #   for n in occurrence:
        #       out[n] -= (presence_penalty + occurrence[n] * count_penalty)
        # Vectorized over the whole vocab here:
        penalty = None
        if use_penalty:
            penalty = (
                presence_penalty * (token_count > 0).to(torch.float32)
                + count_penalty * token_count
            )

        blk = denoise_block_fast(model, ctx_state, block_size, mask_id,
                                 steps, temperature, top_k,
                                 top_p=top_p,
                                 penalty=penalty,
                                 strategy=strategy,
                                 conf_threshold=conf_threshold,
                                 min_per_step=min_per_step)
        committed.append(blk)
        _, ctx_state = model.forward_fast(blk, ctx_state, full_output=False)

        # ChatRWKV's update order: decay first, then count the new tokens.
        #   for x in occurrence: occurrence[x] *= penalty_decay
        #   occurrence[token] = occurrence.get(token, 0) + 1
        # Decay-then-add means the freshly emitted tokens start at full
        # weight (1.0), not pre-decayed.
        if use_penalty:
            if penalty_decay != 1.0:
                token_count.mul_(penalty_decay)
            token_count.scatter_add_(
                0, blk.long(), torch.ones_like(blk, dtype=torch.float32)
            )

        # Diffusion-style early stop: if the model emitted EOS anywhere in
        # this block, treat that as end-of-turn. The block's positions
        # AFTER the EOS are noise the sampler was forced to fill — drop
        # them. Without this, gen_len always materializes in full.
        eos_in_blk = (blk == eos_id).nonzero(as_tuple=True)[0]
        if eos_in_blk.numel() > 0:
            first_eos_local = int(eos_in_blk[0].item())
            eos_pos_in_concat = sum(b.numel() for b in committed[:-1]) + first_eos_local

        # Decode partial output. If EOS was just hit, truncate to it so the
        # noise the sampler had to fill after EOS doesn't pollute the
        # progress print (which is also what causes mid-utf8 decode errs).
        partial_ids = torch.cat(committed).tolist()
        if eos_pos_in_concat is not None:
            partial_ids = partial_ids[: eos_pos_in_concat]
        partial = _safe_decode(tok, partial_ids)
        print(f"--- after block {bi+1}/{n_blocks} ---")
        print(partial)

        if eos_pos_in_concat is not None:
            print(f"[stop] EOS detected at block {bi+1} pos {first_eos_local} -> truncating")
            break

    all_out = torch.cat(committed)
    if eos_pos_in_concat is not None:
        out_ids = all_out[: eos_pos_in_concat].tolist()
    else:
        out_ids = all_out[: gen_len].tolist()
    print("\n=== final generation ===")
    print(_safe_decode(tok, out_ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", default=None,
                    help="path to rwkv_vocab_v20230424.txt; default uses the bundled tokenizer/")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--gen_len", type=int, default=256)
    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0,
                    help="keep only the K highest-prob tokens per position; 0 = no cap")
    ap.add_argument("--top_p", type=float, default=1.0,
                    help="nucleus sampling: keep smallest token set whose cum prob > top_p; 1.0 = off")
    # ChatRWKV / RWKV-Gradio canonical penalties. Same semantics, same names.
    # See: https://github.com/BlinkDL/ChatRWKV/blob/.../rwkv_pip_package/src/rwkv/utils.py
    #   logit[v] -= presence_penalty * 1[count[v] > 0] + count_penalty * count[v]
    # After each step, count *= penalty_decay; then count[just_emitted] += 1.
    # We apply this between blocks (not between intra-block denoise steps) since
    # diffusion commits all B positions in parallel from the same conditioning.
    ap.add_argument("--presence_penalty", type=float, default=0.0,
                    help="subtract this from any logit whose token has appeared in history at all")
    ap.add_argument("--count_penalty", type=float, default=0.0,
                    help="subtract this * count(token) from its logit (compounds with repetition)")
    ap.add_argument("--penalty_decay", type=float, default=0.996,
                    help="multiply running token-count by this after each block; <1 lets old "
                         "penalties fade so the model can revisit topics later")
    ap.add_argument("--penalize_prompt", action="store_true",
                    help="seed the token-count with the prompt's tokens (default off: only "
                         "tokens the sampler emits contribute to the penalty)")
    # LLaDA-2.0 style decoding: threshold + low-confidence fallback. ``linear``
    # keeps the old fixed-pace schedule (commit floor(B*t/T) per step).
    ap.add_argument("--decode_strategy", choices=["threshold", "linear"], default="threshold")
    ap.add_argument("--conf_threshold", type=float, default=0.95,
                    help="(threshold) commit any masked position whose token prob > this in one shot")
    ap.add_argument("--min_per_step", type=int, default=0,
                    help="(threshold) fallback floor: if Phase-1 commits < this, take the top-N most-confident "
                         "instead. 0 -> auto = max(1, block_size // steps)")
    ap.add_argument("--n_layer", type=int, required=True)
    ap.add_argument("--n_embd", type=int, required=True)
    ap.add_argument("--head_size", type=int, default=64)
    ap.add_argument("--vocab_size", type=int, default=65536)
    ap.add_argument("--my_testing", default="x070")
    # LoRA ranks for w/a/v/g time-mix projections; must match the ckpt.
    # Defaults (0) -> n_embd-scaled heuristic (matches RWKV7-G1 small ckpts).
    # For RWKV7-G1f-7.2B specifically: 128 / 128 / 96 / 480.
    ap.add_argument("--d_decay_lora", type=int, default=0)
    ap.add_argument("--d_aaa_lora", type=int, default=0)
    ap.add_argument("--d_mv_lora", type=int, default=0)
    ap.add_argument("--d_gate_lora", type=int, default=0)
    # REPL: keep model resident, take prompts from stdin in a loop. Avoids
    # re-paying the ~30s cold-start cost per generation.
    ap.add_argument("--repl", action="store_true",
                    help="interactive mode: load model once, prompt-generate loop")
    args = ap.parse_args()

    tok = RWKVTokenizer(args.vocab) if args.vocab else RWKVTokenizer()
    mask_id = args.vocab_size - 1

    # Construct model with a minimal args namespace (only the fields RWKV.__init__ reads).
    model_args = SimpleNamespace(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        dim_att=args.n_embd,
        dim_ffn=int((args.n_embd * 3.5) // 32 * 32),
        head_size=args.head_size,
        vocab_size=args.vocab_size,
        ctx_len=4096,
        my_testing=args.my_testing,
        grad_cp=0,
        weight_decay=0.0,
        lr_init=0.0, lr_final=0.0, betas=(0.9, 0.99), adam_eps=1e-18,
        layerwise_lr=0, my_pile_stage=0, train_stage=0,
        diffusion_mode=0,
        d_decay_lora=args.d_decay_lora,
        d_aaa_lora=args.d_aaa_lora,
        d_mv_lora=args.d_mv_lora,
        d_gate_lora=args.d_gate_lora,
    )
    model = build_model(args.ckpt, model_args)

    if args.repl:
        print("\n[REPL] model loaded. Enter prompts one per line.")
        print("[REPL] Use \\n for newlines inside the prompt.")
        print("[REPL] Empty line or Ctrl-D quits. Reset DBG flag each time.\n")
        while True:
            try:
                line = input("> ")
            except EOFError:
                print()
                break
            if not line.strip():
                break
            # Allow `\n` -> real newline so chat templates work
            prompt_text = line.encode("utf-8").decode("unicode_escape")
            denoise_block_fast._dbg_done = False  # re-enable DBG print for new run
            run_one(model, tok, mask_id, args.vocab_size,
                    prompt_text, args.gen_len, args.steps, args.block_size,
                    args.temperature, args.top_k, args.top_p,
                    args.decode_strategy, args.conf_threshold, args.min_per_step,
                    args.presence_penalty, args.count_penalty, args.penalty_decay,
                    args.penalize_prompt)
            print()
    else:
        run_one(model, tok, mask_id, args.vocab_size,
                args.prompt, args.gen_len, args.steps, args.block_size,
                args.temperature, args.top_k, args.top_p,
                args.decode_strategy, args.conf_threshold, args.min_per_step,
                args.presence_penalty, args.count_penalty, args.penalty_decay,
                args.penalize_prompt)


if __name__ == "__main__":
    main()
