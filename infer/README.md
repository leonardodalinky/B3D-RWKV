# Inference scripts

Wrappers around `infer/diffusion_sample.py` for evaluating trained DiffuRWKV
checkpoints. Activates the project venv, sets sane CUDA env vars, exposes the
common knobs as env vars so you can override without editing files.

## Quick start

```bash
# Default — uses rwkv-final.pth, generates 128 tokens with steps=32, temp=0.8
bash infer/run_inference.sh

# Pick a different checkpoint
CKPT=train/out/diff-L32-D4096-x070-blk32-ctx6144_fixEOF/rwkv-4.pth \
  bash infer/run_inference.sh

# Different prompt
PROMPT=$'User: What is the capital of France?\n\nAssistant:' \
  bash infer/run_inference.sh

# Force a thinking block
PROMPT=$'User: Solve x^2 - 5x + 6 = 0.\n\nAssistant: <think>' \
  bash infer/run_inference.sh

# Crank up creativity / break repetition loops
TEMP=1.0 TOP_K=100 GEN_LEN=256 \
  bash infer/run_inference.sh

# Argmax (deterministic, may collapse)
TEMP=0 \
  bash infer/run_inference.sh
```

## Files

| File | Purpose |
|---|---|
| `run_inference.sh` | Standard sampler invocation. Parameterized via env vars. |
| `probe_logits.sh`  | Diagnostic — feeds 32 MASK tokens to the model and dumps argmax predictions, no sampling. Use this when the sampler output looks broken to confirm whether the issue is in the model or the sampler. |

## Knobs (env vars on `run_inference.sh`)

| Var | Default | What it does |
|---|---|---|
| `CKPT` | `train/out/diff-L32-D4096-x070-blk32-ctx6144_fixEOF/rwkv-final.pth` | Model weights path |
| `PROMPT` | `User: Briefly explain the acronym RWKV.\n\nAssistant:` | Prompt text (use `$'...'` for `\n`) |
| `GEN_LEN` | 128 | Tokens to generate after prompt |
| `STEPS` | 32 | Denoise iterations per logical block (≤ block_size) |
| `BLOCK_SIZE` | 32 | MUST equal training BLOCK_SIZE |
| `TEMP` | 0.8 | Softmax temperature for sampling. 0 = argmax, ≥1.0 = creative |
| `TOP_K` | 50 | Limit candidates to top-K. 0 = unrestricted |
| `CUDA_VISIBLE_DEVICES` | 0 | Which GPU |

Architecture knobs (`N_LAYER`, `N_EMBD`, `HEAD_SIZE`, `D_*_LORA`) default to the
RWKV7-G1f-7.2B config that we currently train on. Change them if you swap to
a different ckpt size.

## Common patterns

### A/B compare two ckpts

```bash
for ep in 4 8 final; do
  echo "===== epoch $ep ====="
  CKPT=train/out/.../rwkv-$ep.pth bash infer/run_inference.sh
done
```

### Stress-test for repetition

If output keeps repeating, knobs in order of impact:

1. `TEMP=1.0` — break out of argmax loops
2. `TOP_K=100` — broader candidate pool
3. `GEN_LEN=64` — shorter output, less time to drift
4. `STEPS=16` — fewer denoise iterations (less commit-then-revisit)

### Verify model is healthy after training

```bash
# Should show diverse predictions across positions, NOT all 0s.
bash test/probe_logits.sh

# Then a real sample
bash infer/run_inference.sh
```

## Notes

- The sampler currently does **not** suppress EOS or MASK in logits (lines 153-154 of `infer/diffusion_sample.py` are commented out). If the trained model collapses to predicting 0 everywhere, uncomment to force it past that — but the proper fix is retraining with `--diff_pad_id 65534` (already the default).
- Inference uses `forward_fast` (RNN/state mode), not the chunked training kernel. Numerics should match training exactly because `infer/cuda/wkv7s.{cu,op.cpp}` was switched from `at::Half` to `at::BFloat16`.
- First run does ~30 sec of cold start (kernel JIT cache lookup + 14 GB ckpt load). Subsequent runs are similar — there's no easy way to avoid this without keeping a process resident; use `--repl` flag in `diffusion_sample.py` if you want a stay-alive REPL.
