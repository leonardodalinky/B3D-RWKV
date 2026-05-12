# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

DiffuRWKV adds a discrete-diffusion (dLLM-style) infilling training mode on top of the
RWKV-v7 RNN. The key idea: RWKV is strictly causal, so you cannot do block-internal
bidirectional attention. Instead, each logical block is laid out three times in a row:

    [b1_masked] [b2_masked == b1] [b3_clean]   per logical block, repeated N times per sample

`b2` is the only block that contributes loss; because b1 sits in front of b2 in the RNN
time order, b2's hidden state has already absorbed every unmasked token in b1, giving
pseudo-bidirectional access. `b3` (clean) refreshes the RNN state with ground truth so the
next logical block trains in parallel without compounding errors. The whole sample (3N
blocks) goes through one forward pass — RWKV is linear-time so the 3× sequence length is
fine.

## Repo layout

- [train/](train/) — **modified training code** (this is where dev happens)
  - [train/src/dataset.py](train/src/dataset.py) — `__getitem__` has a `diffusion_mode`
    branch that constructs the triplet layout
  - [train/src/model.py](train/src/model.py) — `training_step` has a diffusion branch
    that uses `F.cross_entropy(..., ignore_index=-100)` instead of the L2-wrap CE CUDA
    kernel (which doesn't support `ignore_index`)
  - [train/train.py](train/train.py) — `--diffusion_mode 1` and friends; sets
    `args.diff_mask_id = args.vocab_size - 1` (reuses last dummy slot)
  - [train/cuda/](train/cuda/) — RWKV-v7 CUDA kernels, JIT-compiled via
    `torch.utils.cpp_extension.load` at model import time; **needs nvcc on PATH**
  - [train/data_prep/](train/data_prep/) — our HF dataset → JSONL conversion scripts
- [RWKV-v7/](RWKV-v7/) — upstream reference implementation and inference demos. **Do not
  modify** unless the user explicitly asks. `train/` was forked from
  `RWKV-v7/train_temp/`. [RWKV-v7/RWKV7-G1x-templates.txt](RWKV-v7/RWKV7-G1x-templates.txt)
  defines the chat prompt format used by tulu-3 conversion.
- [third-party/json2binidx_tool/](third-party/json2binidx_tool/) — vendored JSONL → binidx
  tool, used as-is. `tools/preprocess_data.py` is the entry point; output prefix gets
  `_text_document` appended automatically.
- [tmp_*.slurm](.) — Anvil cluster batch scripts (account `cis260045-{ai,gpu}`,
  partitions `ai` / `gpu` / `gpu-debug`).

## Architecture details that matter

- **No position embeddings.** RWKV-v7 has no absolute / RoPE positions; the only
  positional signal is the per-layer 1-token time-shift (`xx = time_shift(x) - x`,
  [train/src/model.py:567](train/src/model.py#L567)). This is why the triplet repetition
  works without any "position_ids" plumbing.
- **Loss shift convention.** Standard pretraining: dataset returns `x = dix[:-1], y = dix[1:]`
  ([train/src/dataset.py](train/src/dataset.py)); model output at index i is supervised
  against y[i] = next token. Diffusion mode: dataset returns `x` and `y` of the **same
  length, no shift** — model output at index i predicts y[i] directly (target is the
  clean original token at masked positions, -100 elsewhere).
- **Mask token.** `mask_id = vocab_size - 1` (= 65535 for the model's padded vocab of
  65536). The RWKV-world vocab file has 65529 real tokens (ids 1..65529) plus implicit
  EOS at id 0 → 65530 used slots. The model embedding is padded to 65536 for
  power-of-two friendliness, leaving ids 65530..65535 unused — we reuse id 65535 for
  MASK. **Do not extend vocab.** (Heads-up: upstream tokenizer hardcodes the stale
  value `vocab_size = 65525`; our [tokenizer/](tokenizer/) module derives it from the
  data instead, returning 65530.)
- **Tail pad.** Token 0 (EOS) is used as tail padding when `3 * n_blocks * block_size <
  ctx_len`. Loss is masked off there. Do **not** use the MASK token to pad — it would
  pollute its semantics.
- **`RWKV_HEAD_L2WRAP_CE_CHUNK` env var.** Default `'0'` (set in
  [train/src/model.py:33](train/src/model.py#L33)). When `> 0`, `forward()` returns
  hidden states (no head) and a chunked CE CUDA kernel computes loss directly. The
  diffusion branch handles both code paths but always applies `self.head` explicitly.

## Common commands

```bash
# Env setup
uv sync                       # training deps
uv sync --group data          # adds datasets/ftfy/tokenizers etc. for data prep

# Data pipeline (HF tulu-3 → JSONL → binidx)
bash train/data_prep/build_tulu3_binidx.sh              # full
bash train/data_prep/build_tulu3_binidx.sh --limit 100  # smoke

# After conversion, train uses --data_file <prefix>_text_document (json2binidx
# auto-appends _text_document to --output-prefix).

# SLURM submission (Anvil)
sbatch tmp_build_tulu3.slurm     # data prep job
squeue -u $USER                  # status
squeue -j <id> --start           # ETA
```

## Things that bite

- Editing `train/train.py` requires the diffusion-mode block to come **after**
  `args.vocab_size = train_data.vocab_size` (line ~208) but **before** `model = RWKV(args)`.
- `os.environ["RWKV_HEAD_L2WRAP_CE_CHUNK"]` is read at `model.py` import time, not at
  call time. To override it from a launcher, set the env var before any
  `from src.model import ...`.
- Dataset's `__init__` asserts `magic_prime % 3 == 2`, `is_prime(magic_prime)`, and
  `0.9 < magic_prime / (data_size // ctx_len) <= 1`. When picking `magic_prime` for a
  new dataset/ctx_len combo, all three constraints must be satisfied or
  `MyDataset(...)` will assertion-fail at import.
- `pytorch-lightning >= 1.9.5` in `pyproject.toml` but RWKV's training loop is written
  against the 1.9 API; some 2.x renames may break things (e.g.
  `Trainer.add_argparse_args`, `replace_sampler_ddp`). If LightningCLI complains, pin
  to 1.9.5 first.
- The CUDA kernels under [train/cuda/](train/cuda/) are JIT-compiled on first model import.
  This needs `nvcc` on PATH and a matching `gcc` (Anvil: `module load modtree/gpu`).
  Failures here surface as a `subprocess.CalledProcessError` deep inside
  `torch.utils.cpp_extension.load`.
