<div align="center">
  <!-- <img src="./assets/logo.png" alt="Logo" width="400"> -->
  <h1 align="center">Triplet-Block Diffusion RWKV
 </h1>
</div>

<div align="center">
    <!-- <a href="TODO"><img src="https://img.shields.io/badge/Project Page-blue?style=for-the-badge&color=1a1a2e&logo=homepage&logoColor=orange" alt="Project Page"></a> -->
    <a href="https://arxiv.org/abs/TODO"><img src="https://img.shields.io/badge/arXiv-<arxiv.id>-brightred?color=B31B1B&logo=arXiv&style=for-the-badge" alt="ArXiv"></a>
    <!-- <br/> -->
    <a href="https://huggingface.co/spaces/TODO"><img src="https://img.shields.io/badge/Huggingface-555555?logo=huggingface&style=for-the-badge" alt="Hugging Face"></a>
    <!-- <br/> -->
    <a href="https://pre-commit.com/"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&style=for-the-badge" alt="pre-commit">
    </a>
</div>

## 🔥 News

- **[2026, May]**: TODO

## Training

Discrete-diffusion (dLLM-style) infilling on top of RWKV-v7. Each logical block of
`block_size` tokens is laid out three times in a row inside a sample:

    [b1_masked] [b2_masked == b1] [b3_clean]

Loss is computed only on `b2`'s originally-masked positions. Because b1 sits in front
of b2 in the RNN time order, b2's hidden state has already absorbed every unmasked
token in b1, giving block-internal pseudo-bidirectional access. b3 (clean) refreshes
the RNN state with ground truth so the next logical block trains in parallel without
compounding errors.

### 1. Setup

```bash
uv sync                  # training deps (torch cu128, deepspeed, pytorch-lightning, ...)
uv sync --group data     # additionally for data prep (datasets, ftfy, tokenizers, ...)
```

GPU node prerequisites for actually running training:
- `nvcc` on `PATH` and a matching `gcc`.
- The CUDA kernels under [`train/cuda/`](train/cuda/) are JIT-compiled at first model import.

### 2. Build the training data

Two stages: HF → JSONL formatted with the RWKV-v7 G1x chat template, then JSONL → binidx.

```bash
# smoke test with 100 conversations
bash train/data_prep/build_tulu3_jsonl.sh --limit 100

# end-to-end (writes to train/data/ by default)
bash train/data_prep/build_tulu3_jsonl.sh
bash train/data_prep/build_glm_reasoning_jsonl.sh
bash train/data_prep/build_claude_reasoning_jsonl.sh

# Concat these three datasets
cat train/data/* > train/data/combined.jsonl

# Convert to binidx (multi-threaded, CPU-only)
bash train/data_prep/merge_and_binidx.sh data/combined.jsonl data/combined_binidx/combined
```

Output: `train/data/concatenated.{bin,idx}`. The `_text_document` suffix is
appended automatically by the json2binidx tool — train with
`--data_file train/data/concatenated` (no extension).

### 3. Pick `magic_prime`

`MyDataset.__init__` enforces three constraints (see [`train/src/dataset.py`](train/src/dataset.py)):

1. `is_prime(magic_prime)`
2. `magic_prime % 3 == 2`
3. `0.9 < magic_prime / slot_count <= 1`

In **diffusion mode**, `slot_count = number of valid docs after length filtering`
(docs longer than `raw_len = (ctx_len // (3*block_size)) * block_size` are skipped at
runtime — the JSONL/binidx are NOT mutated). In flat-stream mode it's `data_size // ctx_len`.

The helper handles both:

```bash
# Diffusion mode — also reads the .idx for per-doc lengths and prints recommended
# --magic_prime / --my_exit_tokens / CLI snippet ready to paste into the launcher.
uv run python train/data_prep/find_magic_prime.py --diffusion \
    --bin <DATA_DIR>/tulu3_text_document.bin \
    --ctx_len 3072 --block_size 32

# Flat-stream (non-diffusion) pretraining
uv run python train/data_prep/find_magic_prime.py \
    --bin <DATA_DIR>/tulu3_text_document.bin --ctx_len 4096
```

**Re-run any time `ctx_len`, `block_size`, `--diff_max_doc_tokens`, or the dataset
itself changes** — `slot_count` shifts and the prime needs to track it.

### 4. Launch training

The reference launcher is [`train/demo-training-run-diffusion.sh`](train/demo-training-run-diffusion.sh).
Open it, plug in `MAGIC_PRIME` and `EXIT_TOKENS` from step 3 (the script refuses to
run with the placeholder zeros), then:

```bash
bash train/demo-training-run-diffusion.sh
```

Diffusion-mode CLI flags added on top of the upstream `train.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--diffusion_mode` | `0` | Set to `1` to switch from standard LM to triplet-diffusion training. |
| `--diff_block_size` | `32` | Tokens per logical block. |
| `--diff_min_mask_ratio` | `0.0` | Lower bound for the per-sample mask ratio `r ~ Uniform(min, max)`. |
| `--diff_max_mask_ratio` | `1.0` | Upper bound for `r`. |
| `--diff_pad_id` | `65534` | Token used to pad the tail to `ctx_len` (EOS). **Must differ from MASK.** |

Constraints worth knowing:
- `ctx_len >= 3 * diff_block_size` (asserted at startup). Pick `ctx_len` divisible by
  `3 * diff_block_size` to avoid wasted tail padding (e.g. `3072 = 3 * 32 * 32`).
- `--vocab_size 65536` matches BlinkDL's published RWKV-v7 world ckpts. Diffusion
  mode reuses id `vocab_size - 1 = 65535` as MASK (one of the unused dummy slots
  past the tokenizer's real vocabulary); **vocab is not extended**, so any standard
  RWKV-v7 world ckpt loads as-is via `--load_model`.
- The fast L2-wrap CE CUDA kernel does not support `ignore_index`; diffusion mode
  falls back to `F.cross_entropy(..., ignore_index=-100)` (somewhat slower / more
  VRAM, but correct). Toggle is automatic.


## Inference and Evaluation

Please refer to the [inference README](infer/README.md) for details on running the model in inference mode and [serving README](infer/serve/README.md) for details on serving the model.

Please also refer to the [evaluation README](eval/README.md) for details on evaluating the model's performance.
