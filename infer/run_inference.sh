#!/bin/bash
###############################################################################
# Inference for DiffuRWKV via iterative-denoising sampler.
#
# Reads model weights from the rwkv-N.pth files written by training.
# Override CKPT / PROMPT / GEN_LEN / STEPS / TEMP / TOP_K / SEED via env vars
# without editing the file:
#
#   CKPT=train/out/.../rwkv-4.pth bash infer/run_inference.sh
#   PROMPT='User: hi\n\nAssistant:' bash infer/run_inference.sh
#   STEPS=32 TEMP=0.7 bash infer/run_inference.sh
#
# Or just edit the defaults in the "EDIT ME" section below.
###############################################################################
set -euo pipefail

# Resolve repo root regardless of where the script is invoked from.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

###############################################################################
# Environment
###############################################################################
if [ ! -f "$REPO_DIR/.venv/bin/activate" ]; then
  echo "ERROR: $REPO_DIR/.venv not found. Run 'uv sync' first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$REPO_DIR/.venv/bin/activate"

# Keep JIT compile for one arch only — saves a lot of time on first run.
export TORCH_CUDA_ARCH_LIST="9.0"

# Single GPU is enough for 7.2B bf16 inference (~14 GB weights).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

###############################################################################
# EDIT ME — defaults for the standard 7.2B G1f checkpoint
###############################################################################
# Model architecture (must match training configuration / loaded ckpt)
N_LAYER="${N_LAYER:-32}"
N_EMBD="${N_EMBD:-4096}"
HEAD_SIZE="${HEAD_SIZE:-64}"
# 7.2B G1f-specific LoRA ranks (don't change unless you switch ckpts)
D_DECAY_LORA="${D_DECAY_LORA:-128}"
D_AAA_LORA="${D_AAA_LORA:-128}"
D_MV_LORA="${D_MV_LORA:-96}"
D_GATE_LORA="${D_GATE_LORA:-480}"

# Which checkpoint to load
CKPT="${CKPT:-/data/rsync/RWKV/DiffuRWKV/train/out/diff-L32-D4096-x070-blk32-ctx6144/rwkv-21.pth}"

# Sampling knobs
BLOCK_SIZE="${BLOCK_SIZE:-32}"   # MUST match training BLOCK_SIZE
GEN_LEN="${GEN_LEN:-2048}"        # how many tokens to generate
STEPS="${STEPS:-32}"             # denoise iterations per block (max 32 for blk_size=32)
TEMP="${TEMP:-0.85}"              # 0 = argmax; 0.7-1.0 typical
TOP_K="${TOP_K:-100}"             # 0 = no truncation; 20-100 typical
TOP_P="${TOP_P:-0.9}"            # nucleus sampling; 0.85-0.95 typical, 1.0 = off
# Canonical ChatRWKV / RWKV-Gradio penalty knobs. Same names, same defaults.
# - presence_penalty: pushed down once if a token has appeared at all
# - count_penalty:    pushed down N times if it has appeared N times
# - penalty_decay:    multiplied into the running counts each block
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0}"
COUNT_PENALTY="${COUNT_PENALTY:-0}"
PENALTY_DECAY="${PENALTY_DECAY:-1.0}"
PENALIZE_PROMPT="${PENALIZE_PROMPT:-0}"   # 1 = also penalize tokens from prompt

# Prompt — use $'...' so \n is interpreted as a real newline.
# Add a literal " <think>" at the end if you want to force the reasoning block.
PROMPT="${PROMPT:-$'User: How do I ready a guinea pig cage for it is new occupants?\n\nAssistant:<think>'}"

###############################################################################
# Run
###############################################################################
if [ ! -f "$CKPT" ]; then
  echo "ERROR: ckpt not found: $CKPT" >&2
  exit 1
fi

echo "[inference]"
echo "  CKPT      = $CKPT"
echo "  PROMPT    = ${PROMPT}"
echo "  GEN_LEN   = $GEN_LEN  STEPS = $STEPS  BLOCK = $BLOCK_SIZE"
echo "  TEMP      = $TEMP    TOP_K = $TOP_K"
echo

PENALIZE_PROMPT_FLAG=""
if [ "$PENALIZE_PROMPT" = "1" ]; then PENALIZE_PROMPT_FLAG="--penalize_prompt"; fi

python infer/diffusion_sample.py \
  --ckpt "$CKPT" \
  --n_layer "$N_LAYER" --n_embd "$N_EMBD" --head_size "$HEAD_SIZE" \
  --d_decay_lora "$D_DECAY_LORA" \
  --d_aaa_lora   "$D_AAA_LORA" \
  --d_mv_lora    "$D_MV_LORA" \
  --d_gate_lora  "$D_GATE_LORA" \
  --block_size "$BLOCK_SIZE" \
  --gen_len    "$GEN_LEN" \
  --steps      "$STEPS" \
  --temperature "$TEMP" \
  --top_k       "$TOP_K" \
  --top_p       "$TOP_P" \
  --presence_penalty "$PRESENCE_PENALTY" \
  --count_penalty    "$COUNT_PENALTY" \
  --penalty_decay    "$PENALTY_DECAY" \
  $PENALIZE_PROMPT_FLAG \
  --prompt "$PROMPT"
