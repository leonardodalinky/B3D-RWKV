#!/bin/bash
###############################################################################
# Diffusion (dLLM-style) training launcher for RWKV-v7.
#
# Layout per logical block (see CLAUDE.md):
#   [b1_masked] [b2_masked == b1] [b3_clean]   repeated N times per sample
# Only b2's masked positions contribute loss; b3 refreshes the RNN state for
# the next logical block.
#
# Before running:
#   1) Build the binidx with train/data_prep/build_tulu3_binidx.sh.
#   2) Compute MAGIC_PRIME with train/data_prep/find_magic_prime.py:
#        python train/data_prep/find_magic_prime.py \
#          --bin <DATA_PREFIX>.bin --ctx_len $CTX_LEN
#      Plug the printed value (and matching --my_exit_tokens) below.
###############################################################################

MODEL_TYPE="x070"
# 7.2B = 32L 4096D (per https://www.rwkv.cn/tutorials/advanced/Fine-Tune/RWKV-PEFT/State-Tuning).
# These three numbers MUST match the loaded ckpt or load_state_dict will throw size mismatches.
N_LAYER="32"
N_EMBD="4096"

# ---- Diffusion knobs ----
CTX_LEN="6144"           #
BLOCK_SIZE="32"          # tokens per logical block
MIN_R="0.0"
MAX_R="1.0"
PAD_ID="65534"           # Dedicated pad slot (penultimate dummy row in the
                         # 65536-padded vocab; MASK is 65535, EOS is 0). Pad
                         # MUST differ from EOS so dataset.py can drop pad
                         # from loss while still supervising real EOS at
                         # document boundaries — and MUST differ from MASK
                         # so it doesn't pollute MASK semantics.

# ---- Dataset / scheduling ----
DATA_FILE=""   # binidx prefix (no .bin/.idx)
# MAGIC_PRIME="0"          # !!! REPLACE with output of find_magic_prime.py !!!
# EXIT_TOKENS="0"          # !!! REPLACE with the data_size printed by that script !!!

VOCAB_SIZE="65536"

PROJ_DIR="${PROJ_DIR:-out/diff-L${N_LAYER}-D${N_EMBD}-${MODEL_TYPE}-blk${BLOCK_SIZE}-ctx${CTX_LEN}_Math}"
mkdir -p "$PROJ_DIR"

# ---- Optimizer / batch ----
# 8 x H100 80GB. micro_bsz=16 + ACC_GRAD=4 -> effective batch 16×4×8=512.
# Memory peak is set by ONE forward+backward of micro_bsz, so picking 16
# leaves the largest headroom for wkv kernel scratchpad (which scales as
# ~0.375 GB/B). ACC_GRAD=4 amortizes the CPU-Adam fixed overhead 4× and
# matches the effective batch we'd want at LR=6e-5 sqrt-scaling.
M_BSZ="${M_BSZ:-4}"
# LR sqrt-scaled from canonical 3e-5 @ effective_bsz=128 to effective_bsz=512:
#   3e-5 × sqrt(512/128) = 3e-5 × 2 = 6e-5
LR_INIT="${LR_INIT:-1e-5}"
LR_FINAL="${LR_FINAL:-1e-6}"
GRAD_CP="${GRAD_CP:-1}"
EPOCH_SAVE="${EPOCH_SAVE:-1}"
STRATEGY="${STRATEGY:-deepspeed_stage_2}"
# epoch_steps with M_BSZ=16 = 5040/16 = 315 dataloader yields per epoch.
# With ACC_GRAD=4 that's ~78 optimizer steps per epoch. Warmup of 100 steps
# ≈ 1.3 epochs ramp-up, plenty for SFT continuation off rwkv-19.
WARMUP_STEPS="${WARMUP_STEPS:-300}"
# Gradient accumulation amplifies effective batch w/o more memory: each
# optimizer step processes ACC_GRAD micro-batches before stepping. Useful
# when a large micro_bsz OOMs but you still want big effective batch
# (and to amortize stage_2_offload's CPU-Adam overhead).
ACC_GRAD="${ACC_GRAD:-4}"
# Optimizer choice. Switch to adafactor if you want to drop _offload but Adam OOMs.
#   adam       : 12 B/param fp32 Adam state. With offload it lives on CPU.
#   adafactor  : 8 B/param (factorized v); GPU-only; ~3.6 GB/GPU savings vs Adam.
#                LR semantics same as Adam (we set scale_parameter=False).
#   8bit       : ~3 B/param bnb AdamW8bit; numerically unstable here, last resort.
OPTIM="${OPTIM:-adam}"

N_NODE=1
GPU_PER_NODE=8

if [ "$MAGIC_PRIME" = "0" ] || [ "$EXIT_TOKENS" = "0" ]; then
  echo "ERROR: set MAGIC_PRIME and EXIT_TOKENS first (see find_magic_prime.py)." >&2
  exit 1
fi

# Activate your environment
source .venv/bin/activate

# H100 only -> sm_90. Skips compiling for sm_52..sm_87 (saves ~90% JIT time).
export TORCH_CUDA_ARCH_LIST="9.0"

# Reclaim "reserved but unallocated" memory; fixes the optimizer-init OOM at
# stage_2 by avoiding the contiguous big-block requirement for grad partitions.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd train

python train.py \
  --load_model "YOUR/PATH/OF/TRAINED/MODEL" \
  --wandb "" \
  --proj_dir "$PROJ_DIR" \
  --my_testing "$MODEL_TYPE" \
  --ctx_len "$CTX_LEN" \
  --train_stage 3 \
  --epoch_count 10 \
  --epoch_begin 0 \
  --epoch_steps 8000 \
  --data_file "$DATA_FILE" \
  --data_type "binidx" \
  --vocab_size "$VOCAB_SIZE" \
  --my_exit_tokens "$EXIT_TOKENS" \
  --magic_prime "$MAGIC_PRIME" \
  --num_nodes "$N_NODE" \
  --micro_bsz "$M_BSZ" \
  --n_layer "$N_LAYER" \
  --n_embd "$N_EMBD" \
  --lr_init "$LR_INIT" \
  --lr_final "$LR_FINAL" \
  --warmup_steps "$WARMUP_STEPS" \
  --accumulate_grad_batches "$ACC_GRAD" \
  --optim "$OPTIM" \
  --beta1 0.9 --beta2 0.99 --adam_eps 1e-6 \
  --weight_decay 0.001 \
  --epoch_save "$EPOCH_SAVE" \
  --accelerator gpu --devices "$GPU_PER_NODE" \
  --precision bf16 \
  --strategy "$STRATEGY" \
  --grad_cp "$GRAD_CP" \
  --enable_progress_bar True \
  --diffusion_mode 1 \
  --diff_block_size "$BLOCK_SIZE" \
  --diff_min_mask_ratio "$MIN_R" \
  --diff_max_mask_ratio "$MAX_R" \
  --diff_pad_id "$PAD_ID" \
  --diff_conf_lambda 0.5 \
  --d_decay_lora 128 \
  --d_aaa_lora 128 \
  --d_mv_lora 96 \
  --d_gate_lora 480
