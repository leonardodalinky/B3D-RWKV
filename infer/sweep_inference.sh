#!/bin/bash
###############################################################################
# Wrapper for infer/sweep_inference.py — loads the model ONCE and runs a battery
# of sampling presets on one prompt, writing a single markdown file for visual
# side-by-side review. No per-preset cold-start cost.
#
# Defaults match run_inference.sh. Override with env vars:
#
#   CKPT=train/out/.../rwkv-30.pth bash infer/sweep_inference.sh
#   PROMPT='User: hi\n\nAssistant:' bash infer/sweep_inference.sh
#   PRESETS=greedy,heavy_penalty bash infer/sweep_inference.sh
#   OUT_FILE=infer/sweep_30.md bash infer/sweep_inference.sh
#
# Available presets: greedy, natural, focused_no_penalty, chatrwkv_canonical,
# current_default, heavy_penalty, mild_penalty_low_temp, linear_decode,
# strict_threshold, loose_threshold
###############################################################################
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ ! -f "$REPO_DIR/.venv/bin/activate" ]; then
  echo "ERROR: $REPO_DIR/.venv not found. Run 'uv sync' first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$REPO_DIR/.venv/bin/activate"

export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python infer/sweep_inference.py "$@"
