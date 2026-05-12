#!/usr/bin/env bash
# End-to-end: HF Jackrong/GLM-5.1-Reasoning-1M-Cleaned -> JSONL (RWKV-v7 G1x template,
# <think>...</think> preserved) -> binidx (.bin / .idx).
# Forwards extra args to convert_glm_reasoning_to_jsonl.py (e.g. --limit 100).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
DATA_DIR="${DATA_DIR:-/anvil/projects/x-cis260045/RWKV/data/glm_reasoning_binidx}"
TOOL_DIR="$REPO_ROOT/third-party/json2binidx_tool"
JSONL_PATH="$DATA_DIR/glm_reasoning.jsonl"
BINIDX_PREFIX="$DATA_DIR/glm_reasoning"
VOCAB="$REPO_ROOT/tokenizer/rwkv_vocab_v20230424.txt"  # bundled, single source of truth

mkdir -p "$DATA_DIR"

echo "[1/2] HF -> JSONL: $JSONL_PATH"
uv run --group data python "$HERE/convert_glm_reasoning_to_jsonl.py" \
    --output "$JSONL_PATH" "$@"

echo "[2/2] JSONL -> binidx (+ lossable.bin) at prefix $BINIDX_PREFIX"
uv run --group data python "$HERE/preprocess_segments.py" \
    --input "$JSONL_PATH" \
    --output-prefix "$BINIDX_PREFIX" \
    --vocab "$VOCAB" \
    --append-eod

echo "done. Train with --data_file ${BINIDX_PREFIX}_text_document --vocab_size 65536"
