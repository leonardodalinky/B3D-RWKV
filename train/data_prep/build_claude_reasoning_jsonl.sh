#!/usr/bin/env bash
# End-to-end: HF angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k -> JSONL
# (RWKV-v7 G1x template, <think>...</think> preserved) -> binidx (.bin / .idx).
# Forwards extra args to convert_claude_reasoning_to_jsonl.py (e.g. --limit 100,
# --variant instruct).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
DATA_DIR="${DATA_DIR:-/anvil/projects/x-cis260045/RWKV/data/claude_reasoning_binidx}"
JSONL_PATH="$DATA_DIR/claude_reasoning.jsonl"
BINIDX_PREFIX="$DATA_DIR/claude_reasoning"
VOCAB="$REPO_ROOT/tokenizer/rwkv_vocab_v20230424.txt"

mkdir -p "$DATA_DIR"

echo "HF -> JSONL: $JSONL_PATH"
uv run --group data python "$HERE/convert_claude_reasoning_to_jsonl.py" \
    --output "$JSONL_PATH" "$@"
