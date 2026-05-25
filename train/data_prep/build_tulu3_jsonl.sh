#!/usr/bin/env bash
# End-to-end: HF tulu-3-sft-mixture -> JSONL (RWKV-v7 G1x template) -> binidx (.bin / .idx)
# Forwards extra args to convert_tulu3_to_jsonl.py (e.g. --limit 100 for smoke testing).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
DATA_DIR="$REPO_ROOT/train/data"
TOOL_DIR="$REPO_ROOT/third-party/json2binidx_tool"
JSONL_PATH="$DATA_DIR/tulu3.jsonl"
BINIDX_PREFIX="$DATA_DIR/tulu3"
VOCAB="$REPO_ROOT/tokenizer/rwkv_vocab_v20230424.txt"  # bundled, single source of truth

mkdir -p "$DATA_DIR"

echo "HF -> JSONL: $JSONL_PATH"
uv run --group data python "$HERE/convert_tulu3_to_jsonl.py" \
    --output "$JSONL_PATH" "$@"
