#!/usr/bin/env bash
# Run preprocess_segments on an already-merged JSONL and write a binidx
# (.bin / .idx / .lossable.bin). Optionally also runs find_magic_prime
# --diffusion so the resulting MAGIC_PRIME / EXIT_TOKENS can be plugged
# straight into demo-training-run-diffusion.sh.
#
# Usage:
#   bash train/data_prep/merge_and_binidx.sh INPUT_JSONL OUTPUT_PREFIX
#
# Optional env vars:
#   CTX_LEN=3072 BLOCK_SIZE=32   also run find_magic_prime --diffusion at the end
#
# Example:
#   bash train/data_prep/merge_and_binidx.sh \
#       /anvil/projects/x-cis260045/RWKV/data/combined.jsonl \
#       /anvil/projects/x-cis260045/RWKV/data/combined_binidx/combined
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 INPUT_JSONL OUTPUT_PREFIX" >&2
    exit 1
fi

INPUT_JSONL="$1"
OUT_PREFIX="$2"

if [ ! -s "$INPUT_JSONL" ]; then
    echo "ERROR: missing or empty: $INPUT_JSONL" >&2
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
VOCAB="$REPO_ROOT/tokenizer/rwkv_vocab_v20230424.txt"

OUT_DIR="$(dirname "$OUT_PREFIX")"
mkdir -p "$OUT_DIR"

n=$(wc -l < "$INPUT_JSONL")
echo "input: $INPUT_JSONL  ($n lines)"
echo "JSONL -> binidx (+ lossable.bin) at prefix $OUT_PREFIX"

uv run --group data python "$HERE/preprocess_segments.py" \
    --input "$INPUT_JSONL" \
    --output-prefix "$OUT_PREFIX" \
    --vocab "$VOCAB" \
    --append-eod

if [ -n "${CTX_LEN:-}" ] && [ -n "${BLOCK_SIZE:-}" ]; then
    echo
    echo "===== find_magic_prime --diffusion (CTX_LEN=$CTX_LEN, BLOCK_SIZE=$BLOCK_SIZE) ====="
    uv run python "$REPO_ROOT/train/data_prep/find_magic_prime.py" --diffusion \
        --bin "${OUT_PREFIX}_text_document.bin" \
        --ctx_len "$CTX_LEN" --block_size "$BLOCK_SIZE"
else
    echo
    echo "done. Next, compute magic_prime, e.g.:"
    echo "  uv run python train/data_prep/find_magic_prime.py --diffusion \\"
    echo "      --bin ${OUT_PREFIX}_text_document.bin --ctx_len 3072 --block_size 32"
fi
