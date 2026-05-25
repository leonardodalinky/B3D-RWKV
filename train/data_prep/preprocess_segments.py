"""Tokenize a segment-format JSONL into ``<prefix>_text_document.{bin,idx,lossable.bin}``.

Replaces the third-party json2binidx ``preprocess_data.py`` for our pipeline.
Differences from json2binidx:
- Accepts the role-aware segment JSONL emitted by convert_*_to_jsonl.py:
      {"segments": [{"text": "...", "lossable": bool}, ...]}
  Falls back to the legacy ``{"text": "..."}`` format (treats the whole text as
  one all-lossable segment).
- Emits an extra ``<prefix>_text_document.lossable.bin`` aligned with the .bin:
  one ``uint8`` per token, ``1`` if that token should contribute to loss, else
  ``0``. The trailing EOS appended by ``--append-eod`` is marked **not lossable**
  (we don't want the model trained to emit EOS at every assistant boundary; the
  real assistant content already ends naturally).

The .bin / .idx layout is byte-compatible with the json2binidx writer
(see train/src/binidx.py MMapIndexedDataset.Index.writer), so MyDataset can
read it without changes.

Usage:
    uv run python train/data_prep/preprocess_segments.py \\
        --input merged.jsonl --output-prefix /path/combined \\
        [--append-eod] [--vocab tokenizer/rwkv_vocab_v20230424.txt]
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

# Make the bundled tokenizer importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from tokenizer import RWKVTokenizer  # noqa: E402

# Mirror train/src/binidx.py so MyDataset.MMapIndexedDataset can mmap our output.
_HDR_MAGIC = b"MMIDIDX\x00\x00"
# json2binidx dtype codes; we always emit int32 (code 4) for parity with the
# existing binidxes you already have on disk. RWKV ids fit in uint16 too, but
# int32 keeps everything binary-compatible with the data already produced.
_DTYPE_CODE = 4
_DTYPE = np.int32
_DTYPE_SIZE = np.dtype(_DTYPE).itemsize


def write_idx(path: str, sizes: list, doc_idx: list):
    """Write a .idx that MMapIndexedDataset.Index can read."""
    with open(path, "wb") as f:
        f.write(_HDR_MAGIC)
        f.write(struct.pack("<Q", 1))  # version
        f.write(struct.pack("<B", _DTYPE_CODE))  # dtype code
        f.write(struct.pack("<Q", len(sizes)))  # n_docs
        f.write(struct.pack("<Q", len(doc_idx)))  # _doc_count
        # pointers: byte offset of doc i within the .bin
        pointers = []
        addr = 0
        for sz in sizes:
            pointers.append(addr)
            addr += int(sz) * _DTYPE_SIZE
        np.array(sizes, dtype=np.int32).tofile(f)
        np.array(pointers, dtype=np.int64).tofile(f)
        np.array(doc_idx, dtype=np.int64).tofile(f)


def iter_jsonl_lines(path: str):
    """Yield parsed JSON objects from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def row_to_token_streams(row, tokenizer, append_eod: bool):
    """Return (tokens: list[int], lossable: list[int]) for one JSONL row."""
    if "segments" in row:
        segments = row["segments"]
    elif "text" in row:
        # Backward-compat: legacy JSONL with a single text field. Treat the entire
        # text as one all-lossable segment (preserves old training behavior).
        segments = [{"text": row["text"], "lossable": True}]
    else:
        raise ValueError(f"row has neither 'segments' nor 'text': keys={list(row.keys())}")

    tokens: list = []
    lossable: list = []
    for seg in segments:
        seg_text = seg.get("text", "")
        if not seg_text:
            continue
        seg_tokens = tokenizer.encode(seg_text)
        tokens.extend(seg_tokens)
        bit = 1 if seg.get("lossable") else 0
        lossable.extend([bit] * len(seg_tokens))

    if append_eod:
        # Document-end marker token (id 0). Not lossable: the model shouldn't be
        # trained to emit EOS in the middle of an assistant turn, and the
        # genuine end-of-doc EOS is informational rather than part of the answer.
        tokens.append(0)
        lossable.append(0)

    assert len(tokens) == len(lossable), (len(tokens), len(lossable))
    return tokens, lossable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", required=True, help="input JSONL (segment-format or legacy text-only)"
    )
    ap.add_argument(
        "--output-prefix",
        required=True,
        help="output prefix; produces <prefix>_text_document.{bin,idx,lossable.bin}",
    )
    ap.add_argument(
        "--vocab",
        default=None,
        help="path to rwkv_vocab_v20230424.txt; defaults to bundled tokenizer/",
    )
    ap.add_argument(
        "--append-eod",
        action="store_true",
        help="append id 0 (EOS, non-lossable) at the end of every document",
    )
    ap.add_argument("--log-every", type=int, default=10000)
    args = ap.parse_args()

    out_prefix = args.output_prefix + "_text_document"
    bin_path = out_prefix + ".bin"
    idx_path = out_prefix + ".idx"
    lossable_path = out_prefix + ".lossable.bin"

    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    tokenizer = RWKVTokenizer(args.vocab) if args.vocab else RWKVTokenizer()

    sizes: list = []
    n_lossable_tokens = 0
    n_total_tokens = 0
    n_docs = 0
    with open(bin_path, "wb") as fbin, open(lossable_path, "wb") as floss:
        for row in iter_jsonl_lines(args.input):
            tokens, lossable = row_to_token_streams(row, tokenizer, args.append_eod)
            if not tokens:
                continue
            np.asarray(tokens, dtype=_DTYPE).tofile(fbin)
            np.asarray(lossable, dtype=np.uint8).tofile(floss)
            sizes.append(len(tokens))
            n_total_tokens += len(tokens)
            n_lossable_tokens += sum(lossable)
            n_docs += 1
            if n_docs % args.log_every == 0:
                print(
                    f"  ... {n_docs:,} docs, {n_total_tokens:,} tokens "
                    f"({100 * n_lossable_tokens / max(n_total_tokens, 1):.1f}% lossable)"
                )

    # json2binidx writes doc_idx = list(range(n_docs + 1)) for "single-document"
    # mode (one entry per doc + a sentinel). We mirror that.
    doc_idx = list(range(n_docs + 1))
    write_idx(idx_path, sizes, doc_idx)

    print()
    print(f"docs            = {n_docs:,}")
    print(f"tokens          = {n_total_tokens:,}")
    print(
        f"lossable tokens = {n_lossable_tokens:,}  ({100 * n_lossable_tokens / max(n_total_tokens, 1):.2f}%)"
    )
    print(f"-> {bin_path}")
    print(f"-> {idx_path}")
    print(f"-> {lossable_path}")


if __name__ == "__main__":
    main()
