#!/usr/bin/env python3
"""Estimate the average lossable-supervised tokens per training sample.

Reads the .bin / .idx and the .lossable.bin to compute, for each doc:
    S_doc = doc_len × lossable_fraction(doc) × 0.55

Reports the dataset-wide mean S, plus the EXIT_TOKENS needed to reach a
target total lossable-supervision count.

Usage:
    python train/data_prep/estimate_lossable_per_sample.py \
      --bin /data/.../data_text_document \
      --target_lossable 2e9 \
      --ctx_len 6144 --epoch_tokens 247.7e6
"""
import argparse
import os
import sys

import numpy as np

# Allow running from anywhere — find the train/src package.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
from src.binidx import MMapIndexedDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="binidx prefix (no .bin/.idx)")
    ap.add_argument("--target_lossable", type=float, default=2e9,
                    help="target total lossable-supervised tokens to reach")
    ap.add_argument("--ctx_len", type=int, default=6144)
    ap.add_argument("--block_size", type=int, default=32)
    ap.add_argument("--real_bsz", type=int, default=128)
    ap.add_argument("--epoch_steps", type=int, default=315,
                    help="40320 // real_bsz; train.py overrides --epoch_steps to this")
    ap.add_argument("--avg_mask_ratio", type=float, default=0.55,
                    help="0.5 from U[0,1] + 0.10 from full-mask trick → 0.55")
    args = ap.parse_args()

    ds = MMapIndexedDataset(args.bin)
    sizes = ds._index._sizes  # numpy int32 per-doc lengths
    n_docs_total = len(sizes)

    # Filter docs by what dataset.py actually keeps in diffusion mode
    raw_len = (args.ctx_len // (3 * args.block_size)) * args.block_size
    valid_mask = (sizes > 0) & (sizes <= raw_len)
    valid_sizes = sizes[valid_mask].astype(np.int64)
    n_docs = len(valid_sizes)
    print(f"docs: {n_docs:,} valid / {n_docs_total:,} total (filter: 0 < len <= {raw_len})")
    print(f"avg doc len   = {valid_sizes.mean():.1f}  median = {np.median(valid_sizes):.0f}")

    # Try lossable.bin (two name conventions)
    bin_path = args.bin
    cands = [bin_path + "_text_document.lossable.bin",
             bin_path + ".lossable.bin"]
    lossable_path = next((p for p in cands if os.path.exists(p)), None)

    if lossable_path:
        loss_buf = np.memmap(lossable_path, mode="r", dtype=np.uint8)
        n_loss_total = int(loss_buf.sum())
        n_tok_total = int(sizes.sum())
        L_global = n_loss_total / max(n_tok_total, 1)
        print(f"lossable.bin: {n_loss_total:,}/{n_tok_total:,} tokens lossable "
              f"({L_global * 100:.2f}%)")

        # Per-doc lossable count → per-doc S
        S_per_doc = np.zeros(n_docs, dtype=np.float64)
        ptrs = ds._index._pointers
        dtype_size = ds._index._dtype_size
        valid_idx = np.nonzero(valid_mask)[0]
        for i, di in enumerate(valid_idx):
            tok_pos = int(ptrs[di]) // dtype_size
            n = int(sizes[di])
            S_per_doc[i] = loss_buf[tok_pos:tok_pos + n].sum()
        avg_doc_lossable = S_per_doc.mean()
        S = avg_doc_lossable * args.avg_mask_ratio
        print(f"avg lossable / doc      = {avg_doc_lossable:.1f}")
        print(f"avg supervised / sample = {S:.1f}  (× mask_ratio={args.avg_mask_ratio})")
    else:
        print(f"no lossable.bin found; assuming all real tokens lossable (L=1.0)")
        S = float(valid_sizes.mean()) * args.avg_mask_ratio
        print(f"avg supervised / sample = {S:.1f}")

    # ---- Conversion ----
    target = float(args.target_lossable)
    samples_needed = target / S
    real_tokens = samples_needed * args.ctx_len
    samples_per_epoch = args.epoch_steps * args.real_bsz
    epoch_real_tokens = samples_per_epoch * args.ctx_len
    epochs = real_tokens / epoch_real_tokens

    print()
    print(f"target lossable supervisions: {target:.2e}")
    print(f"  samples needed            : {samples_needed:,.0f}")
    print(f"  EXIT_TOKENS (real)        : {real_tokens:.3e}   "
          f"({real_tokens / 1e9:.1f}B)")
    print(f"  epochs                    : {epochs:.1f}  "
          f"(epoch_steps={args.epoch_steps}, samples/epoch={samples_per_epoch:,})")


if __name__ == "__main__":
    main()
