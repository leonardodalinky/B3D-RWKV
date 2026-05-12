#!/usr/bin/env python3
"""Find a magic_prime for MyDataset given a binidx file and ctx_len.

Two modes, matching MyDataset.__init__ in train/src/dataset.py:

* **Flat-stream mode** (default, used for plain pretraining): the constraint is
      0.9 < magic_prime / (data_size // ctx_len) <= 1
  We need just the total token count, which the .bin file size gives us.

* **Diffusion mode** (``--diffusion``): MyDataset reads ONE document per sample
  and filters out docs whose token count exceeds ``raw_len = n_blocks * block_size``
  (or ``--max_doc_tokens`` if set). The constraint becomes
      0.9 < magic_prime / n_valid_docs <= 1
  We read each doc's size from the .idx file to count the valid ones.

EXIT_TOKENS computation in --diffusion mode bakes in a subtle PL+trainer.py
quirk: ``trainer.global_step`` is the OPTIMIZER step, not the micro-batch step,
so the cosine-schedule's ``real_tokens`` counter is implicitly divided by
ACC_GRAD. Pass --acc_grad to get the right EXIT_TOKENS for one full data pass:

    EXIT_TOKENS_per_epoch = magic_prime * ctx_len / acc_grad

Examples:

    # Flat-stream pretraining
    python find_magic_prime.py --bin path/to/data_text_document.bin --ctx_len 4096

    # Diffusion training (per-doc filtering, 1 epoch, ACC_GRAD=4)
    python find_magic_prime.py --diffusion --bin path/to/tulu3_text_document.bin \\
        --ctx_len 6144 --block_size 32 --acc_grad 4

    # Same, but plan for 3 full passes through the data
    python find_magic_prime.py --diffusion --bin path/to/tulu3_text_document.bin \\
        --ctx_len 6144 --block_size 32 --acc_grad 4 --n_epochs 3
"""
import argparse
import os
import struct
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Prime search (shared)
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def find_magic_prime(slot: int) -> int:
    upper = slot
    lower = int(0.9 * slot) + 1
    p = upper
    if p % 3 != 2:
        p -= (p - 2) % 3
    while p >= lower:
        if p % 3 == 2 and is_prime(p):
            return p
        p -= 3
    raise RuntimeError(
        f"no prime with p%3==2 found in ({lower}, {upper}]; "
        f"slot count too small or pathological"
    )


# ---------------------------------------------------------------------------
# Flat-stream helpers
# ---------------------------------------------------------------------------

def infer_data_size_from_bin(bin_path: str, dtype_size: int = 2) -> int:
    """Default RWKV world tokenizer => uint16 => 2 bytes/token."""
    size_bytes = os.path.getsize(bin_path)
    assert size_bytes % dtype_size == 0, f"{bin_path} size {size_bytes} not divisible by {dtype_size}"
    return size_bytes // dtype_size


# ---------------------------------------------------------------------------
# Diffusion helpers (read per-doc sizes from .idx)
# ---------------------------------------------------------------------------

_HDR_MAGIC = b"MMIDIDX\x00\x00"  # matches train/src/binidx.py


def read_doc_sizes(idx_path: str) -> np.ndarray:
    """Parse the json2binidx-style .idx header and return the int32 array of doc sizes."""
    with open(idx_path, "rb") as f:
        magic = f.read(9)
        assert magic == _HDR_MAGIC, f"bad magic {magic!r} in {idx_path}"
        (version,) = struct.unpack("<Q", f.read(8))
        assert version == 1, f"unexpected idx version {version}"
        f.read(1)  # dtype code (we don't need it here)
        (n_docs,) = struct.unpack("<Q", f.read(8))
        f.read(8)  # _doc_count, unused
        sizes = np.frombuffer(f.read(n_docs * 4), dtype=np.int32, count=n_docs).copy()
    return sizes


def derive_idx_path(bin_path: str) -> str:
    """``foo_text_document.bin`` -> ``foo_text_document.idx`` (same dir, swap extension)."""
    if bin_path.endswith(".bin"):
        return bin_path[:-4] + ".idx"
    return bin_path + ".idx"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", type=str, default=None,
                    help="path to <prefix>_text_document.bin (token-count read from file size; "
                         "the matching .idx is read in --diffusion mode)")
    ap.add_argument("--data_size", type=int, default=None,
                    help="(flat-stream only) total tokens in the dataset, if you already know it")
    ap.add_argument("--ctx_len", type=int, required=True)
    ap.add_argument("--dtype_size", type=int, default=2,
                    help="bytes per token in the .bin file (uint16=2, uint32=4)")

    # Diffusion-mode toggles
    ap.add_argument("--diffusion", action="store_true",
                    help="diffusion training mode: count valid docs (size <= max_doc_tokens) "
                         "and size magic_prime against that count")
    ap.add_argument("--block_size", type=int, default=0,
                    help="(diffusion) tokens per logical block; required when --diffusion")
    ap.add_argument("--max_doc_tokens", type=int, default=0,
                    help="(diffusion) keep docs with 0 < size <= max_doc_tokens. "
                         "0 (default) -> auto = (ctx_len // (3*block_size)) * block_size = raw_len")
    # ACC_GRAD matters: trainer.py's real_tokens = global_step * ctx_len * real_bsz,
    # and PL's global_step is the OPTIMIZER step (incremented every ACC_GRAD micro-
    # batches). So if you accumulate, each "trainer token" represents ACC_GRAD real
    # tokens of forward work, and the EXIT_TOKENS needed to cover 1 dataset pass
    # is divided by ACC_GRAD.
    ap.add_argument("--acc_grad", type=int, default=1,
                    help="(diffusion) gradient_accumulation_steps used in training. "
                         "Required to compute correct EXIT_TOKENS — trainer.py's token "
                         "counter does NOT factor this in. Pass the same value as "
                         "$ACC_GRAD in demo-training-run-diffusion.sh.")
    ap.add_argument("--n_epochs", type=float, default=1.0,
                    help="(diffusion) print EXIT_TOKENS for this many full passes "
                         "through the dataset (default 1).")

    args = ap.parse_args()

    if args.diffusion:
        run_diffusion(args)
    else:
        run_flat_stream(args)


def run_flat_stream(args):
    if args.bin is None and args.data_size is None:
        print("ERROR: pass either --bin or --data_size", file=sys.stderr)
        sys.exit(1)
    data_size = args.data_size if args.data_size is not None else infer_data_size_from_bin(args.bin, args.dtype_size)

    slot = data_size // args.ctx_len
    if slot < 100:
        print(f"ERROR: only {slot} ctx_len slots available; dataset too small for ctx_len={args.ctx_len}",
              file=sys.stderr)
        sys.exit(1)

    p = find_magic_prime(slot)
    coverage = p / slot
    print(f"mode            = flat-stream")
    print(f"data_size       = {data_size}")
    print(f"ctx_len         = {args.ctx_len}")
    print(f"slot count      = {slot}")
    print(f"magic_prime     = {p}   (coverage = {coverage:.4f})")
    print(f"  --magic_prime {p} --my_exit_tokens {data_size}")


def run_diffusion(args):
    if args.bin is None:
        print("ERROR: --diffusion requires --bin (the script reads the matching .idx)", file=sys.stderr)
        sys.exit(1)
    if args.block_size <= 0:
        print("ERROR: --diffusion requires --block_size > 0", file=sys.stderr)
        sys.exit(1)

    n_blocks = args.ctx_len // (3 * args.block_size)
    raw_len = n_blocks * args.block_size
    if raw_len <= 0:
        print(f"ERROR: ctx_len ({args.ctx_len}) too small for one triplet (3 * {args.block_size}).",
              file=sys.stderr)
        sys.exit(1)
    max_doc_tokens = args.max_doc_tokens or raw_len

    idx_path = derive_idx_path(args.bin)
    sizes = read_doc_sizes(idx_path)
    n_total = int(sizes.shape[0])
    n_valid = int(((sizes > 0) & (sizes <= max_doc_tokens)).sum())
    if n_valid < 100:
        print(f"ERROR: only {n_valid} valid docs (out of {n_total}); "
              f"raise --max_doc_tokens or use a larger dataset", file=sys.stderr)
        sys.exit(1)

    p = find_magic_prime(n_valid)
    coverage = p / n_valid

    if args.acc_grad < 1:
        print(f"ERROR: --acc_grad must be >= 1 (got {args.acc_grad})", file=sys.stderr)
        sys.exit(1)

    # ----- EXIT_TOKENS derivation (this is the part the old script got wrong) -----
    #
    # trainer.py:    real_tokens = global_step * ctx_len * real_bsz
    #                cosine-schedule terminates when real_tokens >= my_exit_tokens
    #
    # PL semantics: trainer.global_step is the OPTIMIZER step (one increment per
    # ACC_GRAD micro-batches). MyDataset.__len__ = epoch_steps * micro_bsz, and
    # one "MyDataset epoch" feeds samples_per_epoch = epoch_steps * real_bsz =
    # 40320 samples (hardcoded in train.py:147-148). To cover all `magic_prime`
    # unique slots one time, the dataloader must yield magic_prime samples,
    # which is:
    #   total micro-batches  = magic_prime / real_bsz
    #   total optimizer steps = magic_prime / (real_bsz * ACC_GRAD)
    # Plugging into trainer's formula:
    #   real_tokens(1 pass) = (magic_prime / (real_bsz * ACC_GRAD)) * ctx_len * real_bsz
    #                       = magic_prime * ctx_len / ACC_GRAD
    #
    # Note real_bsz cancels — EXIT_TOKENS is independent of how many GPUs / what
    # micro_bsz you pick, only ACC_GRAD shifts it. ctx_len here is the FULL
    # triplet-padded context (e.g. 6144 for 32-block × 64-block-per-sample),
    # NOT the unique content per sample.
    one_pass_exit = (p * args.ctx_len + args.acc_grad - 1) // args.acc_grad  # ceil
    n_pass_exit = int(round(one_pass_exit * args.n_epochs))

    # Sanity: real content tokens (sum of doc sizes; informational only — this is
    # what you'd quote to a collaborator as "tokens trained on", but trainer.py
    # ignores it).
    total_real_tokens = int(sizes[(sizes > 0) & (sizes <= max_doc_tokens)].sum())
    # Forward-pass tokens actually shoved through the model in 1 pass (includes
    # triplet b1/b2/b3 duplication and tail pad). Useful for cost estimates.
    forward_tokens_1pass = p * args.ctx_len

    print(f"mode             = diffusion (per-doc, length filter)")
    print(f"ctx_len          = {args.ctx_len}  (full triplet-padded context)")
    print(f"block_size       = {args.block_size}  (n_blocks/sample = {n_blocks}, raw_len/sample = {raw_len})")
    print(f"max_doc_tokens   = {max_doc_tokens}{' (auto = raw_len)' if args.max_doc_tokens == 0 else ''}")
    print(f"acc_grad         = {args.acc_grad}")
    print(f"n_epochs         = {args.n_epochs}")
    print()
    print(f"docs total           = {n_total}")
    print(f"docs kept            = {n_valid}  ({n_valid / max(n_total, 1) * 100:.1f}%)")
    print(f"magic_prime          = {p}   (coverage = {coverage:.4f})")
    print()
    print("Token counts (informational):")
    print(f"  real content / pass   = {total_real_tokens:>16,}  (sum of kept doc sizes; what to quote externally)")
    print(f"  forward tokens / pass = {forward_tokens_1pass:>16,}  (real content × ~3 for triplet + pad; FLOPs scale with this)")
    print()
    print("EXIT_TOKENS (trainer.py cosine target):")
    print(f"  1 epoch              = magic_prime * ctx_len / acc_grad = {one_pass_exit:>16,}")
    print(f"  {args.n_epochs:g} epochs           = {n_pass_exit:>16,}")
    print()
    print("Plug into demo-training-run-diffusion.sh:")
    print(f"  MAGIC_PRIME=\"{p}\"")
    print(f"  EXIT_TOKENS=\"{n_pass_exit}\"   # = magic_prime*ctx_len/acc_grad * n_epochs")
    print(f"  ACC_GRAD=\"{args.acc_grad}\"          # IF YOU CHANGE ACC_GRAD, RECOMPUTE EXIT_TOKENS")
    print(f"  --diffusion_mode 1 --diff_block_size {args.block_size}"
          f"{f' --diff_max_doc_tokens {max_doc_tokens}' if args.max_doc_tokens else ''}")


if __name__ == "__main__":
    main()
