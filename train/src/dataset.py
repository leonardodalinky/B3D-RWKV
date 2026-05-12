########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset

def is_prime(n):
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

class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        self.vocab_size = args.vocab_size
        rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")

        self.data = MMapIndexedDataset(args.data_file)
        self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
        rank_zero_info(f"Data has {self.data_size} tokens.")

        # Optional per-token lossable mask, parallel to .bin (uint8, 1 byte/token).
        # Produced by train/data_prep/preprocess_segments.py; absence -> all real
        # tokens are lossable (legacy behavior, plus the runtime pad-mask below).
        lossable_path = args.data_file + "_text_document.lossable.bin"
        if not os.path.exists(lossable_path):
            # Fall back to the path WITHOUT the implicit _text_document suffix that
            # json2binidx adds, in case --data_file was set differently.
            lossable_path = args.data_file + ".lossable.bin"
        if os.path.exists(lossable_path):
            n_tokens_bin = sum(int(s) for s in self.data._index._sizes)
            n_loss_bytes = os.path.getsize(lossable_path)
            assert n_loss_bytes == n_tokens_bin, (
                f"{lossable_path} has {n_loss_bytes} bytes but .bin holds "
                f"{n_tokens_bin} tokens (1 byte/token expected)"
            )
            self.lossable_buf = np.memmap(lossable_path, mode="r", dtype=np.uint8)
            n_lossable = int(self.lossable_buf.sum())
            rank_zero_info(
                f"Loaded {lossable_path} -> {n_lossable:,}/{n_tokens_bin:,} tokens lossable "
                f"({100 * n_lossable / max(n_tokens_bin, 1):.2f}%)"
            )
        else:
            self.lossable_buf = None
            rank_zero_info(f"No .lossable.bin found at {lossable_path} -> all real tokens treated as lossable.")

        self.samples_per_epoch = args.epoch_steps * args.real_bsz
        assert self.samples_per_epoch == 40320
        rank_zero_info(f"########## train stage {args.train_stage} ##########")

        if getattr(args, "diffusion_mode", 0) == 1:
            # Diffusion mode samples ONE document per __getitem__ call (not flat-stream
            # chunks). Filter out documents that wouldn't fit a triplet sample, so we
            # don't pollute training with truncated conversations.
            block_size = args.diff_block_size
            n_blocks = args.ctx_len // (3 * block_size)
            raw_len = n_blocks * block_size
            max_doc_tokens = int(getattr(args, "diff_max_doc_tokens", 0)) or raw_len

            sizes = self.data._index._sizes  # numpy int32 array, one entry per binidx doc
            valid_mask = (sizes > 0) & (sizes <= max_doc_tokens)
            self.valid_doc_indices = np.nonzero(valid_mask)[0].astype(np.int64)
            n_valid = int(self.valid_doc_indices.shape[0])
            rank_zero_info(
                f"########## diffusion: {n_valid}/{len(sizes)} docs kept "
                f"(0 < tokens <= {max_doc_tokens}); raw_len={raw_len} ##########"
            )
            assert n_valid > 0, "no documents survived the length filter; relax --diff_max_doc_tokens"

            slot_count = n_valid
        else:
            slot_count = self.data_size // args.ctx_len

        assert is_prime(args.magic_prime)
        assert args.magic_prime % 3 == 2
        assert args.magic_prime / slot_count > 0.9 and args.magic_prime / slot_count <= 1, \
            f"magic_prime ({args.magic_prime}) / slot_count ({slot_count}) = " \
            f"{args.magic_prime / slot_count:.4f}, must be in (0.9, 1]. " \
            f"In diffusion mode slot_count = number of valid docs after filtering; " \
            f"recompute magic_prime with find_magic_prime.py --data_size {slot_count} --ctx_len 1."

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        # print(f"epoch {epoch} idx {idx} rank {rank}/{world_size}")

        ctx_len = args.ctx_len
        magic_prime = args.magic_prime

        ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank

        factor = (math.sqrt(5) - 1) / 2
        factor = int(magic_prime * factor)

        if getattr(args, "diffusion_mode", 0) == 1:
            block_size = args.diff_block_size
            n_blocks = ctx_len // (3 * block_size)
            raw_len = n_blocks * block_size

            # Magic-prime hashing on the *list of valid doc indices* (one doc per sample).
            slot_idx = (factor * ii * ii * ii) % magic_prime
            doc_idx = int(self.valid_doc_indices[slot_idx % len(self.valid_doc_indices)])
            doc_tokens = self.data.get(idx=doc_idx).astype(int)
            n_real = min(int(doc_tokens.shape[0]), raw_len)

            # Right-pad the doc with pad_id (default 65534, a dedicated dummy vocab slot;
            # NOT EOS=0 nor MASK=vocab_size-1) up to raw_len. The trailing blocks beyond
            # n_real are filled with pad_id and excluded from loss below.
            clean = torch.full((raw_len,), args.diff_pad_id, dtype=torch.long)
            clean[:n_real] = torch.tensor(doc_tokens[:n_real], dtype=torch.long)

            # Per-token lossable mask aligned with `clean`. Tokens past `n_real` are pad
            # and therefore non-lossable. Tokens within the doc default to lossable
            # unless an external .lossable.bin overrides them (role-aware mask from
            # preprocess_segments.py).
            doc_lossable = torch.zeros(raw_len, dtype=torch.bool)
            doc_lossable[:n_real] = True
            if self.lossable_buf is not None:
                ptr = int(self.data._index._pointers[doc_idx])
                tok_pos = ptr // self.data._index._dtype_size
                doc_size = int(self.data._index._sizes[doc_idx])
                loss_slice = self.lossable_buf[tok_pos : tok_pos + doc_size][:n_real]
                doc_lossable[:n_real] = torch.from_numpy(loss_slice.astype(np.bool_).copy())

            r_lo = float(args.diff_min_mask_ratio)
            r_hi = float(args.diff_max_mask_ratio)
            # Per-block independent mask ratio (LLaDA-style): every logical block samples
            # its own r ~ U[r_lo, r_hi], so a single forward pass sees a wide spectrum of
            # mask densities.
            r_per_block = r_lo + (r_hi - r_lo) * torch.rand(n_blocks, 1)
            # LLaDA full-mask trick: with prob p_full, override that block's r to 1.0
            # so EVERY lossable position in it becomes MASK in b1/b2. This matches the
            # inference distribution exactly -- at inference, each generation block
            # always starts as all-MASK and gets denoised in place. Without this, the
            # model rarely sees the all-MASK case (P(r ≈ 1) ≈ 0 under uniform), so its
            # behavior on the most extreme of mask ratios is under-trained. Default
            # 0.10 means ~6/64 blocks per sample are fully-masked, mirroring LLaDA-2.0.
            p_full = float(getattr(args, "diff_full_mask_prob", 0.10))
            if p_full > 0.0:
                full_mask_blk = torch.rand(n_blocks, 1) < p_full
                r_per_block = torch.where(full_mask_blk,
                                          torch.ones_like(r_per_block), r_per_block)
            mask_pos = torch.rand(n_blocks, block_size) < r_per_block
            mask_pos = mask_pos.view(raw_len)
            # Only allow mask sampling on lossable positions: User prompts, structural
            # markers ("Assistant: <think>"), and tail padding stay clean in b1/b2 so
            # the RNN state always carries accurate non-loss context. This matches the
            # inference-time distribution where masks only appear in the response.
            mask_pos = mask_pos & doc_lossable
            # Force-mask the document-ending EOS (token 0). EOS is exactly 1
            # position per doc (added by json2binidx --append-eod), so under
            # uniform random masking it would be supervised in only ~50% of
            # samples and contribute < 1/400 of the loss signal -- the model
            # then never learns "stop here". Hard-pinning it to True
            # guarantees every doc contributes one EOS-prediction example.
            #
            # NOTE: we do NOT AND with doc_lossable here. The trailing EOS
            # added by --append-eod can land outside what preprocess_segments.py
            # marks as lossable (segments mark the *response text*; the EOS
            # comes from binidx, not from the segment). Hard-pinning ensures
            # the supervision happens regardless. The y_view branch will
            # still produce target=0 because mask_blk is now True there.
            eos_id = 0
            is_eos = (clean == eos_id)
            if int(getattr(args, "diff_force_mask_eos", 1)) == 1:
                mask_pos = mask_pos | is_eos
            # Force-mask the pads INSIDE the EOS-containing block only. Two
            # reasons the granularity matters:
            # - Within the EOS block: clean = [..., real, EOS, pad, pad, ...].
            #   If the pads stay visible while EOS is masked, the model sees
            #   "MASK followed by visible pad" and shortcuts "this MASK must
            #   be EOS". That's a trivial cue that never appears at inference
            #   (which has no pad). So we MUST mask these pads to break the
            #   shortcut.
            # - Subsequent blocks that are entirely pad: no leakage cue inside
            #   them (no MASK + pad adjacency). Their loss is already 0 (the
            #   is_pad branch on y_view zeros it). Leaving them as pad in
            #   b1/b2/b3 is fine — masking would just churn compute.
            if int(getattr(args, "diff_force_mask_pad", 1)) == 1:
                clean_view = clean.view(n_blocks, block_size)
                # (n_blocks, 1) -- True for blocks containing EOS
                eos_block = is_eos.view(n_blocks, block_size).any(dim=1, keepdim=True)
                # (n_blocks, block_size) -- True for pad cells in those blocks
                pad_in_eos_block = (clean_view == args.diff_pad_id) & eos_block
                mask_pos = mask_pos | pad_in_eos_block.view(raw_len)
            masked = torch.where(mask_pos, torch.full_like(clean, args.diff_mask_id), clean)

            x = torch.full((ctx_len,), args.diff_pad_id, dtype=torch.long)
            y = torch.full((ctx_len,), -100, dtype=torch.long)

            clean_blk = clean.view(n_blocks, block_size)
            masked_blk = masked.view(n_blocks, block_size)
            mask_blk = mask_pos.view(n_blocks, block_size)

            x_view = x[: n_blocks * 3 * block_size].view(n_blocks, 3, block_size)
            x_view[:, 0] = masked_blk
            x_view[:, 1] = masked_blk
            x_view[:, 2] = clean_blk

            # Loss target only at b2 positions that were masked AND not pad.
            # Random masks are already constrained to lossable positions, so
            # User prompts / structural markers / tail padding are excluded.
            # However, force_mask_pad above adds EOS-block pad positions into
            # mask_pos so x has MASK there (kills the "MASK + visible pad"
            # leakage cue). Those positions must NOT contribute loss — otherwise
            # the model is supervised to predict pad_id (65534), which then
            # leaks into inference output. Strip them with the is_pad guard.
            is_pad = (clean == args.diff_pad_id).view(n_blocks, block_size)
            loss_blk = mask_blk & ~is_pad
            y_view = y[: n_blocks * 3 * block_size].view(n_blocks, 3, block_size)
            y_view[:, 1] = torch.where(loss_blk, clean_blk, torch.full_like(clean_blk, -100))

            return x, y

        # ---- Non-diffusion mode: standard pretraining flat-stream sampling. ----
        i = (factor * ii * ii * ii) % magic_prime * ctx_len
        req_len = ctx_len + 1
        dix = self.data.get(idx=0, offset=i, length=req_len).astype(int)
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
