########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging

logging.basicConfig(level=logging.INFO)

import os

print(
    f"[mem-config] PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}"
)
print(
    f"[mem-config] CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG', '<unset>')}"
)


if __name__ == "__main__":
    from argparse import ArgumentParser

    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities import rank_zero_info, rank_zero_only

    rank_zero_info("########## work in progress ##########")

    parser = ArgumentParser()

    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument(
        "--wandb", default="", type=str
    )  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--random_seed", default="-1", type=int)

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str)
    parser.add_argument(
        "--vocab_size", default=0, type=int
    )  # vocab_size = 0 means auto (for char-level LM and .txt data)

    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument(
        "--epoch_steps", default=1000, type=int
    )  # a mini "epoch" has [epoch_steps] steps
    parser.add_argument(
        "--epoch_count", default=500, type=int
    )  # train for this many "epochs". will continue afterwards with lr = lr_final
    parser.add_argument(
        "--epoch_begin", default=0, type=int
    )  # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument(
        "--epoch_save", default=5, type=int
    )  # save the model every [epoch_save] "epochs"

    parser.add_argument(
        "--micro_bsz", default=12, type=int
    )  # micro batch size (batch size per GPU)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)

    parser.add_argument(
        "--lr_init", default=6e-4, type=float
    )  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=-1, type=int)  # try 10 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)
    parser.add_argument("--adam_eps", default=1e-18, type=float)
    parser.add_argument(
        "--grad_cp", default=0, type=int
    )  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--weight_decay", default=0, type=float)  # try 0.1
    parser.add_argument(
        "--grad_clip", default=1.0, type=float
    )  # reduce it to 0.7 / 0.5 / 0.3 / 0.2 for problematic samples

    parser.add_argument("--train_stage", default=0, type=int)  # my special pile mode
    parser.add_argument(
        "--ds_bucket_mb", default=500, type=int
    )  # deepspeed bucket size in MB. 200 seems enough
    parser.add_argument(
        "--head_size", default=64, type=int
    )  # can try larger values for larger models
    # LoRA ranks for w/a/v/g time-mix projections. 0 (default) -> n_embd-scaled
    # heuristic in src/model.py. Set explicitly to match a pretrained ckpt.
    # Reference: RWKV7-G1f-7.2B uses 128 / 128 / 96 / 480.
    parser.add_argument("--d_decay_lora", default=0, type=int)
    parser.add_argument("--d_aaa_lora", default=0, type=int)
    parser.add_argument("--d_mv_lora", default=0, type=int)
    parser.add_argument("--d_gate_lora", default=0, type=int)
    parser.add_argument("--load_partial", default=0, type=int)
    # 1 -> use bitsandbytes AdamW8bit (m/v stored as int8, ~4x smaller than fp32).
    parser.add_argument("--optim_8bit", default=0, type=int)
    # Optimizer choice. "adam" (12 B/param Adam, default; uses CPU offload variant
    # if deepspeed_offload), "adafactor" (8 B/param, factorized v; GPU-only),
    # "8bit" (~3 B/param bnb AdamW8bit, less stable). --optim_8bit 1 still selects 8bit.
    parser.add_argument("--optim", default="adam", type=str, choices=["adam", "adafactor", "8bit"])
    parser.add_argument("--magic_prime", default=0, type=int)
    parser.add_argument("--my_testing", default="x070", type=str)
    parser.add_argument("--my_exit_tokens", default=0, type=int)

    # ---- Diffusion (dLLM-style infilling) training mode ----
    parser.add_argument(
        "--diffusion_mode", default=0, type=int
    )  # 0 = standard LM, 1 = triplet diffusion
    parser.add_argument("--diff_block_size", default=32, type=int)  # tokens per logical block
    parser.add_argument(
        "--diff_min_mask_ratio", default=0.0, type=float
    )  # lower bound for per-sample r
    parser.add_argument(
        "--diff_max_mask_ratio", default=1.0, type=float
    )  # upper bound for per-sample r
    parser.add_argument(
        "--diff_pad_id", default=65534, type=int
    )  # pad token for tail. Dedicated dummy slot (penultimate; MASK uses last). MUST NOT be EOS (0) or MASK (vocab_size-1).
    parser.add_argument(
        "--diff_max_doc_tokens", default=0, type=int
    )  # 0 => auto (= n_blocks * block_size); drop docs longer than this at MyDataset.__init__
    # Force-mask the doc-ending EOS (token 0) in every sample so the model gets
    # consistent supervision on "when to stop". Without this, EOS is masked
    # only ~50% of the time on average, making it a < 1/400 fraction of loss
    # signals -> model can't learn to emit it reliably.
    parser.add_argument("--diff_force_mask_eos", default=1, type=int)
    # Force-mask trailing PAD positions in b1/b2 too. Otherwise the model can
    # cheat: it sees "MASK followed by pad" as a giveaway that the masked
    # position is EOS (a trivial pattern that doesn't transfer to inference,
    # where pad never appears). Pad has no loss signal either way.
    parser.add_argument("--diff_force_mask_pad", default=1, type=int)
    # LLaDA full-mask trick: with this probability, override a block's r to 1.0
    # so all its lossable positions get masked. Brings the training input
    # distribution closer to inference (where every generation block starts
    # as all-MASK before denoising). 0 disables; 0.05-0.15 is typical.
    parser.add_argument("--diff_full_mask_prob", default=0.10, type=float)
    parser.add_argument(
        "--diff_conf_lambda", default=0.0, type=float
    )  # LLaDA-2.0 CAP: weight of the entropy-minimization aux loss on correctly-predicted masked positions; 0 disables.

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    ########################################################################################################

    import datetime
    import math
    import os
    import sys
    import time
    import warnings

    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    if "deepspeed" in args.strategy:
        import deepspeed
    from pytorch_lightning import seed_everything

    if args.random_seed >= 0:
        print(
            f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n"
            * 3
        )
        seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings(
        "ignore", ".*Consider increasing the value of the `num_workers` argument*"
    )
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
    # os.environ["WDS_SHOW_SEED"] = "1"

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = args.grad_clip
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.max_epochs = -1  # continue forever
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE"] = str(args.head_size)
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)  # default = 3.5x emb size

    args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)

    args.epoch_count = args.magic_prime // 40320
    args.epoch_steps = 40320 // args.real_bsz
    assert args.epoch_steps * args.real_bsz == 40320

    if args.train_stage >= 2:  # find latest saved model
        list_p = []
        for p in os.listdir(args.proj_dir):
            if p.startswith("rwkv") and p.endswith(".pth"):
                p = ((p.split("-"))[1].split("."))[0]
                if p != "final":
                    if p == "init":
                        p = -1
                    else:
                        p = int(p)
                    list_p += [p]
        list_p.sort()
        if not list_p:
            # First run: no checkpoints in proj_dir yet. Keep args.load_model as
            # whatever the user passed (an external pretrained ckpt, or "" for
            # from-scratch which triggers generate_init_weight below).
            rank_zero_info(
                f"########## train_stage={args.train_stage}: no rwkv-*.pth in "
                f"{args.proj_dir}, falling back to --load_model='{args.load_model}' ##########"
            )
        else:
            max_p = list_p[-1]
            if len(list_p) > 1:
                args.my_pile_prev_p = list_p[-2]  # in case max_p is corrupted
            if max_p == -1:
                args.load_model = f"{args.proj_dir}/rwkv-init.pth"
            else:
                args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
                if args.warmup_steps < 0:
                    args.warmup_steps = 10
            args.epoch_begin = max_p + 1

            # Look for a Lightning resume ckpt at the same epoch. If present we
            # pass its path to trainer.fit(ckpt_path=...) so model weights AND
            # optimizer state (Adam m/v, fp32 master, lr scheduler) are all
            # restored — losslessly resumes across container kills. Without it,
            # we still load the model weights manually below but the optimizer
            # starts cold.
            resume_candidate = f"{args.proj_dir}/rwkv-{max_p}-resume.ckpt"
            if os.path.exists(resume_candidate):
                args.resume_ckpt_path = resume_candidate
                rank_zero_info(
                    f"########## Found resume ckpt {resume_candidate} -> "
                    f"will restore model + optimizer + lr_scheduler ##########"
                )
            else:
                args.resume_ckpt_path = None
                if max_p >= 0:
                    rank_zero_info(
                        f"########## No -resume.ckpt for epoch {max_p}; loading "
                        f"weights only (Adam state will reset, expect a transient "
                        f"loss bump for ~hundreds of steps) ##########"
                    )

    samples_per_epoch = args.epoch_steps * args.real_bsz
    tokens_per_epoch = samples_per_epoch * args.ctx_len
    try:
        deepspeed_version = deepspeed.__version__
    except:
        deepspeed_version = None
        pass
    rank_zero_info(
        f"""
############################################################################
#
# RWKV-7 {args.precision.upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, bsz {args.num_nodes}x{args.devices}x{args.micro_bsz}={args.real_bsz}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
#
# Data = {args.data_file} ({args.data_type}), ProjDir = {args.proj_dir}
#
# Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
#
# Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
#
# Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
#
# Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
#
# Found torch {torch.__version__}, recommend latest torch
# Found deepspeed {deepspeed_version}, recommend latest deepspeed
# Found pytorch_lightning {pl.__version__}, recommend 1.9.5
#
############################################################################
"""
    )
    rank_zero_info(str(vars(args)) + "\n")

    assert args.data_type in ["binidx"]

    if args.lr_final == 0 or args.lr_init == 0:
        rank_zero_info(
            "\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n"
        )

    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info(
                "\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n"
            )
    if args.precision == "fp16":
        rank_zero_info(
            "\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n"
        )

    os.environ["RWKV_JIT_ON"] = "1"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0"  # somehow incompatible

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    ########################################################################################################

    from src.dataset import MyDataset
    from src.trainer import generate_init_weight, train_callback

    train_data = MyDataset(args)
    args.vocab_size = train_data.vocab_size

    if args.diffusion_mode == 1:
        # Reuse the last vocab slot as MASK. The RWKV-world tokenizer covers ids 0..65529
        # (65530 real slots: id 0 is EOS, ids 1..65529 from rwkv_vocab_v20230424.txt).
        # Pretrained RWKV-v7 world ckpts have emb/head padded to 65536 for power-of-two
        # alignment, leaving ids 65530..65535 as unused embedding rows that never get
        # indexed by real data. We commandeer id (vocab_size - 1) = 65535 for MASK, so
        # vocab_size stays at the model's native 65536 and no ckpt resize is needed.
        args.diff_mask_id = args.vocab_size - 1
        assert (
            args.ctx_len >= 3 * args.diff_block_size
        ), f"ctx_len ({args.ctx_len}) too small to fit one diffusion triplet (3 * block_size = {3 * args.diff_block_size})"
        assert args.diff_pad_id != args.diff_mask_id, (
            f"diff_pad_id ({args.diff_pad_id}) must differ from diff_mask_id ({args.diff_mask_id}); "
            f"using MASK as tail pad would pollute its semantics."
        )
        # If pad_id equals EOS (0), dataset.py's per-pad-position loss masking will
        # also kill the loss at *real* document-ending EOS tokens — the model then
        # never learns when to emit EOS. Use a dedicated dummy slot (e.g. 65534).
        assert args.diff_pad_id != 0, (
            f"diff_pad_id must NOT be 0 (EOS). Use a dummy vocab slot like 65534 "
            f"so dataset.py can mask pad from loss without also masking real EOS."
        )
        assert 0.0 <= args.diff_min_mask_ratio <= args.diff_max_mask_ratio <= 1.0, (
            f"need 0 <= diff_min_mask_ratio ({args.diff_min_mask_ratio}) "
            f"<= diff_max_mask_ratio ({args.diff_max_mask_ratio}) <= 1"
        )
        n_blocks_per_sample = args.ctx_len // (3 * args.diff_block_size)
        rank_zero_info(
            f"########## Diffusion mode ON: block_size={args.diff_block_size}, "
            f"n_blocks_per_sample={n_blocks_per_sample}, "
            f"mask_id={args.diff_mask_id}, pad_id={args.diff_pad_id}, "
            f"mask_ratio in [{args.diff_min_mask_ratio}, {args.diff_max_mask_ratio}], "
            f"vocab_size={args.vocab_size} (reusing last dummy slot) ##########"
        )

    from src.model import RWKV

    model = RWKV(args)

    if len(args.load_model) == 0 or args.train_stage == 1:  # shall we build the initial weights?
        init_weight_name = f"{args.proj_dir}/rwkv-init.pth"
        generate_init_weight(model, init_weight_name)  # save initial weights
        args.load_model = init_weight_name

    rank_zero_info(f"########## Loading {args.load_model}... ##########")
    try:
        load_dict = torch.load(args.load_model, map_location="cpu", weights_only=True, mmap=True)
        load_keys = list(load_dict.keys())
        for k in load_keys:
            if k.startswith("_forward_module."):
                load_dict[k.replace("_forward_module.", "")] = load_dict[k]
                del load_dict[k]
    except:
        rank_zero_info(f"Bad checkpoint {args.load_model}")
        if args.train_stage >= 2:  # try again using another checkpoint
            max_p = args.my_pile_prev_p
            if max_p == -1:
                args.load_model = f"{args.proj_dir}/rwkv-init.pth"
            else:
                args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
            args.epoch_begin = max_p + 1
            rank_zero_info(f"Trying {args.load_model}")
            load_dict = torch.load(
                args.load_model, map_location="cpu", weights_only=True, mmap=True
            )

    if args.load_partial == 1:
        load_keys = load_dict.keys()
        for k in model.state_dict():
            if k not in load_keys:
                load_dict[k] = model.state_dict()[k]

    model.load_state_dict(load_dict)

    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[train_callback(args)],
    )

    if trainer.global_rank == 0:
        for n in model.state_dict():
            shape = model.state_dict()[n].shape
            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            s3 = str(shape[3]) if len(shape) > 3 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}")

    if "deepspeed" in args.strategy:
        # PL's Trainer(gradient_clip_val=...) is silently ignored on the DS
        # path — DeepSpeed's bf16 optimizer reads config["gradient_clipping"]
        # instead. Without this, FusedAdam (stage_2 no-offload) sees raw
        # un-clipped grads; an inf grad → m/v become inf → next step writes
        # NaN to that param row. (DeepSpeedCPUAdam on the offload path has
        # its own NaN/Inf skip, which is why stage_2_offload masked this bug.)
        trainer.strategy.config["gradient_clipping"] = float(args.grad_clip)
        # Force fp32 grad accumulation + fp32 inter-rank communication. With
        # bf16 training + ZeRO-2 (no offload), DeepSpeed's default is to
        # accumulate grads in bf16 and allreduce in bf16 too. With ACC_GRAD>1
        # × 8-rank sum, the bf16 7-bit mantissa cancels into rare-but-large
        # garbage values (sometimes literal inf), which then poison Adam's
        # m/v on the very next optimizer step. ZeRO-2 OFFLOAD escapes this
        # because grads must be D2H-copied to fp32 for DeepSpeedCPUAdam,
        # giving free fp32 accumulation. Force the same behavior on-GPU.
        # `data_types.grad_accum_dtype` requires DeepSpeed >= 0.10; older
        # versions silently ignore it (harmless, but you'd still see NaNs).
        trainer.strategy.config["communication_data_type"] = "fp32"
        trainer.strategy.config.setdefault("data_types", {})["grad_accum_dtype"] = "fp32"
        # Belt-and-braces: explicitly turn off any fp16 master grads path.
        if "bf16" in trainer.strategy.config and isinstance(trainer.strategy.config["bf16"], dict):
            trainer.strategy.config["bf16"].setdefault("enabled", True)
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = (
            args.ds_bucket_mb * 1000 * 1000
        )
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = (
            args.ds_bucket_mb * 1000 * 1000
        )

    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    data_loader = DataLoader(
        train_data,
        shuffle=False,
        pin_memory=True,
        batch_size=args.micro_bsz,
        num_workers=1,
        persistent_workers=False,
        drop_last=True,
    )

    if trainer.global_rank == 0:
        print(f"### Preparing for training (loaded {args.load_model}). Please wait...")
    trainer.fit(model, data_loader, ckpt_path=getattr(args, "resume_ckpt_path", None))
