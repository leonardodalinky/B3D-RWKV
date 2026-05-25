import datetime
import glob
import math
import os
import shutil
import subprocess
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from torch.utils.data import DataLoader


def my_save(args, trainer, dd, ff):
    """Save model state for inference (rwkv-N.pth). Compact, no optimizer state."""
    if "deepspeed_stage_3" in args.strategy:
        trainer.save_checkpoint(ff, weights_only=True)
    else:
        torch.save(dd, ff)


def my_save_resume(args, trainer, ff_resume, keep_n=2):
    """Save full Lightning checkpoint (model + optimizer + lr_scheduler) so the
    next process can resume losslessly across container kills. Path becomes a
    DIRECTORY for DeepSpeed strategies (sharded format) and a FILE otherwise.
    Old resume ckpts are pruned to bound disk usage at ~keep_n × per-ckpt size.
    """
    try:
        trainer.save_checkpoint(ff_resume)
    except Exception as e:
        rank_zero_info(f"[resume] save FAILED (non-fatal, training continues): {e}")
        return
    rank_zero_info(f"[resume] saved {ff_resume}")
    if not trainer.is_global_zero:
        return  # rank 0 owns cleanup
    pattern = os.path.join(args.proj_dir, "rwkv-*-resume.ckpt")
    paths = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p))
    for p in paths[:-keep_n]:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
            rank_zero_info(f"[resume] pruned old ckpt {p}")
        except OSError as e:
            rank_zero_info(f"[resume] could not prune {p}: {e}")


class train_callback(pl.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args

        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        # LR schedule
        w_step = args.warmup_steps

        if args.my_exit_tokens != 0:  # cosine decay
            real_tokens = real_step * args.ctx_len * args.real_bsz
            warmup_tokens = w_step * args.ctx_len * args.real_bsz
            progress = (real_tokens - warmup_tokens) / (abs(args.my_exit_tokens) - warmup_tokens)
            progress = max(0, min(1, progress))
            lr_final_factor = args.lr_final / args.lr_init
            lr_mult = (0.5 + lr_final_factor / 2) + (0.5 - lr_final_factor / 2) * math.cos(
                math.pi * progress
            )
            if args.my_exit_tokens > 0:
                lr = args.lr_init * lr_mult
            else:
                lr = (lr + args.lr_init * lr_mult) / 2
            if progress >= 1:
                if (trainer.is_global_zero) or ("deepspeed_stage_3" in args.strategy):
                    my_save(
                        args,
                        trainer,
                        pl_module.state_dict(),
                        f"{args.proj_dir}/rwkv-final.pth",
                    )
                    exit(0)
        if trainer.global_step < w_step:
            lr = lr * (0.01 + 0.99 * trainer.global_step / w_step)

        wd_now = args.weight_decay

        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_now
            param_group["lr"] = lr * param_group["my_lr_scale"]

        trainer.my_lr = lr
        trainer.my_wd = wd_now

        # On a fresh run global_step is 0; on resume Lightning restores it to
        # the saved value, so we also need to bootstrap when the attributes are
        # simply missing on this Trainer instance.
        if trainer.is_global_zero and not hasattr(trainer, "my_loss_sum"):
            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0
            trainer.my_log = open(args.proj_dir + "/train_log.txt", "a")
            trainer.my_log.write(f"NEW RUN {args.my_timestamp}\n{vars(self.args)}\n")
            try:
                print(f"\n{trainer.strategy.config}\n")
                trainer.my_log.write(f"{trainer.strategy.config}\n")
            except:
                pass
            trainer.my_log.flush()
            if len(args.wandb) > 0:
                print("Login to wandb...")
                import wandb

                wandb.init(
                    project=args.wandb,
                    name=args.run_name + " " + args.my_timestamp,
                    config=args,
                    save_code=False,
                )
                trainer.my_wandb = wandb

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, optimizer_idx=0):
        # Catch the FIRST optimizer step where any grad is non-finite, BEFORE
        # the optimizer consumes it and writes NaN into the weights. Tells us
        # whether the failure is upstream (forward/backward emitted inf grad
        # → wkv kernel / CE / activation overflow) or downstream (grads clean
        # but Adam still corrupts weights → optimizer-side numerical issue).
        #
        # ZeRO-2 caveat: each rank only owns its grad partition. We let every
        # rank scan its own and print with [rank=N] so a NaN sharded onto
        # rank 5 isn't invisible. Expect up to world_size lines per failure.
        rank = trainer.global_rank
        with torch.no_grad():
            worst_n, worst_max = "?", 0.0
            for n, p in pl_module.named_parameters():
                if p.grad is None:
                    continue
                gf = p.grad.detach().float()
                if not torch.isfinite(gf).all():
                    finite_mask = torch.isfinite(gf)
                    finite_max = (
                        gf[finite_mask].abs().max().item() if finite_mask.any() else float("nan")
                    )
                    print(
                        f"[grad-NaN][rank={rank}] step={trainer.global_step} "
                        f"param={n}: non-finite "
                        f"(nan={int(torch.isnan(gf).sum())}, "
                        f"inf={int(torch.isinf(gf).sum())}, numel={gf.numel()})  "
                        f"|g|.max(finite)={finite_max:.3g}",
                        flush=True,
                    )
                    return
                m = gf.abs().max().item()
                if m > worst_max:
                    worst_max, worst_n = m, n
            if worst_max > 1e3:
                print(
                    f"[grad-large][rank={rank}] step={trainer.global_step} "
                    f"param={worst_n}: |g|.max={worst_max:.3g}",
                    flush=True,
                )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        token_per_step = args.ctx_len * args.real_bsz
        real_step = trainer.global_step + args.epoch_begin * args.epoch_steps

        # Per-step weight health probe: scan max |w| of bf16 params and emit a
        # loud warning the FIRST step it goes non-finite. Catches the moment
        # weights wreck themselves, before the NaN propagates into Adam's m/v
        # and you can never recover. Pairs with on_before_optimizer_step:
        # if grad-NaN fires first, root cause is upstream of the optimizer;
        # if weight-NaN fires without grad-NaN, root cause is the optimizer.
        if trainer.is_global_zero:
            with torch.no_grad():
                worst_n, worst_max = "?", 0.0
                any_bad = False
                for n, p in pl_module.named_parameters():
                    if p is None or p.numel() == 0:
                        continue
                    pf = p.detach().float()
                    if not torch.isfinite(pf).all():
                        rank_zero_info(
                            f"[weight-NaN] step={trainer.global_step} param={n}: "
                            f"non-finite (nan={int(torch.isnan(pf).sum())}, "
                            f"inf={int(torch.isinf(pf).sum())})"
                        )
                        any_bad = True
                        break
                    m = pf.abs().max().item()
                    if m > worst_max:
                        worst_max, worst_n = m, n
                if not any_bad and worst_max > 1e3:
                    rank_zero_info(
                        f"[weight-large] step={trainer.global_step} param={worst_n}: "
                        f"max|w|={worst_max:.3g}"
                    )

        if trainer.is_global_zero:  # logging
            t_now = time.time_ns()
            kt_s = 0
            t_cost = 0.0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                self.log("s/step", t_cost, prog_bar=True, on_step=True)
                self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            trainer.my_loss = trainer.my_loss_all.float().mean().item()
            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)
            diff_acc = getattr(trainer, "my_diff_acc", None)
            if diff_acc is not None:
                self.log("mask_acc", diff_acc, prog_bar=True, on_step=True)
                # Surface n_masked alongside mask_acc — when micro_bsz=1 + a
                # short doc, n_masked can drop to single digits and mask_acc
                # will spike to 1.0 on a tiny denominator. Without this
                # progress-bar field, mask_acc=1.0 looks like the model is
                # superhuman when in fact it just nailed 1-of-1 EOS.
                diff_n = getattr(trainer, "my_diff_n_masked", None)
                if diff_n is not None:
                    self.log("n_mask", float(diff_n), prog_bar=True, on_step=True)

            if len(args.wandb) > 0:
                lll = {
                    "loss": trainer.my_loss,
                    "lr": trainer.my_lr,
                    "wd": trainer.my_wd,
                    "Gtokens": real_step * token_per_step / 1e9,
                }
                if kt_s > 0:
                    lll["kt/s"] = kt_s
                    lll["s/step"] = t_cost
                if diff_acc is not None:
                    lll["mask_acc"] = diff_acc
                    lll["n_masked_per_step"] = getattr(trainer, "my_diff_n_masked", 0)
                trainer.my_wandb.log(lll, step=int(real_step))

        if (trainer.is_global_zero) or ("deepspeed_stage_3" in args.strategy):  # save pth
            if args.magic_prime > 0:
                if int(real_step) == int(args.magic_prime // args.real_bsz) - 1:
                    to_save_dict = pl_module.state_dict()
                    my_save(
                        args,
                        trainer,
                        to_save_dict,
                        f"{args.proj_dir}/rwkv-final.pth",
                    )

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        dataset = trainer.train_dataloader.dataset.datasets
        assert "MyDataset" in str(dataset)
        dataset.global_rank = trainer.global_rank
        dataset.real_epoch = int(args.epoch_begin + trainer.current_epoch)
        dataset.world_size = trainer.world_size
        # print(f'########## world_size {dataset.world_size} global_rank {dataset.global_rank} real_epoch {dataset.real_epoch} ##########')

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        to_save_dict = {}
        epoch_no = args.epoch_begin + trainer.current_epoch
        save_this_epoch = (
            args.epoch_save > 0 and trainer.current_epoch % args.epoch_save == 0
        ) or (trainer.current_epoch == args.epoch_count - 1)
        if (trainer.is_global_zero) or ("deepspeed_stage_3" in args.strategy):  # save pth
            if save_this_epoch:
                if args.data_type == "wds_img":
                    raw_dict = pl_module.state_dict()
                    for k in raw_dict:
                        if k.startswith("encoder.") or k.startswith("decoder."):
                            to_save_dict[k] = raw_dict[k]
                else:
                    to_save_dict = pl_module.state_dict()
                try:
                    my_save(
                        args,
                        trainer,
                        to_save_dict,
                        f"{args.proj_dir}/rwkv-{epoch_no}.pth",
                    )
                except Exception as e:
                    print("Error\n\n", e, "\n\n")

        # Save the resumable Lightning ckpt (model + optimizer + lr_scheduler).
        # All ranks must call this — DeepSpeed needs every rank to write its
        # shard. Only rank 0's prune step actually deletes files. Cadence
        # follows EPOCH_SAVE so it stays aligned with the model-only ckpts.
        if save_this_epoch:
            my_save_resume(
                args,
                trainer,
                f"{args.proj_dir}/rwkv-{epoch_no}-resume.ckpt",
                keep_n=2,
            )

        if trainer.is_global_zero:  # logging
            trainer.my_log.write(
                f"{args.epoch_begin + trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()} {trainer.current_epoch}\n"
            )
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0


@rank_zero_only
def generate_init_weight(model, init_weight_name):
    mm = model.generate_init_weight()

    if model.args.train_stage == 1:
        if len(model.args.load_model) > 0:
            print(f"Combine weights from {model.args.load_model}...")
            load_dict = torch.load(model.args.load_model, map_location="cpu")
            for k in load_dict:
                try:
                    assert k in mm
                except:
                    print("missing", k)
                    exit(0)
                src = load_dict[k]
                try:
                    mm[k] = src.reshape(mm[k].shape)
                except:
                    tmp = mm[k].squeeze().clone()
                    print(k, src.shape, "-->", mm[k].shape)
                    ss = src.shape[0]
                    dd = tmp.shape[0]
                    for i in range(dd):
                        pos = i / dd * ss
                        if pos >= ss - 1:
                            tmp[i] = src[ss - 1]
                        else:
                            p0 = int(math.floor(pos))
                            ii = pos - p0
                            tmp[i] = src[p0] * (1 - ii) + src[p0 + 1] * (ii)
                    mm[k] = tmp.reshape(mm[k].shape)
                    sss = src.squeeze().float().cpu().numpy()
                    print(sss[:10], "...", sss[-10:])
                    mmm = mm[k].squeeze().float().cpu().numpy()
                    print(mmm[:10], "...", mmm[-10:])

    print(f"Save to {init_weight_name}...")
    torch.save(mm, init_weight_name)

    if model.args.train_stage == 1:
        print("Done. Now go for stage 2.")
        exit(0)
