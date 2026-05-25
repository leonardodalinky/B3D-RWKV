#!/usr/bin/env python3
"""Benchmark a list of (M_BSZ, STRATEGY, GRAD_CP) configs by running the demo
training script just long enough to capture peak GPU memory and avg s/step,
then killing it. Outputs a comparison table.

Usage on the GPU box:
    python train/bench_configs.py
"""
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/data/rsync/RWKV/DiffuRWKV").resolve()
SCRIPT = REPO / "train" / "demo-training-run-diffusion.sh"
LOG_DIR = REPO / "log" / "bench"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# How many step-time samples to collect per config before killing.
SAMPLES_NEEDED = 3
# Hard cap (seconds) per config. DeepSpeed Adam offload allocates ~86GB CPU
# RAM (7.2B × 12B per param) at startup which takes 3-5min on first run, then
# is mmap'd cheap. Plus model load + DDP rendezvous. Budget 15min per config.
HARD_TIMEOUT = 900
# Polling interval for nvidia-smi
POLL_INTERVAL = 5.0
# Use a throwaway proj_dir that we pre-populate with a SYMLINK to one
# rwkv-N.pth (so auto-detect grabs it for --load_model) but no -resume.ckpt
# (so DeepSpeed doesn't spend minutes loading sharded optim state).
BENCH_PROJ_DIR = "/tmp/bench_proj_dir_DELETE_ME"
SOURCE_PTH = (
    "/data/rsync/RWKV/DiffuRWKV/train/out/diff-L32-D4096-x070-blk32-ctx6144_fixEOF/rwkv-19.pth"
)


def _prep_proj_dir():
    """Set up the throwaway proj_dir with a single rwkv-N.pth symlink and no
    resume.ckpt sibling, so train.py's auto-detect path runs cleanly."""
    os.makedirs(BENCH_PROJ_DIR, exist_ok=True)
    # Clean any prior content from a previous run.
    for fn in os.listdir(BENCH_PROJ_DIR):
        p = os.path.join(BENCH_PROJ_DIR, fn)
        if os.path.islink(p) or os.path.isfile(p):
            os.unlink(p)
    if not os.path.exists(SOURCE_PTH):
        raise SystemExit(f"source pth not found: {SOURCE_PTH}")
    # Symlink as rwkv-19.pth so train.py auto-detect logic sees max_p=19,
    # uses --load_model = this path, and (no resume.ckpt sibling) starts
    # with cold optimizer state.
    target = os.path.join(BENCH_PROJ_DIR, "rwkv-19.pth")
    os.symlink(SOURCE_PTH, target)
    print(f"[prep] proj_dir = {BENCH_PROJ_DIR} -> {SOURCE_PTH}")


def gpu_memory_used() -> list[int]:
    """Return per-GPU memory used in MiB for all visible GPUs."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [int(x.strip()) for x in out.strip().splitlines() if x.strip()]


def gpu_util() -> list[int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [int(x.strip()) for x in out.strip().splitlines() if x.strip()]


def kill_proc_tree(pid: int):
    try:
        # Negative pid -> kill whole process group
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    for _ in range(15):
        time.sleep(1)
        if not _is_running(pid):
            return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_gpu_clear(threshold_mib: int = 1000, timeout: float = 60.0):
    """After kill, wait until GPU memory drops below threshold (so next run
    starts from clean slate). Returns final max usage observed."""
    start = time.time()
    last = max(gpu_memory_used())
    while time.time() - start < timeout:
        cur = max(gpu_memory_used())
        if cur < threshold_mib:
            return cur
        last = cur
        time.sleep(2)
    return last


# Match the lightning prog-bar format like:  s/step=1.234, REAL it/s=0.81, ...
_SSTEP_RE = re.compile(r"s/step=([0-9]+\.[0-9]+)")
_KTPS_RE = re.compile(r"Kt/s=([0-9]+\.[0-9]+)")
_OOM_RE = re.compile(r"OutOfMemoryError|CUDA out of memory")


def run_one_config(name: str, env_overrides: dict) -> dict:
    log_path = LOG_DIR / f"{name}.log"
    print(f"\n{'='*70}\n[{name}] env: {env_overrides}\n{'='*70}", flush=True)

    # Pre-check GPU is clean
    mem = gpu_memory_used()
    if max(mem) > 1000:
        print(f"  WARN: GPU memory not clean before start: {mem} MiB; sleeping 10s")
        time.sleep(10)

    # Always force a throwaway proj_dir so we never load the real resume.ckpt.
    full_overrides = {"PROJ_DIR": BENCH_PROJ_DIR, **env_overrides}
    env = {**os.environ, **{k: str(v) for k, v in full_overrides.items()}}
    # New process group so we can kill the whole tree (python + 8 ddp workers)
    proc = subprocess.Popen(
        ["bash", str(SCRIPT)],
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        text=True,
        bufsize=1,
    )

    sstep_samples: list[float] = []
    kts_samples: list[float] = []
    peak_mem = 0
    peak_per_gpu = [0] * 8
    peak_util = 0
    util_samples = []
    oom = False
    start = time.time()
    last_progress_print = start

    try:
        with open(log_path, "w") as logf:
            for line in proc.stdout:
                logf.write(line)
                logf.flush()

                if _OOM_RE.search(line):
                    oom = True
                    print(f"  ❌ OOM detected; killing")
                    break

                m = _SSTEP_RE.search(line)
                if m:
                    val = float(m.group(1))
                    if val > 0.05:  # skip first bogus tiny values
                        sstep_samples.append(val)
                m = _KTPS_RE.search(line)
                if m:
                    kts_samples.append(float(m.group(1)))

                # Periodic GPU snapshot
                now = time.time()
                if now - last_progress_print >= POLL_INTERVAL:
                    last_progress_print = now
                    try:
                        cur = gpu_memory_used()
                        cu = gpu_util()
                        peak_mem = max(peak_mem, max(cur))
                        peak_per_gpu = [max(a, b) for a, b in zip(peak_per_gpu, cur)]
                        peak_util = max(peak_util, max(cu))
                        util_samples.append(sum(cu) / len(cu))  # avg across GPUs
                        if sstep_samples:
                            elapsed = now - start
                            print(
                                f"  t={elapsed:5.0f}s  peak_mem={peak_mem/1024:.1f}GB  "
                                f"util={cu}  s/step samples={len(sstep_samples)}",
                                flush=True,
                            )
                    except Exception as e:
                        print(f"  warn: nvidia-smi failed: {e}")

                if len(sstep_samples) >= SAMPLES_NEEDED:
                    print(f"  ✅ collected {SAMPLES_NEEDED} samples; killing")
                    break
                if time.time() - start > HARD_TIMEOUT:
                    print(f"  ⏱  timeout {HARD_TIMEOUT}s; killing")
                    break
    finally:
        kill_proc_tree(proc.pid)
        proc.wait(timeout=30)
        # Drain remaining stdout into log
        if proc.stdout:
            try:
                rest = proc.stdout.read()
                with open(log_path, "a") as logf:
                    logf.write(rest)
            except Exception:
                pass

    final_mem = wait_gpu_clear(threshold_mib=1500, timeout=90.0)
    if final_mem > 1500:
        print(f"  WARN: GPU memory still {final_mem} MiB after kill; sleep 30s extra")
        time.sleep(30)

    avg_sstep = sum(sstep_samples) / len(sstep_samples) if sstep_samples else float("nan")
    avg_kts = sum(kts_samples) / len(kts_samples) if kts_samples else float("nan")
    avg_util = sum(util_samples) / len(util_samples) if util_samples else 0
    return {
        "name": name,
        "env": env_overrides,
        "oom": oom,
        "n_samples": len(sstep_samples),
        "avg_sstep": avg_sstep,
        "avg_kts": avg_kts,
        "peak_mem_gb": peak_mem / 1024,
        "peak_per_gpu_gb": [m / 1024 for m in peak_per_gpu],
        "peak_util_pct": peak_util,
        "avg_util_pct": avg_util,
        "log": str(log_path),
    }


CONFIGS = [
    # (name, env overrides). All M_BSZ values divide 5040 = 40320/8.
    # Goal: find max throughput on 8x H100 given offload is mandatory and
    # B=80 stage_2_offload OOMs. Probe stage_3 (param sharding frees ~12GB/GPU
    # so it can fit larger micro_bsz) and grad accumulation (amortize CPU-Adam).
    #
    # ---- stage_2_offload anchors (probe practical max B) ----
    ("a_s2o_b16_a1", dict(M_BSZ=16, STRATEGY="deepspeed_stage_2_offload", GRAD_CP=1, ACC_GRAD=1)),
    ("b_s2o_b48_a1", dict(M_BSZ=48, STRATEGY="deepspeed_stage_2_offload", GRAD_CP=1, ACC_GRAD=1)),
    ("c_s2o_b63_a1", dict(M_BSZ=63, STRATEGY="deepspeed_stage_2_offload", GRAD_CP=1, ACC_GRAD=1)),
    # ---- stage_2_offload + grad_accum: amortize CPU-Adam fixed overhead ----
    # Same effective batch as B=63 but each accum step does less compute,
    # exposing whether CPU-Adam is dominant.
    ("d_s2o_b16_a4", dict(M_BSZ=16, STRATEGY="deepspeed_stage_2_offload", GRAD_CP=1, ACC_GRAD=4)),
    ("e_s2o_b48_a4", dict(M_BSZ=48, STRATEGY="deepspeed_stage_2_offload", GRAD_CP=1, ACC_GRAD=4)),
    # ---- stage_3 + offload: weights also sharded, frees ~12GB/GPU ----
    # Should let M_BSZ go higher than stage_2_offload's ceiling.
    ("f_s3o_b63_a1", dict(M_BSZ=63, STRATEGY="deepspeed_stage_3_offload", GRAD_CP=1, ACC_GRAD=1)),
    ("g_s3o_b80_a1", dict(M_BSZ=80, STRATEGY="deepspeed_stage_3_offload", GRAD_CP=1, ACC_GRAD=1)),
    ("h_s3o_b112_a1", dict(M_BSZ=112, STRATEGY="deepspeed_stage_3_offload", GRAD_CP=1, ACC_GRAD=1)),
    # ---- stage_3 + offload + grad_accum (the throughput-king candidate) ----
    ("i_s3o_b80_a4", dict(M_BSZ=80, STRATEGY="deepspeed_stage_3_offload", GRAD_CP=1, ACC_GRAD=4)),
]


def main():
    _prep_proj_dir()
    results = []
    skip_remaining_no_offload_above = None  # if a no-offload OOMs at some B, skip larger
    skip_offload_above = None  # if offload OOMs at B=X, skip larger offload configs too
    for name, env in CONFIGS:
        b_now = int(env.get("M_BSZ", 0))
        if (
            skip_offload_above is not None
            and env.get("STRATEGY") == "deepspeed_stage_2_offload"
            and b_now >= skip_offload_above
        ):
            print(f"\n[skip] {name}: M_BSZ={b_now} >= known-OOM offload {skip_offload_above}")
            continue
        if (
            skip_remaining_no_offload_above is not None
            and env.get("STRATEGY") == "deepspeed_stage_2"
            and b_now >= skip_remaining_no_offload_above
        ):
            print(
                f"\n[skip] {name}: M_BSZ={b_now} >= known-OOM no-offload {skip_remaining_no_offload_above}"
            )
            continue
        try:
            r = run_one_config(name, env)
        except KeyboardInterrupt:
            print("\ninterrupted")
            break
        results.append(r)
        if r["oom"]:
            if env.get("STRATEGY") == "deepspeed_stage_2_offload":
                skip_offload_above = min(skip_offload_above or b_now, b_now)
            else:
                skip_remaining_no_offload_above = min(
                    skip_remaining_no_offload_above or b_now, b_now
                )

    print("\n\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(
        f"{'name':<28}  {'OOM':>4}  {'samples':>8}  {'s/step':>8}  {'Kt/s':>8}  {'peak GB':>8}  {'util%':>6}"
    )
    for r in results:
        oom = "YES" if r["oom"] else "no"
        print(
            f"{r['name']:<28}  {oom:>4}  {r['n_samples']:>8}  "
            f"{r['avg_sstep']:>8.3f}  {r['avg_kts']:>8.1f}  "
            f"{r['peak_mem_gb']:>8.1f}  {r['peak_util_pct']:>6}"
        )
    print("=" * 90)

    # Print recommendation
    valid = [r for r in results if not r["oom"] and r["n_samples"] > 0]
    if valid:
        best = min(valid, key=lambda r: r["avg_sstep"])
        print(
            f"\nBEST throughput: {best['name']} → {best['avg_sstep']:.3f}s/step, "
            f"{best['avg_kts']:.1f} Kt/s, peak {best['peak_mem_gb']:.1f}GB"
        )
        print(f"  env: {best['env']}")


if __name__ == "__main__":
    main()
