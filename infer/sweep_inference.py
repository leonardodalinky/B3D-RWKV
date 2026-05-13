#!/usr/bin/env python3
"""Sweep multiple sampling configurations on a single prompt.

Loads the model ONCE and reuses it across every preset — no per-preset
cold-start cost. Calls ``diffusion_sample.run_one`` with ``verbose=False``
and writes the returned text into a single markdown file for side-by-side
visual comparison.

Usage:
    python infer/sweep_inference.py
    CKPT=train/out/.../rwkv-30.pth python infer/sweep_inference.py
    PROMPT='User: hi\\n\\nAssistant:' python infer/sweep_inference.py
    PRESETS=greedy,default,heavy_penalty python infer/sweep_inference.py
"""
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace


REPO_DIR = Path(__file__).resolve().parent.parent
INFER_DIR = REPO_DIR / "infer"

# Make `import diffusion_sample` work from anywhere. diffusion_sample now
# lives at infer/diffusion_sample.py after the train/infer code split.
sys.path.insert(0, str(INFER_DIR))

# ----------------------------------------------------------------------------
# Defaults (mirror run_inference.sh); env vars override.
# ----------------------------------------------------------------------------
CKPT = os.environ.get(
    "CKPT",
    "/data/rsync/RWKV/DiffuRWKV/train/out/diff-L32-D4096-x070-blk32-ctx6144/rwkv-21.pth",
)
PROMPT = os.environ.get(
    "PROMPT",
    "System: You are a helpful assistant. After you finish thinking, please respond in a concise manner.\n\n"
    # "User: Briefly explain the 1 + 1 = ?.\n\nAssistant:",
    # "User: How to make a cup of coffee?\n\nAssistant:<think>",
    "User: Now I try to buy a car. My choice is Mazda CX-5 and Toyota Rav4. Which one should I choose?\n\nAssistant: <think>",
    # "User: Calculate 1 - 2 + 3 - 4 = ?.\n\nAssistant:",
    # "User: Give me a Python script that generates Fibonacci numbers.\n\nAssistant:",
    # "User: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat? You have 4 choices: 'A. dry palms', 'B. wet palms', 'C. palms covered with oil', 'D. palms covered with lotion'.\n\nAssistant:<think>"
    # "User: Which of the following statements best explains why magnets usually stick to a refrigerator door? You have 4 choices: 'A. The refrigerator door is smooth', 'B. The refrigerator door contains iron', 'C. The refrigerator door is a good conductor', 'D. The refrigerator door has electric wires in it'.\n\nAssistant:<think>"
)
GEN_LEN = int(os.environ.get("GEN_LEN", "2048"))
BLOCK_SIZE = int(os.environ.get("BLOCK_SIZE", "32"))
STEPS = int(os.environ.get("STEPS", "32"))

# Architecture (must match the loaded ckpt)
N_LAYER = int(os.environ.get("N_LAYER", "32"))
N_EMBD = int(os.environ.get("N_EMBD", "4096"))
HEAD_SIZE = int(os.environ.get("HEAD_SIZE", "64"))
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", "65536"))
MY_TESTING = os.environ.get("MY_TESTING", "x070")
D_DECAY_LORA = int(os.environ.get("D_DECAY_LORA", "128"))
D_AAA_LORA = int(os.environ.get("D_AAA_LORA", "128"))
D_MV_LORA = int(os.environ.get("D_MV_LORA", "96"))
D_GATE_LORA = int(os.environ.get("D_GATE_LORA", "480"))

OUT_FILE = Path(os.environ.get("OUT_FILE", REPO_DIR / "infer" / "sweep_results.md"))


# ----------------------------------------------------------------------------
# Presets — each preset only lists what differs from BASE_KNOBS.
# ----------------------------------------------------------------------------
BASE_KNOBS = dict(
    temperature=1.0,
    top_k=50,
    top_p=0.85,
    presence_penalty=0.0,
    count_penalty=0.0,
    penalty_decay=1.0,
    penalize_prompt=False,
    decode_strategy="threshold",
    conf_threshold=0.95,
    min_per_step=0,
)

PRESETS: dict[str, dict] = {
    # 1) Pure argmax — deterministic baseline. Often shows repetition pathology.
    # "greedy": dict(temperature=0.0, top_k=0, top_p=1.0),
    # 2) Light sampling, no penalty — what the model "naturally" produces.
    "natural": dict(temperature=1.0, top_k=0, top_p=1.0),
    "natural_topp": dict(temperature=1.0, top_k=0, top_p=0.9),
    # "natural_topp_temp": dict(temperature=0.8, top_k=0, top_p=0.9),
    # 3) Focused sampling, no penalty.
    "focused_no_penalty": dict(temperature=0.7, top_k=40, top_p=0.9),
    # 4) ChatRWKV canonical defaults.
    # "chatrwkv_canonical": dict(
    #     temperature=1.0, top_k=0, top_p=0.85,
    #     presence_penalty=0.4, count_penalty=0.4, penalty_decay=0.996,
    # ),
    # 5) The current run_inference.sh default — heavy presence, light count.
    # "current_default": dict(
    #     temperature=1.0, top_k=50, top_p=0.5,
    #     presence_penalty=2.0, count_penalty=0.2, penalty_decay=0.99,
    # ),
    # 8) Linear commit strategy — uses every denoise iteration regardless of
    #    confidence. Slower but matches LLaDA's original schedule.
    # "linear_decode": dict(
    #     temperature=0.7, top_k=40, top_p=0.9,
    #     decode_strategy="linear",
    # ),
    # 9) Stricter threshold — commits fewer positions per step, more iterations.
    # "strict_threshold": dict(
    #     temperature=0.7, top_k=40, top_p=0.9,
    #     decode_strategy="threshold", conf_threshold=0.99, min_per_step=1,
    # ),
    # 10) Loose threshold — fast commit, may rush past good completions.
    # "loose_threshold": dict(
    #     temperature=0.7, top_k=40, top_p=0.9,
    #     decode_strategy="threshold", conf_threshold=0.80,
    # ),
    # ---- Diffusion-scaled penalty band ------------------------------------
    # Why these are scaled down: ChatRWKV's `pres=0.4` was tuned for AR (one
    # token / step). Diffusion commits ~B parallel positions from the SAME
    # conditioning per block, so the per-token effective penalty is
    # AR_penalty * K_committed. With B=32 and ~K_eff ≈ 4-8 unique commits per
    # step, the diffusion-equivalent canonical lands around 0.05-0.1. Above
    # ~0.4 the EOS:common ratio inflates by exp(penalty) and triggers the
    # early-stop pathology (because EOS is the only token whose count stays 0).
    # 11) Diffusion-equivalent of chatrwkv canonical (0.4 / 8).
    # "diff_light": dict(
    #     temperature=0.7, top_k=40, top_p=0.9,
    #     presence_penalty=0.05, count_penalty=0.05, penalty_decay=0.9,
    # ),
    # "diff_light_high_temp": dict(
    #     temperature=1, top_k=40, top_p=0.9,
    #     presence_penalty=0.05, count_penalty=0.05, penalty_decay=0.9,
    # ),
    # "diff_light_pres": dict(
    #     temperature=1, top_k=40, top_p=0.9,
    #     presence_penalty=0.05, count_penalty=0.0, penalty_decay=0.9,
    # ),
    # "diff_light_cnt": dict(
    #     temperature=1, top_k=40, top_p=0.9,
    #     presence_penalty=0.0, count_penalty=0.05, penalty_decay=0.9,
    # ),
}


def fmt_knobs(knobs: dict) -> str:
    return (
        f"T={knobs['temperature']:g}  "
        f"top_k={knobs['top_k']}  "
        f"top_p={knobs['top_p']:g}  "
        f"pres={knobs['presence_penalty']:g}  "
        f"cnt={knobs['count_penalty']:g}  "
        f"decay={knobs['penalty_decay']:g}  "
        f"strat={knobs['decode_strategy']}  "
        f"conf={knobs['conf_threshold']:g}  "
        f"min/step={knobs['min_per_step']}"
    )


def main():
    selected = os.environ.get("PRESETS")
    if selected:
        names = [s.strip() for s in selected.split(",") if s.strip()]
        for n in names:
            if n not in PRESETS:
                print(f"ERROR: unknown preset {n!r}. Available: {', '.join(PRESETS)}",
                      file=sys.stderr)
                sys.exit(1)
    else:
        names = list(PRESETS)

    ckpt_abs = CKPT if Path(CKPT).is_absolute() else str(REPO_DIR / CKPT)
    if not Path(ckpt_abs).is_file():
        print(f"ERROR: ckpt not found: {ckpt_abs}", file=sys.stderr)
        sys.exit(1)

    print(f"sweeping {len(names)} preset(s) on prompt:")
    print(f"  {PROMPT!r}")
    print(f"ckpt: {ckpt_abs}")
    print(f"out:  {OUT_FILE}")
    print()

    # Import here so build_model's chdir/env-var setup is contained, and so
    # the import isn't paid before we've validated paths above. diffusion_sample
    # adds repo root to sys.path so `tokenizer` resolves to the bundled one.
    import diffusion_sample as ds

    tok = ds.RWKVTokenizer()
    mask_id = VOCAB_SIZE - 1

    model_args = SimpleNamespace(
        n_layer=N_LAYER,
        n_embd=N_EMBD,
        dim_att=N_EMBD,
        dim_ffn=int((N_EMBD * 3.5) // 32 * 32),
        head_size=HEAD_SIZE,
        vocab_size=VOCAB_SIZE,
        ctx_len=4096,
        my_testing=MY_TESTING,
        grad_cp=0,
        weight_decay=0.0,
        lr_init=0.0, lr_final=0.0, betas=(0.9, 0.99), adam_eps=1e-18,
        layerwise_lr=0, my_pile_stage=0, train_stage=0,
        diffusion_mode=0,
        d_decay_lora=D_DECAY_LORA,
        d_aaa_lora=D_AAA_LORA,
        d_mv_lora=D_MV_LORA,
        d_gate_lora=D_GATE_LORA,
    )

    print("[load] building model (one-time cost)...")
    t_load = time.time()
    model = ds.build_model(ckpt_abs, model_args)
    print(f"[load] done in {time.time() - t_load:.1f}s\n")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        f.write(f"# DiffuRWKV inference sweep\n\n")
        f.write(f"- **ckpt**: `{ckpt_abs}`\n")
        f.write(f"- **gen_len** = {GEN_LEN}, **block_size** = {BLOCK_SIZE}, **steps** = {STEPS}\n")
        f.write(f"- **prompt**:\n\n```\n{PROMPT}\n```\n\n")
        f.flush()

        for i, name in enumerate(names, 1):
            knobs = {**BASE_KNOBS, **PRESETS[name]}
            cfg_line = fmt_knobs(knobs)
            print(f"[{i}/{len(names)}] {name}    {cfg_line}")

            t0 = time.time()
            text = ""
            finish_reason = ""
            n_completion = 0
            ok = True
            err: BaseException | None = None
            try:
                text, finish_reason, n_completion = ds.run_one(
                    model, tok, mask_id, VOCAB_SIZE,
                    PROMPT, GEN_LEN, STEPS, BLOCK_SIZE,
                    knobs["temperature"], knobs["top_k"], knobs["top_p"],
                    knobs["decode_strategy"], knobs["conf_threshold"], knobs["min_per_step"],
                    knobs["presence_penalty"], knobs["count_penalty"], knobs["penalty_decay"],
                    knobs["penalize_prompt"],
                    verbose=False,
                )
            except KeyboardInterrupt:
                print("interrupted; partial results written to", OUT_FILE)
                return
            except BaseException as e:
                ok = False
                err = e
            elapsed = time.time() - t0

            status = (
                f"OK [{finish_reason}, {n_completion} tok]"
                if ok else f"FAIL: {type(err).__name__}"
            )
            print(f"    -> {status}  ({elapsed:.1f}s)")

            f.write(f"## {i}. `{name}`  ({status}, {elapsed:.1f}s)\n\n")
            f.write(f"```\n{cfg_line}\n```\n\n")
            if ok:
                f.write(f"```\n{text}\n```\n\n")
            else:
                f.write(f"```\n{err!r}\n```\n\n")
            f.write("---\n\n")
            f.flush()

    print(f"\nDONE. results -> {OUT_FILE}")


if __name__ == "__main__":
    main()
