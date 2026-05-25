"""Pre-download every HF dataset our eval tasks need into the local HF cache.

Run this on a host that can reach huggingface.co (e.g. Anvil), then rsync
~/.cache/huggingface/{hub,datasets,modules} to the GPU host's $HF_HOME (which
will run lm-eval offline). The mapping below mirrors what each lm-eval task in
configs/default.yaml does internally via `datasets.load_dataset(...)`.

GPQA is gated -- requires HF login + accepting the license at
https://huggingface.co/datasets/Idavidrein/gpqa before this script can fetch
it. Set the token via `huggingface-cli login` or the HF_TOKEN env var.

Usage:
    .venv/bin/python eval/prefetch_datasets.py            # all
    .venv/bin/python eval/prefetch_datasets.py gsm8k mbpp # subset by friendly name
    SKIP_GATED=1 .venv/bin/python eval/prefetch_datasets.py  # skip GPQA
"""

import os
import sys

# (friendly_name, dataset_path, list-of-configs-or-[None])
DATASETS = [
    # already-cached commonly; harmless to re-run (HF Hub skips downloads).
    ("mmlu", "cais/mmlu", None),  # ALL_SUBJECTS resolved at load_dataset
    # ("arc",          "allenai/ai2_arc",             ["ARC-Challenge", "ARC-Easy"]),
    # ("piqa",         "baber/piqa",                  [None]),
    # ("race",         "EleutherAI/race",             ["high"]),
    # # likely missing on first run.
    # ("gsm8k",        "openai/gsm8k",                ["main"]),
    # ("humaneval",    "openai/openai_humaneval",     [None]),
    # ("mbpp",         "google-research-datasets/mbpp", ["full"]),
    # ("minerva_math", "EleutherAI/hendrycks_math",   [
    #     "algebra", "counting_and_probability", "geometry",
    #     "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    # ]),
    # # gated -- skipped if HF account hasn't accepted the license.
    # ("gpqa",         "Idavidrein/gpqa",             ["gpqa_main"]),
]

# MMLU has 57 subjects; pulling all of them via load_dataset with no config
# downloads the union, but the offline read still needs every config to be
# resolvable from the cache. Easiest reliable way is to list them.
_MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def fetch_one(path, name):
    import datasets

    label = f"{path}" + (f":{name}" if name else "")
    try:
        ds = datasets.load_dataset(path, name) if name else datasets.load_dataset(path)
        print(f"  OK  {label}  splits={list(ds.keys())}")
        return True
    except Exception as e:
        msg = str(e).splitlines()[0][:200]
        print(f"  FAIL {label}  -> {type(e).__name__}: {msg}")
        return False


def main():
    skip_gated = os.environ.get("SKIP_GATED", "0") == "1"
    want = set(sys.argv[1:]) if len(sys.argv) > 1 else None

    failures = []
    for friendly, path, configs in DATASETS:
        if want is not None and friendly not in want:
            continue
        if friendly == "gpqa" and skip_gated:
            print(f"[skip] {friendly} (SKIP_GATED=1)")
            continue
        if friendly == "mmlu" and (configs is None):
            configs = _MMLU_SUBJECTS

        print(f"\n[{friendly}] {path}")
        for cfg in configs or [None]:
            if not fetch_one(path, cfg):
                failures.append(f"{path}:{cfg}")

    print("\n================================================================")
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
        # GPQA-gated failure is expected unless the user authenticated; soft exit
        only_gpqa = all("Idavidrein/gpqa" in f for f in failures)
        sys.exit(0 if only_gpqa else 1)
    print("All datasets cached.")


if __name__ == "__main__":
    main()
