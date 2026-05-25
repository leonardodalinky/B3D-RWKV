#!/bin/bash
# One-shot helper: rsync the prefetched HF cache from Anvil's
# ~/.cache/huggingface to Hetao-cpu:/data/.cache/huggingface. SSH-via-bastion
# is slow (~1 MB/min), so this is meant to run detached in the background.
#
# Started by run_eval.sh? No -- this is invoked once manually:
#     nohup bash eval/_rsync_hf_cache.sh > /tmp/rsync_hf.log 2>&1 &
#
# Watch with:  tail -f /tmp/rsync_hf.log
set -euo pipefail
cd ~/.cache/huggingface

DATASETS=(
    cais--mmlu
    allenai--ai2_arc
    baber--piqa
    EleutherAI--race
    openai--gsm8k
    openai--openai_humaneval
    google-research-datasets--mbpp
    EleutherAI--hendrycks_math
)

echo "[$(date)] starting hub/ rsync ($(du -ch hub/datasets--{cais--mmlu,allenai--ai2_arc,baber--piqa,EleutherAI--race,openai--gsm8k,openai--openai_humaneval,google-research-datasets--mbpp,EleutherAI--hendrycks_math} 2>/dev/null | tail -1 | awk '{print $1}'))"
for d in "${DATASETS[@]}"; do
    src="hub/datasets--$d"
    [ -d "$src" ] || { echo "  skip missing $src"; continue; }
    echo "[$(date)] rsync $src"
    rsync -rlptDvz --partial --inplace "$src" Hetao-cpu:/data/.cache/huggingface/hub/ 2>&1 \
        | grep -vE "^(sending|sent|total)" | tail -5
done

echo
echo "[$(date)] starting datasets/ rsync"
for d in "${DATASETS[@]}"; do
    src="datasets/${d/--/___}"
    [ -d "$src" ] || { echo "  skip missing $src"; continue; }
    echo "[$(date)] rsync $src"
    rsync -rlptDvz --partial --inplace "$src" Hetao-cpu:/data/.cache/huggingface/datasets/ 2>&1 \
        | grep -vE "^(sending|sent|total)" | tail -5
done

echo
echo "[$(date)] all done. remote sizes:"
ssh Hetao-cpu "du -sh /data/.cache/huggingface/hub/datasets--* /data/.cache/huggingface/datasets/*___* 2>/dev/null" | sort
