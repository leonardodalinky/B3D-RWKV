#!/bin/bash
###############################################################################
# DiffuRWKV evaluation launcher (API mode).
#
# Calls the deployed OpenAI-compatible server (infer/serve) over HTTP instead
# of loading the model in-process. The model stays resident on the GPU host;
# this launcher fires lm-eval requests at it, one task at a time, so each
# task can have its own num_fewshot / chat template / generation budget.
#
# REQUIREMENT: the server must already be running on the GPU host.
#     ssh Hetao-gpu
#     cd /data/rsync/RWKV/DiffuRWKV
#     CKPT=/path/to/rwkv-N.pth python -m infer.serve   # listens on :8000
# This script is then run on the same GPU host so localhost:8000 resolves.
#
# Quick start (on Hetao-gpu):
#     bash eval/run_eval.sh                                    # use defaults
#     TASKS=gsm8k,mbpp bash eval/run_eval.sh                   # subset of $CONFIG
#     CONFIG=eval/configs/quick.yaml bash eval/run_eval.sh     # different task list
#     LIMIT=20 bash eval/run_eval.sh                           # smoke test (20 docs/task)
#     RESUME=1 OUT_DIR=eval/results/<existing-dir> bash eval/run_eval.sh
#                                                              # skip already-finished tasks
#     API_URL=http://localhost:8000 bash eval/run_eval.sh      # different server port
#
# Because the API is chat-completions-only (no logprobs), every task here MUST
# use ``output_type: generate_until``. The default.yaml swaps loglikelihood
# benchmarks (MMLU, ARC, GPQA, ...) to their generative variants.
###############################################################################
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Activate the project venv so `lm-eval` and `python` resolve to the same env.
if [ -f "$REPO_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_DIR/.venv/bin/activate"
fi

# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------
# Base URL of the served model. Default targets the local server started via
# `python -m infer.serve`. The chat-completions endpoint is appended later.
API_URL="${API_URL:-http://localhost:8000}"
API_MODEL="${API_MODEL:-diffurwkv}"          # /v1/models id; from MODEL_NAME on server

# How many in-flight requests to keep against the server. The server has a
# Semaphore(1) so it processes them sequentially anyway; setting >1 only helps
# overlap HTTP / payload assembly. Keep low to avoid timeouts on long tasks.
NUM_CONCURRENT="${NUM_CONCURRENT:-1}"

# Per-request timeout in seconds. Long-CoT tasks (minerva_math, gpqa) can take
# 1+ minutes per response on a 7.2B diffusion model -- raise if you see
# httpx.ReadTimeout in the logs.
API_TIMEOUT="${API_TIMEOUT:-600}"

# Used as a tag in the output dir name so multiple eval runs against different
# ckpts don't overwrite each other. The launcher can't see CKPT (that's the
# server's secret), so we ask the caller to pass it in.
RUN_TAG="${RUN_TAG:-${API_MODEL}}"

# YAML defining tasks + per-task num_fewshot. See eval/configs/default.yaml.
CONFIG="${CONFIG:-eval/configs/default.yaml}"

# Output dir for per-task JSON results. Default groups runs by tag + timestamp.
OUT_DIR="${OUT_DIR:-eval/results/${RUN_TAG}-$(date +%Y%m%dT%H%M%S)}"

# Optional: comma-separated subset of tasks to run (intersect with $CONFIG).
TASKS_OVERRIDE="${TASKS:-}"

# Optional: cap docs per task for smoke-tests. Empty = run full eval set.
LIMIT="${LIMIT:-}"

# Optional: extra args appended to every lm-eval call (e.g. --log_samples).
EXTRA_ARGS="${EXTRA_ARGS:-}"

# If RESUME=1, skip any task whose JSON already exists under $OUT_DIR.
RESUME="${RESUME:-0}"

# If LOG_SAMPLES=1 (default on), lm-eval writes a per-doc samples_<task>_<ts>.jsonl
# alongside each task's aggregate JSON. Each line has the prompt, model
# response, target, and per-filter exact_match score -- required for any
# kind of error-pattern analysis. Set LOG_SAMPLES=0 to skip and shave a
# small amount of IO + result-dir size on full eval runs.
LOG_SAMPLES="${LOG_SAMPLES:-1}"

# pass@k for humaneval / mbpp. PASS_K=1 (default) keeps the standard greedy
# behavior. PASS_K>1 swaps the task names to humaneval_passk / mbpp_passk
# and materializes their yamls from eval/tasks/passk_template/. The pass-k
# tasks force do_sample=true + temperature=0.8 in the yaml (sampling is
# REQUIRED for pass@k -- greedy would produce K identical predictions); set
# PASSK_TEMPERATURE to override the temperature without touching the yaml.
PASS_K="${PASS_K:-5}"
PASSK_TEMPERATURE="${PASSK_TEMPERATURE:-}"   # empty -> keep yaml default (0.8)

# Generation cap for free-form tasks. Per-task overrides in YAML take precedence
# (set max_gen_toks: N for the task). 8192 fits think-mode CoT + final answer;
# drop to 1024 if you turn REASONING_EFFORT off.
MAX_GEN_TOKS="${MAX_GEN_TOKS:-8192}"

# server-side think-mode switch. Maps to OpenAI's `reasoning_effort` field
# (DiffuRWKV server reads it: see infer/serve/engine.py:147). Anything other
# than "minimal" / None primes the assistant with "<think>" and the model
# emits a reasoning span before the final answer. The server splits on
# "</think>" and only returns the post-think content to lm-eval, so the
# grader is unaffected by reasoning content. Set to empty to disable.
REASONING_EFFORT="${REASONING_EFFORT:-medium}"

# Path that ships custom task YAMLs (e.g. arc_easy_chat) we wrote in this repo.
# Empty -> skip --include_path.
INCLUDE_PATH="${INCLUDE_PATH:-eval/tasks}"

# HuggingFace cache + offline mode. Hetao-gpu can't reach huggingface.co, so
# every dataset has to live in HF_HOME/{hub,datasets}. Populate via
# eval/prefetch_datasets.py on a host with internet, then rsync the cache
# to HF_PERSISTENT (default /data/.cache/huggingface on the quarkfs share).
#
# Runtime caveat: the persistent share (quarkfs) doesn't follow POSIX
# unlink-then-flock semantics, which the `datasets` library's FileLock
# relies on -- builder __init__ crashes with FileNotFoundError on every
# load_dataset(). So we copy the cache to a local-fs working location
# (default /root/.cache/huggingface, container overlay/xfs) and point
# HF_HOME there. The copy is ~200 MB and a few seconds; the persistent
# share keeps the master copy across container restarts.
HF_PERSISTENT="${HF_PERSISTENT:-/data/.cache/huggingface}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
# `evaluate` library (used by humaneval/mbpp for code_eval) has its own
# offline flag, separate from HF_HUB_OFFLINE.
export HF_EVALUATE_OFFLINE="${HF_EVALUATE_OFFLINE:-1}"
# Acknowledge that humaneval/mbpp execute the model's generated code; without
# this the code_eval metric raises ValueError. Paired with the per-task
# `unsafe_code: true` flag in default.yaml (which adds --confirm_run_unsafe_code).
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"

# If HF_HOME doesn't have the cache yet but the persistent share does,
# materialize it locally. cp -ru is idempotent: re-runs don't re-copy.
# modules/ covers `evaluate` metric scripts (code_eval for humaneval/mbpp).
if [ -d "$HF_PERSISTENT/hub" ]; then
    mkdir -p "$HF_HOME"
    for sub in hub datasets modules; do
        if [ -d "$HF_PERSISTENT/$sub" ] && [ ! -d "$HF_HOME/$sub" ]; then
            echo "[hf-cache] materializing $HF_PERSISTENT/$sub -> $HF_HOME/$sub"
            cp -ru "$HF_PERSISTENT/$sub" "$HF_HOME/" 2>/dev/null || true
        fi
    done
fi

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [ ! -d "$HF_HOME/hub" ]; then
    echo "ERROR: HF_HOME=$HF_HOME has no hub/ subdirectory." >&2
    echo "  HF_PERSISTENT=$HF_PERSISTENT also missing -- did you forget to rsync" >&2
    echo "  the prefetched cache? See eval/prefetch_datasets.py + eval/_rsync_hf_cache.sh." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Pass@k task materialization. When PASS_K>1, expand
# eval/tasks/passk_template/<task>_passk.yaml.tmpl into
# eval/tasks/_dyn/<task>_passk/<task>_passk.yaml. lm-eval's --include_path
# recurses (root.glob("**/*.yaml")), so dropping these under eval/tasks/_dyn
# picks them up automatically. Each task lives in its own subdir because both
# task yamls reference `!function utils.foo` and the two upstream utils.py
# files have conflicting `build_predictions` definitions; subdirs let each
# task have its own utils.py.
# ---------------------------------------------------------------------------
DYN_DIR="$REPO_DIR/eval/tasks/_dyn"

# pass@k materialization
if [ "$PASS_K" -gt 1 ] 2>/dev/null; then
    TMPL_DIR="$REPO_DIR/eval/tasks/passk_template"
    rm -rf "$DYN_DIR/humaneval_passk" "$DYN_DIR/mbpp_passk"
    mkdir -p "$DYN_DIR/humaneval_passk" "$DYN_DIR/mbpp_passk"

    # humaneval: yaml + symlink to upstream's utils.py (build_predictions /
    # pass_at_k are both there and we reuse them verbatim).
    sed "s/__PASS_K__/$PASS_K/g" "$TMPL_DIR/humaneval_passk.yaml.tmpl" \
        > "$DYN_DIR/humaneval_passk/humaneval_passk.yaml"
    UPSTREAM_HE_UTILS="$REPO_DIR/.venv/lib/python3.11/site-packages/lm_eval/tasks/humaneval/utils.py"
    ln -sf "$UPSTREAM_HE_UTILS" "$DYN_DIR/humaneval_passk/utils.py"

    # mbpp: yaml + our own utils.py (re-exports upstream + adds pass_at_k).
    sed "s/__PASS_K__/$PASS_K/g" "$TMPL_DIR/mbpp_passk.yaml.tmpl" \
        > "$DYN_DIR/mbpp_passk/mbpp_passk.yaml"
    cp "$TMPL_DIR/mbpp_utils.py" "$DYN_DIR/mbpp_passk/utils.py"

    echo "[passk] materialized humaneval_passk + mbpp_passk yamls with k=$PASS_K"
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

# Probe the server up-front so we don't burn time queueing requests against a
# down endpoint. Returns 200 with a small JSON listing the loaded model.
if ! curl -sf --max-time 5 "${API_URL}/v1/models" > /dev/null; then
    echo "ERROR: cannot reach API at $API_URL/v1/models" >&2
    echo "Is the server running? On the GPU host:" >&2
    echo "    cd /data/rsync/RWKV/DiffuRWKV && CKPT=... python -m infer.serve" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
echo "================================================================"
echo "DiffuRWKV eval (API mode)"
echo "  api:     $API_URL  (model=$API_MODEL)"
echo "  config:  $CONFIG"
echo "  output:  $OUT_DIR"
echo "  HF_HOME: $HF_HOME  (persistent=$HF_PERSISTENT, offline=$HF_HUB_OFFLINE)"
echo "  think:   reasoning_effort=${REASONING_EFFORT:-(none)}  max_gen_toks=$MAX_GEN_TOKS"
echo "  samples: log_samples=$LOG_SAMPLES"
[ "$PASS_K" -gt 1 ] 2>/dev/null && \
    echo "  pass@k:  PASS_K=$PASS_K  (humaneval/mbpp -> *_passk variants)"
[ -n "$TASKS_OVERRIDE" ] && echo "  tasks:   $TASKS_OVERRIDE (subset)"
[ -n "$LIMIT" ]          && echo "  limit:   $LIMIT docs/task (smoke mode)"
[ "$RESUME" = "1" ]      && echo "  resume:  on (skip tasks with existing *.json)"
echo "================================================================"

# lm-eval's local-chat-completions wants the FULL chat-completions URL.
CHAT_URL="${API_URL%/}/v1/chat/completions"

# Common model_args for every per-task lm-eval call.
#   model=diffurwkv             id the server returns from /v1/models
#   base_url=...                full URL of the chat-completions endpoint
#   num_concurrent=1            sequential by default (server is single-slot)
#   timeout=600                 per-request HTTP timeout in seconds
#   tokenizer_backend=None      no local tokenizer -> chat history JSON-passes
#                               through, server does its own G1x templating
#   tokenized_requests=False    send strings, not token ids (paired with above)
#   max_gen_toks=$MAX_GEN_TOKS  default cap (per-task YAML override allowed)
MODEL_ARGS="model=$API_MODEL"
MODEL_ARGS+=",base_url=$CHAT_URL"
MODEL_ARGS+=",num_concurrent=$NUM_CONCURRENT"
MODEL_ARGS+=",timeout=$API_TIMEOUT"
MODEL_ARGS+=",tokenizer_backend=None"
MODEL_ARGS+=",tokenized_requests=False"
MODEL_ARGS+=",max_gen_toks=$MAX_GEN_TOKS"

# gen_kwargs are the ONLY model-arg style fields lm-eval's local-chat-completions
# actually puts into the JSON request body (via `**gen_kwargs` splat at the
# end of LocalChatCompletion._create_payload). The constructor kwargs above
# (max_gen_toks, etc.) are read by the lm-eval wrapper itself but not sent.
# So reasoning_effort -- which the DiffuRWKV server reads off the JSON
# top-level via Pydantic extra='allow' -- has to come through --gen_kwargs.
#
# We also force max_tokens here: many built-in task YAMLs hardcode tiny
# max_gen_toks (humaneval=1024, mbpp_instruct=256, ours had 256). CLI
# gen_kwargs are merged into the per-task generation_kwargs with `update=True`
# (lm-eval evaluator.py:311), so this override wins; and inside
# LocalChatCompletion._create_payload the `max_tokens` field takes precedence
# over `max_gen_toks` (openai_completions.py:192).
GEN_KWARGS="max_tokens=$MAX_GEN_TOKS"
if [ -n "$REASONING_EFFORT" ]; then
    GEN_KWARGS+=",reasoning_effort=$REASONING_EFFORT"
fi
# Only inject temperature if the user explicitly set PASSK_TEMPERATURE.
# Otherwise leave the yaml's value alone (humaneval/mbpp yamls default to 0
# which becomes greedy, but pass-k yamls hardcode 0.8 -- that's what we want).
if [ -n "$PASSK_TEMPERATURE" ]; then
    GEN_KWARGS+=",temperature=$PASSK_TEMPERATURE"
fi

# ---------------------------------------------------------------------------
# Parse YAML -> "task fewshot unsafe multiturn max_gen" lines
# (avoids adding a yq dependency).
# ---------------------------------------------------------------------------
TASKS_LIST=$(python - <<PY
import sys, yaml
with open("$CONFIG") as f:
    cfg = yaml.safe_load(f) or {}
for k, v in cfg.items():
    v = v or {}
    fewshot   = int(v.get("num_fewshot", 0))
    unsafe    = 1 if v.get("unsafe_code", False) else 0
    multiturn = 1 if v.get("fewshot_as_multiturn", False) else 0
    max_gen   = int(v.get("max_gen_toks", 0))   # 0 = use launcher default
    # Per-task gen_kwargs (top_p / temperature / presence_penalty / etc.)
    # Encoded as comma-joined key=value pairs; "-" sentinel = absent.
    # Whitespace is forbidden inside values -- awk field-split would break.
    gk = v.get("gen_kwargs", {}) or {}
    gk_str = ",".join(f"{kk}={vv}" for kk, vv in gk.items()) if gk else "-"
    if " " in gk_str:
        raise SystemExit(f"task {k}: gen_kwargs values cannot contain spaces: {gk_str!r}")
    print(k, fewshot, unsafe, multiturn, max_gen, gk_str)
PY
)

# ---------------------------------------------------------------------------
# Run each task
# ---------------------------------------------------------------------------
FAILED_TASKS=()
SUCCESS_TASKS=()
SKIPPED_TASKS=()

include_args=()
if [ -n "$INCLUDE_PATH" ] && [ -d "$INCLUDE_PATH" ]; then
    include_args+=("--include_path" "$INCLUDE_PATH")
fi

while IFS= read -r line; do
    [ -z "$line" ] && continue
    task=$(echo         "$line" | awk '{print $1}')
    fewshot=$(echo      "$line" | awk '{print $2}')
    unsafe=$(echo       "$line" | awk '{print $3}')
    multiturn=$(echo    "$line" | awk '{print $4}')
    task_max_gen=$(echo "$line" | awk '{print $5}')
    task_gen_kwargs=$(echo "$line" | awk '{print $6}')   # "-" if none

    # Honor the TASKS=... override (run only listed tasks). Match against the
    # default.yaml-level name (humaneval / mbpp) BEFORE the pass-k swap below,
    # so users say `TASKS=humaneval` regardless of PASS_K.
    if [ -n "$TASKS_OVERRIDE" ]; then
        if ! echo ",$TASKS_OVERRIDE," | grep -q ",$task,"; then
            continue
        fi
    fi

    # Swap humaneval / mbpp -> their pass@k variants when PASS_K>1. The
    # _dyn yamls were materialized above; lm-eval --include_path eval/tasks
    # finds them recursively under _dyn/.
    if [ "$PASS_K" -gt 1 ] 2>/dev/null; then
        case "$task" in
            humaneval) task="humaneval_passk" ;;
            mbpp)      task="mbpp_passk"      ;;
        esac
    fi

    out_path="$OUT_DIR/${task}.json"
    log_file="$OUT_DIR/${task}.log"

    # Resume: lm-eval writes <task>_<lm-eval-ts>.json (it adds its own ts suffix
    # to whatever we pass via --output_path), or sometimes a subdir under
    # <task>/. Treat any of these as "done".
    if [ "$RESUME" = "1" ]; then
        if [ -f "$out_path" ] \
           || compgen -G "$OUT_DIR/${task}_*.json" >/dev/null \
           || ls "$OUT_DIR/${task}/" >/dev/null 2>&1; then
            echo "[skip] $task -- output already exists under $OUT_DIR"
            SKIPPED_TASKS+=("$task")
            continue
        fi
    fi

    # Every task here is generation-mode + chat-API, so chat templating is
    # mandatory: lm-eval needs to package each context as a messages list
    # so the server's messages_to_prompt can render the G1x template.
    extra_args=("--apply_chat_template")
    # Pass the multiturn flag EXPLICITLY (true or false). lm-eval CLI uses
    # default=argparse.SUPPRESS, so not passing it leaves cfg.fewshot_as_multiturn=None,
    # which the config-resolver then COERCES to True whenever apply_chat_template
    # is on (evaluate_config.py:306-308). That silently overrode our YAML
    # `fewshot_as_multiturn: false`. Passing the value explicitly avoids the trap.
    if [ "$multiturn" = "1" ]; then
        extra_args+=("--fewshot_as_multiturn" "true")
    else
        extra_args+=("--fewshot_as_multiturn" "false")
    fi
    if [ "$unsafe" = "1" ]; then
        extra_args+=("--confirm_run_unsafe_code")
    fi
    if [ -n "$LIMIT" ]; then
        extra_args+=("--limit" "$LIMIT")
    fi
    if [ "$LOG_SAMPLES" = "1" ]; then
        extra_args+=("--log_samples")
    fi
    if [ -n "$EXTRA_ARGS" ]; then
        # shellcheck disable=SC2206
        extra_args+=($EXTRA_ARGS)
    fi

    # Per-task max_gen_toks override (rebuild MODEL_ARGS for this task only).
    task_model_args="$MODEL_ARGS"
    if [ "$task_max_gen" != "0" ]; then
        task_model_args=$(echo "$task_model_args" | sed -E "s/max_gen_toks=[0-9]+/max_gen_toks=$task_max_gen/")
    fi

    echo
    echo "----------------------------------------------------------------"
    flags="num_fewshot=$fewshot"
    [ "$unsafe"    = "1" ] && flags="$flags unsafe_code"
    [ "$multiturn" = "1" ] && flags="$flags multiturn"
    [ "$task_max_gen" != "0" ] && flags="$flags max_gen=$task_max_gen"
    echo "[$task] $flags"
    echo "----------------------------------------------------------------"

    # Append per-task gen_kwargs (from default.yaml) onto the global GEN_KWARGS.
    # Per-task entries come last so they win on conflict with --gen_kwargs merge
    # semantics (later keys override earlier ones in `simple_parse_args_string`).
    task_gk="$GEN_KWARGS"
    if [ "$task_gen_kwargs" != "-" ] && [ -n "$task_gen_kwargs" ]; then
        task_gk="$task_gk,$task_gen_kwargs"
    fi
    gen_args=()
    if [ -n "$task_gk" ]; then
        # shellcheck disable=SC2206
        gen_args+=(--gen_kwargs "$task_gk")
    fi

    # We invoke lm-eval via eval/lm_eval_wrapper.py (NOT bare `lm-eval`) so the
    # LocalChatCompletion.parse_generations monkey-patch in the wrapper is
    # applied before any task runs. The patch concatenates the server's
    # `reasoning_content` (pre-</think> span) into the text filters scan,
    # which is required for grading responses where the model emits its
    # final answer inside the think span (or runs out of budget before
    # </think>, leaving content="").
    if python "$REPO_DIR/eval/lm_eval_wrapper.py" \
        --model local-chat-completions \
        --model_args "$task_model_args" \
        --tasks "$task" \
        --num_fewshot "$fewshot" \
        --batch_size 1 \
        --output_path "$out_path" \
        "${gen_args[@]}" \
        "${include_args[@]}" \
        "${extra_args[@]}" \
        2>&1 | tee "$log_file"
    then
        SUCCESS_TASKS+=("$task")
    else
        FAILED_TASKS+=("$task")
        echo "[!] Task $task FAILED -- see $log_file" >&2
    fi
done <<< "$TASKS_LIST"

# ---------------------------------------------------------------------------
# Aggregate summary -- pluck the top-line metric from each task JSON.
# ---------------------------------------------------------------------------
SUMMARY="$OUT_DIR/summary.json"
python - "$OUT_DIR" "$SUMMARY" <<'PY' || true
import json, os, sys, glob

out_dir = sys.argv[1]
summary_path = sys.argv[2]
rows = {}

for json_path in glob.glob(os.path.join(out_dir, "**", "*.json"), recursive=True):
    if os.path.basename(json_path) == "summary.json":
        continue
    try:
        with open(json_path) as f:
            data = json.load(f)
    except Exception as e:
        rows[os.path.relpath(json_path, out_dir)] = {"error": str(e)}
        continue
    results = data.get("results") or {}
    for task, metrics in results.items():
        if isinstance(metrics, dict):
            clean = {k: v for k, v in metrics.items()
                     if not k.startswith("alias") and isinstance(v, (int, float))}
            if clean:
                rows[task] = clean

with open(summary_path, "w") as f:
    json.dump(rows, f, indent=2, sort_keys=True)
print(f"\n[summary] {len(rows)} tasks scored -> {summary_path}")
for task, metrics in sorted(rows.items()):
    pieces = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
              for k, v in metrics.items()]
    print(f"  {task:40s}  {' '.join(pieces)}")
PY

echo
echo "================================================================"
echo "Done. Results in $OUT_DIR"
echo "  succeeded: ${#SUCCESS_TASKS[@]}  (${SUCCESS_TASKS[*]:-})"
echo "  failed:    ${#FAILED_TASKS[@]}   (${FAILED_TASKS[*]:-})"
[ "${#SKIPPED_TASKS[@]}" -gt 0 ] && \
    echo "  skipped:   ${#SKIPPED_TASKS[@]}   (${SKIPPED_TASKS[*]:-})  (resume)"
echo "================================================================"

if [ ${#FAILED_TASKS[@]} -gt 0 ]; then
    exit 1
fi
