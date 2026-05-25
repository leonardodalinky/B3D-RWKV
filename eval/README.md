# DiffuRWKV evaluation

## Server-side setup
Please refer to the [serving README](infer/serve/README.md) for details on setting up the server.
Start the server first; `run_eval.sh` aborts if `${API_URL}/v1/models`
isn't reachable.

## Common commands

```bash
bash eval/run_eval.sh                                      # full eval
TASKS=mmlu_generative_new,gsm8k_cot bash eval/run_eval.sh  # subset
LIMIT=20 TASKS=arc_easy_chat       bash eval/run_eval.sh   # smoke test
API_URL=http://localhost:18000     bash eval/run_eval.sh   # remote server
RESUME=1 OUT_DIR=eval/results/...  bash eval/run_eval.sh   # resume crash
RUN_TAG=rwkv-21                    bash eval/run_eval.sh   # tag output dir
```

Model architecture flags (`N_LAYER`, `N_EMBD`, ...) live on the server side
at startup — this launcher doesn't need to know.

## Gotchas

- **Single-slot server.** `infer/serve` holds a `Semaphore(1)`; raising
  `NUM_CONCURRENT` just queues server-side. MMLU's 57 subjects run
  sequentially — expect multi-hour runs, or use `LIMIT=200` for a snapshot.
- **Long CoT timeouts.** Launcher uses `timeout=600`; raise `API_TIMEOUT`
  if minerva_math / gpqa hit `httpx.ReadTimeout`.
- **`max_gen_toks=8192` everywhere** to leave room for think-mode reasoning.
  Drop to 64–256 if you switch back to no-think eval.
- **HumanEval / MBPP execute model output.** `unsafe_code: true` adds
  `--confirm_run_unsafe_code`. Run in a container if untrusted.
- **PIQA / RACE chat variants are ours**, graded by `exact_match` on the
  answer letter — not directly comparable to LLaDA / RWKV paper numbers.

## Adding a task

1. Find it: `lm-eval --tasks list | grep -i <keyword>`. Must be
   `output_type: generate_until`, else write a chat variant ([arc_easy_chat.yaml](tasks/arc_easy_chat.yaml)
   is the template).
2. Add a line to `configs/default.yaml`:
   ```yaml
   new_task: { num_fewshot: 0, max_gen_toks: 8192 }
   ```
   Knobs: `unsafe_code`, `fewshot_as_multiturn`, `gen_kwargs` (no
   whitespace in values — awk parses it).
