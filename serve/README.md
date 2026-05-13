# DiffuRWKV OpenAI-compatible server

A small FastAPI app that wraps a single loaded DiffuRWKV model behind
`POST /v1/chat/completions` so any OpenAI-SDK client (or a plain `curl`)
can hit it like a standard LLM endpoint.

## Design constraints (intentional)

- **One request at a time.** The engine holds an `asyncio.Semaphore(1)`;
  concurrent HTTP requests queue at the lock. We do NOT batch.
- **One worker process.** `uvicorn.run(..., workers=1)` — the semaphore
  and the loaded GPU model are per-process state, so spawning more
  workers would each load their own model (OOM) and silently break the
  "max-1-concurrent" guarantee.
- **No streaming.** Clients sending `stream: true` get a 400.
- **No auth.** Trusted-network only. If you need to expose this beyond
  localhost, put it behind a reverse proxy with auth, or wrap a bearer
  check in middleware.

## What's supported

| OpenAI field            | Maps to                              |
|-------------------------|--------------------------------------|
| `model`                 | echoed (no model switching)          |
| `messages`              | flattened to G1x template, primed with `<think>` |
| `temperature`           | `temperature`                        |
| `top_p`                 | `top_p`                              |
| `presence_penalty`      | `presence_penalty`                   |
| `frequency_penalty`     | DiffuRWKV `count_penalty`            |
| `max_tokens` / `max_completion_tokens` | `gen_len`, capped at `MAX_TOKENS` |
| `n`                     | must be 1 (else 400)                 |
| `stream`                | must be false (else 400)             |
| `reasoning_effort`      | think-mode switch (see below)        |

### Think mode via `reasoning_effort`

The model supports two priming styles for the assistant turn:

| `reasoning_effort` value           | Prompt ends with     | Behavior |
|------------------------------------|----------------------|----------|
| omitted (default) or `"minimal"`   | `Assistant:`         | direct answer, no `<think>` span |
| `"low"` / `"medium"` / `"high"` / any other string | `Assistant: <think>` | model reasons inside `<think>...</think>`, then answers |

When think mode is on, the server splits the model output on `</think>`
for you:

- `choices[0].message.content` → the answer (text after `</think>`)
- `choices[0].message.reasoning_content` → the reasoning span (text before `</think>`)

If the model runs out of budget before emitting `</think>`, the whole
partial output goes into `reasoning_content` and `content` is `""`.
For no-think requests `reasoning_content` is `null`.

(Format follows the DeepSeek / vLLM convention. OpenAI's stock SDK
accepts the extra field silently; older clients that strictly type-check
the response may need to access it via `model.model_extra` or `dict(...)`.)

### DiffuRWKV-only knobs (pass as extra fields in the request body)

```json
{
  "model": "diffurwkv",
  "messages": [...],
  "top_k": 40,
  "conf_threshold": 0.95,
  "min_per_step": 0,
  "penalty_decay": 1.0,
  "penalize_prompt": false,
  "decode_strategy": "threshold",
  "block_size": 32,
  "steps": 32
}
```

All have server-side defaults (see env vars below) and round-trip via
Pydantic's `extra="allow"`. Anything not listed (`logit_bias`, `seed`,
`tools`, `stop` strings, `response_format`, ...) is silently ignored.

### Reasoning output

The model is trained to emit reasoning between `<think>` and `</think>`
markers. The server primes generation with `Assistant: <think>` so the
response **includes** the closing `</think>` and the actual answer:

```
... reasoning text ...
</think>
The actual answer.
```

Clients that want only the answer can split on `"</think>"`.

## Install

```bash
cd /path/to/DiffuRWKV
uv sync --group serve
```

## Run

```bash
CKPT=/path/to/rwkv-N.pth MODEL_NAME=diffurwkv python -m serve
```

You should see `[serve] engine ready ...` followed by uvicorn's listen
line. The first request includes a one-time CUDA-JIT cost (~30s) before
inference proper starts.

## Env vars

| Var                | Default        | Notes |
|--------------------|----------------|-------|
| `CKPT`             | **required**   | path to a `rwkv-*.pth` |
| `MODEL_NAME`       | `diffurwkv`    | id returned by `/v1/models` |
| `HOST`             | `0.0.0.0`      | uvicorn listen |
| `PORT`             | `8000`         | uvicorn listen |
| `MAX_TOKENS`       | `2048`         | server-side cap on completion length |
| `N_LAYER`          | `32`           | must match ckpt |
| `N_EMBD`           | `4096`         | must match ckpt |
| `HEAD_SIZE`        | `64`           | must match ckpt |
| `VOCAB_SIZE`       | `65536`        | must match ckpt |
| `MY_TESTING`       | `x070`         | model family tag |
| `D_DECAY_LORA`     | `128`          | must match ckpt |
| `D_AAA_LORA`       | `128`          | must match ckpt |
| `D_MV_LORA`        | `96`           | must match ckpt |
| `D_GATE_LORA`      | `480`          | must match ckpt |
| `BLOCK_SIZE`       | `32`           | diffusion block size |
| `STEPS`            | `32`           | denoise iterations per block |
| `TOP_K`            | `50`           | default for clients not sending an override |
| `DECODE_STRATEGY`  | `threshold`    | `threshold` | `linear` |
| `CONF_THRESHOLD`   | `0.95`         | LLaDA threshold |
| `MIN_PER_STEP`     | `0`            | fallback floor |
| `PENALTY_DECAY`    | `1.0`          | 1.0 = no decay |
| `PENALIZE_PROMPT`  | `0`            | `1` to seed penalty with prompt tokens |

## Quick smoke

```bash
# list models
curl http://localhost:8000/v1/models | python -m json.tool

# chat completion w/o thinking
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "diffurwkv",
    "messages": [{"role": "user", "content": "How to make a cup of coffee?"}],
    "max_tokens": 1024
  }' | python -m json.tool

# chat completion w/ thinking
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "diffurwkv",
    "messages": [{"role": "user", "content": "How to make a cup of coffee?"}],
    "reasoning_effort": "medium",
    "max_tokens": 1024
  }' | python -m json.tool

# chat completion overriding DiffuRWKV-only knobs (e.g. steps=24, top_k=40,
# block_size=32). They're just extra top-level JSON fields; the engine picks
# them up via Pydantic's extra="allow" and falls back to server defaults for
# anything missing.
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "diffurwkv",
    "messages": [{"role": "user", "content": "How to make a cup of coffee?"}],
    "max_tokens": 1024,
    "steps": 24,
    "top_k": 40,
    "block_size": 32
  }' | python -m json.tool

# error envelope check
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"stream":true}'
# expect 400 with {"error": {"message": "streaming is not supported ...", ...}}
```

### OpenAI Python SDK

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
r = c.chat.completions.create(
    model="diffurwkv",
    messages=[{"role": "user", "content": "hi"}],
    max_tokens=512,
)
print(r.choices[0].message.content)
print(r.usage)
```

### Queueing behavior

```bash
# Fire 3 in parallel — they will run serially on the GPU but all return.
for i in 1 2 3; do
  curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"diffurwkv","messages":[{"role":"user","content":"hi"}],"max_tokens":128}' &
done
wait
```

Server logs will show `[engine] generated ...` lines emitted strictly
sequentially — total wall time ≈ 3× single-request latency.
