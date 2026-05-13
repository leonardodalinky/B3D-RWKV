"""FastAPI app exposing an OpenAI-compatible /v1/chat/completions endpoint
backed by a single DiffuRWKV model instance. See serve/README.md for the
supported subset of OpenAI semantics and what's intentionally rejected.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .engine import InferenceEngine
from .protocol import (
    APIError,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Model,
    ModelList,
)


# ----------------------------------------------------------------------------
# Lifespan: build the engine on startup. One model lives for the whole process.
# ----------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _build_engine() -> InferenceEngine:
    ckpt = os.environ["CKPT"]  # required
    model_args = SimpleNamespace(
        n_layer=_env_int("N_LAYER", 32),
        n_embd=_env_int("N_EMBD", 4096),
        dim_att=_env_int("N_EMBD", 4096),
        dim_ffn=int((_env_int("N_EMBD", 4096) * 3.5) // 32 * 32),
        head_size=_env_int("HEAD_SIZE", 64),
        vocab_size=_env_int("VOCAB_SIZE", 65536),
        ctx_len=_env_int("CTX_LEN", 4096),
        my_testing=os.environ.get("MY_TESTING", "x070"),
        grad_cp=0,
        weight_decay=0.0,
        lr_init=0.0, lr_final=0.0, betas=(0.9, 0.99), adam_eps=1e-18,
        layerwise_lr=0, my_pile_stage=0, train_stage=0,
        diffusion_mode=0,
        d_decay_lora=_env_int("D_DECAY_LORA", 128),
        d_aaa_lora=_env_int("D_AAA_LORA", 128),
        d_mv_lora=_env_int("D_MV_LORA", 96),
        d_gate_lora=_env_int("D_GATE_LORA", 480),
    )
    defaults = {
        "block_size": _env_int("BLOCK_SIZE", 32),
        "steps": _env_int("STEPS", 32),
        "top_k": _env_int("TOP_K", 50),
        "decode_strategy": os.environ.get("DECODE_STRATEGY", "threshold"),
        "conf_threshold": _env_float("CONF_THRESHOLD", 0.9),
        "min_per_step": _env_int("MIN_PER_STEP", 0),
        "penalty_decay": _env_float("PENALTY_DECAY", 1.0),
        "penalize_prompt": bool(int(os.environ.get("PENALIZE_PROMPT", "0"))),
    }
    return InferenceEngine(
        ckpt,
        model_args,
        defaults=defaults,
        max_tokens_cap=_env_int("MAX_TOKENS", 2048),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[serve] building engine ...", flush=True)
    app.state.engine = _build_engine()
    app.state.model_name = os.environ.get("MODEL_NAME", "diffurwkv")
    print(f"[serve] engine ready (model_name={app.state.model_name!r})", flush=True)
    yield
    # No teardown needed — process exit releases CUDA.


app = FastAPI(title="DiffuRWKV OpenAI-compatible server", lifespan=lifespan)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _err(status: int, message: str, *, param: str | None = None) -> JSONResponse:
    """Return an OpenAI-shaped error envelope:
    ``{"error": {"message": ..., "type": ...}}``."""
    err = APIError(message=message, param=param)
    return JSONResponse(status_code=status, content={"error": err.model_dump()})


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.get("/v1/models")
async def list_models(request: Request) -> ModelList:
    return ModelList(data=[Model(id=request.app.state.model_name)])


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    # ---- Reject the OpenAI features we don't implement ----
    if req.stream:
        return _err(400, "streaming is not supported by this server", param="stream")
    if req.n != 1:
        return _err(400, "n > 1 is not supported by this server", param="n")
    if not req.messages:
        return _err(400, "messages must be non-empty", param="messages")

    engine: InferenceEngine = request.app.state.engine
    prompt = InferenceEngine.messages_to_prompt(req.messages, req.reasoning_effort)
    if prompt is None:
        return _err(400, "messages must end with a user message", param="messages")

    try:
        return await engine.generate(
            req, prompt=prompt, model_name=request.app.state.model_name
        )
    except Exception as e:
        # Unexpected — log and return a generic 500 so we don't leak stack.
        print(f"[serve] inference error: {type(e).__name__}: {e}", flush=True)
        return _err(500, f"internal error: {type(e).__name__}")


@app.get("/health")
async def health(request: Request):
    ready = hasattr(request.app.state, "engine")
    return {"status": "ok" if ready else "loading"}
