"""Pydantic models for the OpenAI-compatible chat completions endpoint.

Field shapes mirror OpenAI's public spec (and vLLM's reference implementation
of an OpenAI-compatible server) — only the subset DiffuRWKV actually supports
is declared explicitly. `extra="allow"` on the request lets clients pass
DiffuRWKV-specific knobs (`top_k`, `conf_threshold`, `decode_strategy`,
`min_per_step`, `penalty_decay`, `penalize_prompt`) without us hardcoding
them in the schema; the engine reads them from `model_extra` with defaults.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def _new_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _now() -> int:
    return int(time.time())


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    # `extra="allow"` lets DiffuRWKV-only knobs ride through Pydantic.
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]

    # OpenAI sampling
    temperature: float = 1.0
    top_p: float = 0.7
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0  # mapped to DiffuRWKV count_penalty

    # OpenAI length / batch / streaming
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    n: int = 1
    stream: bool = False

    # OpenAI bookkeeping we don't use but accept silently
    stop: str | list[str] | None = None
    user: str | None = None
    seed: int | None = None

    # OpenAI o-series reasoning effort hint. We use it as the on/off switch
    # for DiffuRWKV's think-mode priming:
    #   - None or "minimal" -> no <think> primer (model answers directly)
    #   - anything else ("low" | "medium" | "high" | ...) -> prime with <think>
    reasoning_effort: str | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str
    # DeepSeek / vLLM-style extension: when the model generated a
    # <think>...</think> span, the text before </think> is surfaced here
    # so clients can show / hide reasoning separately. Stays None for
    # no-think requests.
    reasoning_content: str | None = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatCompletionResponseMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=_new_id)
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=_now)
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class Model(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=_now)
    owned_by: str = "local"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[Model]


class APIError(BaseModel):
    """OpenAI-shaped error envelope: ``{"error": {"message": ..., "type": ...}}``."""

    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None
