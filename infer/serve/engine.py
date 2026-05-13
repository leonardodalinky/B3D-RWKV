"""InferenceEngine: thin wrapper over diffusion_sample.run_one for the
FastAPI server. Holds the loaded model + tokenizer + a single-slot
semaphore that serializes inference (no batching). Translates an
OpenAI-style ChatCompletionRequest into the G1x prompt + DiffuRWKV
sampling knobs, awaits inference on a worker thread (so the asyncio loop
stays responsive for queued requests), and packages the result back into
an OpenAI-style ChatCompletionResponse.
"""
from __future__ import annotations

import asyncio
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

DEFAULT_SYSTEM_PROMPT = (
    "System: You are a helpful assistant. Respond in a concise and informative manner.\n\n"
)

# Reach the same tokenizer + diffusion_sample modules sweep_inference.py
# uses. diffusion_sample.build_model also adds the repo root to sys.path
# for `tokenizer`, but we add it now so `from tokenizer import ...` works
# during engine construction too. Layout: this file is at
# DiffuRWKV/infer/serve/engine.py, so parent.parent.parent is DiffuRWKV/.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_INFER_DIR = _REPO_ROOT / "infer"
for p in (str(_REPO_ROOT), str(_INFER_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from .protocol import (  # noqa: E402  (after sys.path mutation)
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseMessage,
    ChatMessage,
    UsageInfo,
)

# Collapse 2+ newlines per RWKV7-G1x-templates.txt's `clean_txt`.
_BLANK_LINES_RE = re.compile(r"\n{2,}")


def _clean(text: str) -> str:
    return _BLANK_LINES_RE.sub("\n", text.replace("\r\n", "\n")).strip()


class InferenceEngine:
    def __init__(
        self,
        ckpt: str,
        model_args: SimpleNamespace,
        *,
        defaults: dict[str, Any],
        max_tokens_cap: int,
    ) -> None:
        # Import here so the heavy CUDA-touching modules don't load at
        # `import infer.serve.engine` time (FastAPI startup imports the module
        # before lifespan fires; we want model loading to happen inside
        # lifespan).
        import diffusion_sample as ds  # type: ignore
        from tokenizer import RWKVTokenizer  # type: ignore

        self._ds = ds
        self.model = ds.build_model(ckpt, model_args)
        self.tok = RWKVTokenizer()
        self.vocab_size = int(model_args.vocab_size)
        self.mask_id = self.vocab_size - 1
        self.defaults = defaults
        self.max_tokens_cap = max_tokens_cap
        # Serialize inference: only one request goes through the GPU at a
        # time; the rest queue at this lock. `asyncio.to_thread` keeps the
        # event loop free to accept and queue more requests while one is
        # running.
        self._sem = asyncio.Semaphore(1)

    # ------------------------------------------------------------------
    # G1x prompt assembly
    # ------------------------------------------------------------------
    @staticmethod
    def messages_to_prompt(
        messages: list[ChatMessage],
        reasoning_effort: str | None = None,
    ) -> str | None:
        """Render a list of OpenAI chat messages into a single G1x prompt
        string. Returns None if the message list is unusable (empty, or
        doesn't end on a user turn).

        ``reasoning_effort`` controls think-mode priming:
          * ``None`` or ``"minimal"``  -> finishes with ``"Assistant:"``;
            the model answers directly without a ``<think>`` reasoning span.
          * anything else (``"low"`` | ``"medium"`` | ``"high"`` | ...)
            -> finishes with ``"Assistant: <think>"``, so the model first
            emits reasoning, then ``</think>``, then the answer.

        Layout follows RWKV-v7/RWKV7-G1x-templates.txt and the converter
        in train/data_prep/convert_glm_reasoning_to_jsonl.py:

            [optional] "System: <sys>\\n\\n"
            "User: <u1>\\n\\n"
            "Assistant: <a1>\\n\\n"      # only for past turns
            ...
            "User: <last>\\n\\n"
            "Assistant: <think>"  OR  "Assistant:"
        """
        if not messages:
            return None
        # The conversation must end with a user message — otherwise we
        # don't know what to generate.
        if messages[-1].role != "user":
            return None

        parts: list[str] = []
        # Only ONE leading system message is honored (collapse multiples).
        system_msgs = [m for m in messages if m.role == "system"]
        if system_msgs:
            parts.append(f"System: {_clean(system_msgs[0].content)}\n\n")
        else:
            # Default system prompt when client didn't supply one. The model
            # was trained on a lot of "System: You are a helpful assistant" data
            # so always providing some system text matches the training distribution.
            parts.append(DEFAULT_SYSTEM_PROMPT)

        for m in messages:
            match m.role:
                case "system":
                    continue
                case "user":
                    label = "User"
                case "assistant":
                    label = "Assistant"
                case _:
                    # Skip unsupported roles (e.g. "function" and "tool").
                    continue
            parts.append(f"{label}: {_clean(m.content)}\n\n")

        # Priming — `<think>` switches the model into reasoning mode (it
        # will emit `</think>` once done, then the answer). Skip it when
        # the caller passed no effort hint or asked for "minimal" — the
        # model then answers immediately, which is what most short queries
        # actually want.
        think = reasoning_effort is not None and reasoning_effort != "minimal"
        parts.append("Assistant: <think>" if think else "Assistant: <think></think>")
        return "".join(parts)

    # ------------------------------------------------------------------
    # Public generate (called from app.py)
    # ------------------------------------------------------------------
    async def generate(
        self, req: ChatCompletionRequest, *, prompt: str, model_name: str
    ) -> ChatCompletionResponse:
        gen_len = self._resolve_gen_len(req)
        sampling = self._resolve_sampling(req)

        async with self._sem:
            t0 = time.time()
            text, finish_reason, n_completion = await asyncio.to_thread(
                self._generate_sync, prompt, gen_len, sampling
            )
            elapsed = time.time() - t0
            print(
                f"[engine] generated {n_completion} tok in {elapsed:.2f}s "
                f"(finish={finish_reason})",
                flush=True,
            )

        # When the request enabled think mode (prompt was primed with
        # "Assistant: <think>"), peel off the reasoning span and surface
        # it as a separate field. Matches the DeepSeek / vLLM convention
        # so OpenAI-compatible clients can choose to hide or render
        # reasoning independently of the final answer.
        think_enabled = (
            req.reasoning_effort is not None
            and req.reasoning_effort != "minimal"
        )
        reasoning_content, content = self._split_reasoning(text, think_enabled=think_enabled)

        # Prompt token count via the tokenizer (cheap; same routine
        # diffusion_sample.run_one would do internally — we duplicate it
        # here to surface it in `usage`).
        n_prompt = len(self.tok.encode(prompt))
        return ChatCompletionResponse(
            model=model_name,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatCompletionResponseMessage(
                        content=content,
                        reasoning_content=reasoning_content,
                    ),
                    finish_reason=finish_reason,  # "stop" | "length"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=n_prompt,
                completion_tokens=n_completion,
                total_tokens=n_prompt + n_completion,
            ),
        )

    @staticmethod
    def _split_reasoning(text: str, *, think_enabled: bool) -> tuple[str | None, str]:
        """Partition the model output around the first ``</think>``.

        Returns ``(reasoning_content, content)``:
          * non-think request -> ``(None, text)``  (no split, content is full text)
          * think request, ``</think>`` found -> ``(reasoning, answer)``
          * think request, ``</think>`` missing (model ran out of budget
            mid-reasoning) -> ``(text, "")`` so the partial reasoning isn't lost

        Strips the surrounding ``\\n`` that the G1x template inserts around
        the reasoning span (training: ``"<think>\\n{reasoning}\\n</think>\\n{answer}"``).
        """
        if not think_enabled:
            return None, text
        sep = "</think>"
        if sep in text:
            reasoning, _, answer = text.partition(sep)
            return reasoning.strip("\n"), answer.lstrip("\n")
        return text, ""

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _generate_sync(
        self, prompt: str, gen_len: int, sampling: dict[str, Any]
    ) -> tuple[str, str, int]:
        """Blocking call into diffusion_sample.run_one. Runs on a worker
        thread via asyncio.to_thread so the event loop keeps draining HTTP.
        """
        return self._ds.run_one(
            self.model, self.tok, self.mask_id, self.vocab_size,
            prompt, gen_len, sampling["steps"], sampling["block_size"],
            sampling["temperature"], sampling["top_k"], sampling["top_p"],
            sampling["decode_strategy"], sampling["conf_threshold"], sampling["min_per_step"],
            sampling["presence_penalty"], sampling["count_penalty"], sampling["penalty_decay"],
            sampling["penalize_prompt"],
            verbose=False,
        )

    def _resolve_gen_len(self, req: ChatCompletionRequest) -> int:
        # OpenAI clients send either `max_tokens` (legacy) or
        # `max_completion_tokens` (newer). Honor whichever is set, then
        # clamp to the configured server cap.
        raw = req.max_completion_tokens or req.max_tokens or self.max_tokens_cap
        return max(1, min(int(raw), self.max_tokens_cap))

    def _resolve_sampling(self, req: ChatCompletionRequest) -> dict[str, Any]:
        # DiffuRWKV-specific knobs come in via Pydantic's `model_extra`
        # (because protocol.ChatCompletionRequest has extra="allow").
        # Anything missing falls back to the engine-wide defaults.
        extra = req.model_extra or {}

        def pick(name: str, default):
            return extra.get(name, default)

        return {
            # OpenAI -> DiffuRWKV direct mappings
            "temperature": float(req.temperature),
            "top_p": float(req.top_p),
            "presence_penalty": float(req.presence_penalty),
            "count_penalty": float(req.frequency_penalty),  # OpenAI's nearest equivalent
            # DiffuRWKV-only knobs (via extra), with server-wide defaults
            "top_k": int(pick("top_k", self.defaults["top_k"])),
            "decode_strategy": str(pick("decode_strategy", self.defaults["decode_strategy"])),
            "conf_threshold": float(pick("conf_threshold", self.defaults["conf_threshold"])),
            "min_per_step": int(pick("min_per_step", self.defaults["min_per_step"])),
            "penalty_decay": float(pick("penalty_decay", self.defaults["penalty_decay"])),
            "penalize_prompt": bool(pick("penalize_prompt", self.defaults["penalize_prompt"])),
            # Fixed inference loop knobs (block-diffusion specific)
            "block_size": int(pick("block_size", self.defaults["block_size"])),
            "steps": int(pick("steps", self.defaults["steps"])),
        }
