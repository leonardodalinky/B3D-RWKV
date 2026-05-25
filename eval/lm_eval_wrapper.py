"""lm-eval CLI wrapper that exposes the server's reasoning_content to filters.

Why this exists
---------------
DiffuRWKV's infer/serve splits each chat response on the first ``</think>``:
``reasoning_content`` holds the pre-``</think>`` span, ``content`` holds the
post-``</think>`` answer. lm-eval's stock LocalChatCompletion only reads
``choices[0].message.content`` (openai_completions.py: parse_generations).
Two failure modes that causes:

  1. The model runs out of budget before emitting ``</think>``. Server policy
     in that case is to dump everything into ``reasoning_content`` and leave
     ``content`` as ``""``. lm-eval then sees an empty string and every
     filter fails -- even though the model wrote the correct answer inside
     reasoning.
  2. The model writes its final answer inside the think span and emits a
     terse / non-canonical ``content`` ("OK." etc). Same outcome as (1) for
     scoring purposes.

Patch
-----
We monkey-patch LocalChatCompletion.parse_generations to return
``reasoning_content + "\\n\\n" + content`` so filters scan both spans. The
order (reasoning first, content last) matches the model's actual emission
order:

  * Filters with ``group_select: -1`` (gsm8k_cot flexible-extract, arc/piqa
    answer-letter cascade) pick the LAST regex match -> they prefer
    content. Correct behavior preserved.
  * Filters with default ``take_first`` (gsm8k_cot strict-match) pick the
    FIRST match -> they will now extract from reasoning if it contains a
    "The answer is N" line before the final ``</think>``. This is the
    intended behavior for case (1) above. A user who disagrees should
    switch to a ``group_select: -1`` filter or use this wrapper without
    the patch.

Usage
-----
Drop-in replacement for the ``lm-eval`` console script:

    .venv/bin/python eval/lm_eval_wrapper.py --model local-chat-completions ...

eval/run_eval.sh calls this wrapper instead of bare ``lm-eval``.
"""

import logging
import sys

from lm_eval.models.openai_completions import LocalChatCompletion

eval_logger = logging.getLogger("lm_eval")


def _parse_generations_with_reasoning(outputs, **kwargs):
    res = []
    if not isinstance(outputs, list):
        outputs = [outputs]
    for out in outputs:
        try:
            tmp = [None] * len(out["choices"])
            for ch in out["choices"]:
                msg = ch.get("message", {}) or {}
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
                if reasoning and content:
                    tmp[ch["index"]] = f"{reasoning}\n\n{content}"
                else:
                    tmp[ch["index"]] = reasoning or content
        except Exception as e:
            eval_logger.warning(f"Could not parse generations: {e}")
            tmp = [""]
        res += tmp
    return res


LocalChatCompletion.parse_generations = staticmethod(_parse_generations_with_reasoning)


if __name__ == "__main__":
    from lm_eval.__main__ import cli_evaluate

    sys.exit(cli_evaluate())
