"""Convert angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k into RWKV-v7 G1x
JSONL format, emitting role-aware segments with per-segment lossable bits
(see preprocess_segments.py).

Schema of the source dataset (per row):
    category: str  (one of 28 domains)
    model:    str  ("claude-opus-4-6" or "claude-opus-4-7")
    messages: list[{role, content}]   # OpenAI chat format, possibly multi-turn

All assistant turns contain a synthetic ``<think>...</think>`` reasoning span
followed by the user-facing answer (per the dataset card — the CoT is
synthetic, not Claude's actual chain-of-thought).

Combines two existing patterns in this repo:
  * tulu-3 converter's role-aware multi-turn walk
  * GLM-reasoning converter's <think> block parsing for assistant content

Unlike the GLM converter, we DO NOT inject a fixed system prompt — this
dataset has its own per-conversation system prompts (~5800 unique ones)
that carry task context (e.g. "You are a senior backend engineer ...").
If a row has no system message we fall back to the same default the
inference server uses ("You are a helpful assistant").

Output JSONL line layout (multi-turn example):

    {"segments": [
        {"text": "System: <system content>\\n\\n",                "lossable": false},
        {"text": "User: <user content>\\n\\n",                     "lossable": false},
        {"text": "Assistant: <think>",                             "lossable": false},
        {"text": "\\n<reasoning>\\n</think>\\n<answer>",            "lossable": true},
        {"text": "\\n\\nUser: <user content>\\n\\n",               "lossable": false},
        {"text": "Assistant: <think>",                             "lossable": false},
        {"text": "\\n<reasoning>\\n</think>\\n<answer>",            "lossable": true},
        ...
    ]}

Source-repo variants (selectable via ``--variant``):
    full      -> full_train.jsonl      (8,706 ex, all 28 categories)
    instruct  -> instruct_train.jsonl  (7,217 ex, instructional only)
    roleplay  -> roleplay_train.jsonl  (1,489 ex, creative/roleplay)
    code      -> code_train.jsonl      (1,840 ex, coding+math)
"""

import argparse
import json
import re

from datasets import load_dataset

_BLANK_LINES_RE = re.compile(r"\n{2,}")
# Non-greedy, DOTALL so reasoning can span newlines. Tolerates whitespace inside the tag.
_THINK_RE = re.compile(r"<think\s*>(.*?)</think\s*>", re.DOTALL | re.IGNORECASE)

_TURN_SEP = "\n\n"
_DEFAULT_SYSTEM = "You are a helpful assistant"

VARIANTS = {
    "full": "full_train.jsonl",
    "instruct": "instruct_train.jsonl",
    "roleplay": "roleplay_train.jsonl",
    "code": "code_train.jsonl",
}


def clean_txt(txt: str) -> str:
    """Per-message content cleaner from RWKV7-G1x-templates.txt."""
    return _BLANK_LINES_RE.sub("\n", txt.replace("\r\n", "\n")).strip()


def parse_assistant(content: str):
    """Split assistant content into (reasoning, answer).

    Returns ("", content) if no well-formed <think>...</think> block is present.
    """
    if content is None:
        return "", ""
    m = _THINK_RE.search(content)
    if m is None:
        return "", content
    return m.group(1), content[m.end() :]


def conversation_to_segments(messages):
    """Apply RWKV-v7 G1x template + role-aware lossable mask.

    schema: [{"role": "system"|"user"|"assistant", "content": str}, ...]

    A System prefix is always emitted as the first segment — if the source
    row has no system message we fall back to ``_DEFAULT_SYSTEM`` so the
    sequence layout matches the inference server's prompt assembly.
    """
    segments: list[dict] = []
    has_system = bool(messages) and messages[0].get("role") == "system"

    # Emit System segment first (either dataset-provided or fallback).
    if has_system:
        sys_content = clean_txt(messages[0]["content"])
        rest = messages[1:]
    else:
        sys_content = _DEFAULT_SYSTEM
        rest = list(messages)
    segments.append({"text": f"System: {sys_content}\n\n", "lossable": False})

    # Walk remaining messages.
    for i, m in enumerate(rest):
        # Inter-turn separator BEFORE every non-first non-system message.
        # (System already wrote its own trailing \n\n.)
        if i > 0:
            segments.append({"text": _TURN_SEP, "lossable": False})
        role = m["role"]
        content = m["content"] or ""
        if role == "user":
            segments.append({"text": f"User: {clean_txt(content)}", "lossable": False})
        elif role == "assistant":
            reasoning, answer = parse_assistant(content)
            reasoning = clean_txt(reasoning)
            answer = clean_txt(answer)
            if reasoning:
                # Structural prefix not lossable; everything inside the think
                # block + after </think> (the actual answer) is lossable.
                segments.append({"text": "Assistant: <think>", "lossable": False})
                segments.append({"text": f"\n{reasoning}\n</think>\n{answer}", "lossable": True})
            else:
                # No parseable <think> -> empty fake-think prefix, only the
                # answer carries loss. Matches the tulu3 / GLM converter
                # fallback behavior.
                segments.append({"text": "Assistant: <think></think>\n", "lossable": False})
                segments.append({"text": answer, "lossable": True})
        elif role == "system":
            # Unexpected: a later system message (should be the first if any).
            # Treat as a non-lossable note prepended to the turn.
            segments.append({"text": f"System: {clean_txt(content)}", "lossable": False})
        else:
            raise ValueError(f"unknown role: {role!r}")
    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="output JSONL path")
    ap.add_argument("--dataset", default="angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k")
    ap.add_argument(
        "--variant",
        default="full",
        choices=sorted(VARIANTS.keys()),
        help="which source file to convert. 'full' = all categories (default), "
        "'instruct' = instructional only, 'roleplay' = creative, "
        "'code' = coding+math.",
    )
    ap.add_argument(
        "--data-file", default=None, help="raw override for the source file (bypasses --variant)"
    )
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None, help="cap number of rows (smoke testing)")
    ap.add_argument(
        "--messages-field",
        default="messages",
        help="name of the messages column (override if dataset uses a different field)",
    )
    args = ap.parse_args()

    data_file = args.data_file or VARIANTS[args.variant]
    print(
        f"loading dataset={args.dataset!r} data_file={data_file!r} split={args.split!r} ...",
        flush=True,
    )
    ds = load_dataset(args.dataset, data_files=data_file, split=args.split)

    n_written = 0
    n_no_think_turn = 0  # rows where at least one assistant turn lacked <think>
    n_total_assist = 0  # total assistant turns
    n_missing_think = 0  # total assistant turns w/o <think>
    with open(args.output, "w", encoding="utf-8") as f:
        for row in ds:
            if args.limit is not None and n_written >= args.limit:
                break
            msgs = row.get(args.messages_field)
            if not msgs:
                continue
            # Pre-scan to track think coverage for reporting.
            row_missing = False
            for m in msgs:
                if m.get("role") == "assistant":
                    n_total_assist += 1
                    if not _THINK_RE.search(m.get("content") or ""):
                        n_missing_think += 1
                        row_missing = True
            if row_missing:
                n_no_think_turn += 1
            segments = conversation_to_segments(msgs)
            f.write(json.dumps({"segments": segments}, ensure_ascii=False))
            f.write("\n")
            n_written += 1

    msg = f"wrote {n_written} conversations to {args.output} " f"(variant={args.variant})"
    if n_total_assist:
        pct = 100.0 * n_missing_think / n_total_assist
        msg += (
            f"  [assistant turns: {n_total_assist}, "
            f"missing <think>: {n_missing_think} ({pct:.1f}%)]"
        )
    print(msg)


if __name__ == "__main__":
    main()
