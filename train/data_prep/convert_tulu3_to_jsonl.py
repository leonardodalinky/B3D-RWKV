"""Convert allenai/tulu-3-sft-mixture conversations into RWKV-v7 G1x JSONL format,
emitting **role-aware segments** so downstream tools can build a per-token lossable
mask (see preprocess_segments.py).

Each output JSONL line is one conversation:

    {"segments": [
        {"text": "System: You are a helpful assistant",   "lossable": false},
        {"text": "\\n\\n",                                "lossable": false},
        {"text": "User: hi",                              "lossable": false},
        {"text": "\\n\\n",                                "lossable": false},
        {"text": "Assistant: <think></think>\\n",          "lossable": false},
        {"text": "the assistant's actual reply",          "lossable": true},
        {"text": "\\n\\n",                                "lossable": false},
        ...
    ]}

Concatenating ``segment["text"]`` in order yields exactly the same string as the
old ``"text"``-only format (so the surface tokenization is unchanged), plus a
parallel lossable bit per segment.

Loss-mask conventions:
- System / User / role prefix / inter-turn separators -> lossable = false.
- "Assistant: <think></think>\\n" prefix is treated like a structural marker (not
  lossable) per the user's spec; the assistant's actual reply is the only lossable
  span per turn.
- tulu-3 conversations have no explicit think block -> we always use the empty
  fake-think prefix (matches RWKV7-G1x "fake thinking" template).
- If a row has no system message, we prepend the default
  ``"You are a helpful assistant"`` so every example has a consistent
  ``System: ...`` prefix (matches the inference server's prompt assembly
  and the GLM / Claude reasoning converters in this dir).
"""

import argparse
import json
import re

from datasets import load_dataset

_BLANK_LINES_RE = re.compile(r"\n{2,}")
_TURN_SEP = "\n\n"
_ASSISTANT_PREFIX = "Assistant: <think></think>\n"
_DEFAULT_SYSTEM = "You are a helpful assistant"


def clean_txt(txt: str) -> str:
    """Per-message content cleaner from RWKV7-G1x-templates.txt."""
    return _BLANK_LINES_RE.sub("\n", txt.replace("\r\n", "\n")).strip()


def conversation_to_segments(messages):
    """Apply RWKV-v7 G1x template + role-aware lossable mask.

    tulu-3 schema: [{"role": "system" | "user" | "assistant", "content": "..."}, ...]
    Returns: list of {"text": str, "lossable": bool} segments.

    If the row doesn't start with a system message, we prepend a synthetic
    one with ``_DEFAULT_SYSTEM`` content so the System: prefix is always
    present (mirrors the inference server + the other converters).
    """
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": _DEFAULT_SYSTEM}, *messages]

    segments = []
    for i, m in enumerate(messages):
        if i > 0:
            segments.append({"text": _TURN_SEP, "lossable": False})
        role = m["role"]
        content = clean_txt(m["content"])
        if role == "system":
            segments.append({"text": f"System: {content}", "lossable": False})
        elif role == "user":
            segments.append({"text": f"User: {content}", "lossable": False})
        elif role == "assistant":
            # Structural prefix is not part of the assistant's "answer" -> not lossable.
            segments.append({"text": _ASSISTANT_PREFIX, "lossable": False})
            segments.append({"text": content, "lossable": True})
        else:
            raise ValueError(f"unknown role: {role!r}")
    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="output JSONL path")
    ap.add_argument("--dataset", default="allenai/tulu-3-sft-mixture")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None, help="cap number of rows (smoke testing)")
    ap.add_argument(
        "--messages-field",
        default="messages",
        help="name of the messages column (override if dataset uses a different field)",
    )
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split, streaming=False)

    n_written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if args.limit is not None and n_written >= args.limit:
                break
            segments = conversation_to_segments(row[args.messages_field])
            f.write(json.dumps({"segments": segments}, ensure_ascii=False))
            f.write("\n")
            n_written += 1

    print(f"wrote {n_written} conversations to {args.output}")


if __name__ == "__main__":
    main()
