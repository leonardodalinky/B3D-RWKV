"""Convert Jackrong/GLM-5.1-Reasoning-1M-Cleaned into RWKV-v7 G1x JSONL format,
emitting role-aware segments with per-segment lossable bits (see preprocess_segments.py).

Each row has:
    input:  user prompt (str)
    output: assistant response, typically wrapped as "<think>{reasoning}</think>{answer}"

Output JSONL line layout (single 2-turn User -> Assistant conversation, with a
fixed system prompt prepended):

    {"segments": [
        {"text": "System: You are a helpful assistant\n\n", "lossable": false},
        {"text": f"User: {user}\n\n", "lossable": false},
        {"text": f"Assistant: <tool_call>", "lossable": false},   # structural prefix
        {"text": f"\n{reasoning}\n<tool_call>\n{answer}", "lossable": true},
    ]}

Concatenation reproduces the original RWKV-v7 G1x think-mode template, but now
only the reasoning + answer span carries loss; the system prompt, user prompt
and the structural "Assistant: <think>" marker do not.

If <think> tags are missing or malformed in ``output`` we fall back to the
fake-think prefix (``<think></think>``), matching the tulu3 converter.

The dataset has multiple HF subsets (configs). By default we concatenate
``main``, ``PHD-Science``, ``Multilingual-STEM`` and ``Math`` into the
output JSONL; override with ``--subset`` (accepts a list).
"""

import argparse
import json
import re

from datasets import load_dataset

_BLANK_LINES_RE = re.compile(r"\n{2,}")
# Non-greedy, DOTALL so reasoning can span newlines. Tolerates whitespace inside the tag.
_THINK_RE = re.compile(r"<think\s*>(.*?)</think\s*>", re.DOTALL | re.IGNORECASE)


def clean_txt(txt: str) -> str:
    """Per-message content cleaner from RWKV7-G1x-templates.txt."""
    return _BLANK_LINES_RE.sub("\n", txt.replace("\r\n", "\n")).strip()


def parse_output(output: str):
    """Split ``output`` into (reasoning, answer).

    Returns ("", output) if no well-formed <think>...</think> block is present.
    """
    if output is None:
        return "", ""
    m = _THINK_RE.search(output)
    if m is None:
        return "", output
    return m.group(1), output[m.end() :]


def row_to_segments(user_input: str, assistant_output: str):
    """Build the role-aware segment list for one (input, output) row."""
    user = clean_txt(user_input or "")
    reasoning, answer = parse_output(assistant_output or "")
    reasoning = clean_txt(reasoning)
    answer = clean_txt(answer)

    segments = [
        # Fixed system prompt prepended to every conversation. Structural,
        # not lossable.
        {"text": "System: You are a helpful assistant\n\n", "lossable": False},
        # User prompt (incl. role prefix and turn separator) -> never lossable.
        {"text": f"User: {user}\n\n", "lossable": False},
    ]
    if reasoning:
        # "Assistant: <think>" structural marker -> not lossable.
        # Everything after it (the reasoning, </think>, the actual answer) -> lossable.
        segments.append({"text": "Assistant: <think>", "lossable": False})
        segments.append({"text": f"\n{reasoning}\n</think>\n{answer}", "lossable": True})
    else:
        # No (parseable) reasoning -> empty fake-think prefix, only the answer is lossable.
        segments.append({"text": "Assistant: <think></think>\n", "lossable": False})
        segments.append({"text": answer, "lossable": True})
    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="output JSONL path")
    ap.add_argument("--dataset", default="Jackrong/GLM-5.1-Reasoning-1M-Cleaned")
    ap.add_argument(
        "--subset",
        nargs="+",
        default=["main", "PHD-Science", "Multilingual-STEM", "Math"],
        help="HF dataset config / subset name(s); accepts multiple, the output JSONL "
        "concatenates them in order. Default covers the four standard subsets.",
    )
    ap.add_argument("--split", default="train")
    ap.add_argument(
        "--limit", type=int, default=None, help="total cap across all subsets (smoke testing)"
    )
    ap.add_argument("--input-field", default="input")
    ap.add_argument("--output-field", default="output")
    args = ap.parse_args()

    n_written = 0
    n_no_think = 0
    per_subset_counts: list[tuple[str, int]] = []
    with open(args.output, "w", encoding="utf-8") as f:
        for subset in args.subset:
            if args.limit is not None and n_written >= args.limit:
                break
            print(f"loading subset={subset!r} ...", flush=True)
            ds = load_dataset(args.dataset, subset, split=args.split, streaming=False)
            n_before = n_written
            for row in ds:
                if args.limit is not None and n_written >= args.limit:
                    break
                segments = row_to_segments(row[args.input_field], row[args.output_field])
                # The "fake-think" path is the one whose segment text starts with the
                # empty-think prefix. Count for reporting.
                if any(s["text"].startswith("Assistant: <think></think>") for s in segments):
                    n_no_think += 1
                f.write(json.dumps({"segments": segments}, ensure_ascii=False))
                f.write("\n")
                n_written += 1
            per_subset_counts.append((subset, n_written - n_before))

    msg = f"wrote {n_written} conversations to {args.output}"
    if per_subset_counts:
        breakdown = ", ".join(f"{s}={n}" for s, n in per_subset_counts)
        msg += f"  [per subset: {breakdown}]"
    if n_no_think:
        msg += f" ({n_no_think} rows had no <think> block, used fake-think prefix)"
    print(msg)


if __name__ == "__main__":
    main()
