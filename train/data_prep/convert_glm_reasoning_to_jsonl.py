"""Convert Jackrong/GLM-5.1-Reasoning-1M-Cleaned into RWKV-v7 G1x JSONL format,
emitting role-aware segments with per-segment lossable bits (see preprocess_segments.py).

Each row has:
    input:  user prompt (str)
    output: assistant response, typically wrapped as "<think>{reasoning}</think>{answer}"

Output JSONL line layout (single 2-turn User -> Assistant conversation):

    {"segments": [
        {"text": "User: {input}",                 "lossable": false},
        {"text": "\\n\\nAssistant: <think>",       "lossable": false},   # structural prefix
        {"text": "\\n{reasoning}\\n</think>\\n{answer}", "lossable": true},
    ]}

Concatenation reproduces the original RWKV-v7 G1x think-mode template, but now
only the reasoning + answer span carries loss; user prompt and the structural
"Assistant: <think>" marker do not.

If <think> tags are missing or malformed in ``output`` we fall back to the
fake-think prefix (``<think></think>``), matching the tulu3 converter.
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
    return m.group(1), output[m.end():]


def row_to_segments(user_input: str, assistant_output: str):
    """Build the role-aware segment list for one (input, output) row."""
    user = clean_txt(user_input or "")
    reasoning, answer = parse_output(assistant_output or "")
    reasoning = clean_txt(reasoning)
    answer = clean_txt(answer)

    segments = [
        # User prompt (incl. role prefix and turn separator) -> never lossable.
        {"text": f"User: {user}", "lossable": False},
        {"text": "\n\n", "lossable": False},
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
    ap.add_argument("--subset", default=None,
                    help="HF dataset config / subset name (None -> default). "
                         "GLM-5.1-Reasoning-1M-Cleaned has subsets like 'math', 'code', etc.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None, help="cap number of rows (smoke testing)")
    ap.add_argument("--input-field", default="input")
    ap.add_argument("--output-field", default="output")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, args.subset, split=args.split, streaming=False)

    n_written = 0
    n_no_think = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for row in ds:
            if args.limit is not None and n_written >= args.limit:
                break
            segments = row_to_segments(row[args.input_field], row[args.output_field])
            # The "fake-think" path is the one whose 3rd segment text starts with the
            # empty-think prefix. Count for reporting.
            if any(s["text"].startswith("Assistant: <think></think>") for s in segments):
                n_no_think += 1
            f.write(json.dumps({"segments": segments}, ensure_ascii=False))
            f.write("\n")
            n_written += 1

    msg = f"wrote {n_written} conversations to {args.output}"
    if n_no_think:
        msg += f" ({n_no_think} rows had no <think> block, used fake-think prefix)"
    print(msg)


if __name__ == "__main__":
    main()
