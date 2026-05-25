"""Custom scorer for mmlu_generative_new (all 57 subjects share this).

Mirrors the gsm8k_cot three-metric pattern, adapted for A/B/C/D answers:

  exact_match     -- the response ENDS with the gold letter (after stripping
                     trailing punctuation/whitespace). The strict case where
                     the model finished with "...is C." or just "C".
  flexible_match  -- the LAST standalone uppercase A/B/C/D in the response
                     equals gold. Catches chat-style answers like "B. ATP"
                     where the letter is the last salient token but the
                     response continues with extra prose.
  loose_contains  -- the gold letter appears ANYWHERE in the response as a
                     standalone A/B/C/D. Most permissive; useful when the
                     model picked the right option somewhere in <think>...
                     </think> but the post-</think> content went off-format.

Hooked up via:
  process_results: !function mmlu_utils.process_results
in _default_template_yaml. Each subject yaml `include:`s the template, so
the 57 subjects all use the same scorer.

The response text fed into `results[0]` is the merged
``reasoning_content + "\\n\\n" + content`` produced by
eval/lm_eval_wrapper.py -- meaning all three modes also see the think span.
"""

import re

# Standalone uppercase A/B/C/D, bounded by non-word chars (or string edges).
# `\b` works here because A-D are single chars and the boundary excludes
# adjacent letters/digits -- "A" in "ATP" won't match, "(A)" will.
_LETTER_RE = re.compile(r"\b([A-D])\b")

# Trailing junk we strip when locating the response's final character:
# whitespace, periods, closing brackets/parens, colons.
_TAIL_JUNK_RE = re.compile(r"[\s.):\]]+$")


def _gold(doc: dict) -> str:
    """Resolve the gold letter for an MMLU doc.

    cais/mmlu test rows store the gold as an integer index 0..3 in
    ``doc["answer"]``. (The template's doc_to_target =
    "{{['A', 'B', 'C', 'D'][answer]}}" does this conversion for the
    standard scoring path; we replicate it here for process_results.)
    """
    ans = doc.get("answer")
    if isinstance(ans, int) and 0 <= ans <= 3:
        return "ABCD"[ans]
    # Already a letter (e.g., from a fewshot demo dict)
    if isinstance(ans, str):
        return ans.strip().upper()[:1]
    return ""


def process_results(doc, results):
    pred = (results[0] if results else "") or ""
    gold = _gold(doc)

    # ---- exact_match: response ENDS with gold (trailing punc/whitespace stripped) ----
    end = _TAIL_JUNK_RE.sub("", pred)
    strict_pred = end[-1] if end and end[-1] in "ABCD" else ""
    exact_match = 1.0 if strict_pred == gold else 0.0

    # Find every standalone uppercase A/B/C/D in the (merged) response.
    letters = _LETTER_RE.findall(pred)

    # ---- flexible_match: LAST extracted letter equals gold ----
    flex_pred = letters[-1] if letters else ""
    flexible_match = 1.0 if flex_pred == gold else 0.0

    # ---- loose_contains: gold appears ANYWHERE in the extracted letters ----
    loose_contains = 1.0 if gold and gold in letters else 0.0

    return {
        "exact_match": exact_match,
        "flexible_match": flexible_match,
        "loose_contains": loose_contains,
    }
