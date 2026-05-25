"""Custom scorer for gsm8k_cot: three numerical match modes in one pass.

We swap the upstream dual-filter setup (strict-match / flexible-extract +
exact_match metric per column) for ONE passthrough filter + a custom
``process_results`` that derives all three scores from the raw response.
That gets us a clean per-row report (one filter column, three metric
columns) instead of the 2x2-with-noise table the filter+metric approach
would produce.

Metrics:
  exact_match     -- literal "The answer is N" with N == gold (after
                     normalization). Same regex as upstream strict-match.
                     Kept for continuity with reported leaderboard numbers.
  flexible_match  -- the LAST numeric token in the response equals gold.
                     Same logic as upstream flexible-extract (group_select=-1).
  loose_contains  -- gold appears ANYWHERE in the response as a standalone
                     numeric token. Most permissive; useful when the model
                     gets the right answer but mis-formats the final line
                     or has the answer stuck inside the <think> span.

Hooked up via:
  process_results: !function gsm8k_cot_utils.process_results
in eval/tasks/gsm8k_cot.yaml. lm-eval resolves the module path relative to
the yaml's directory (utils.py:584 import_function).
"""

import re

# "The answer is <num>" -- mirrors upstream strict-match.
_STRICT_RE = re.compile(r"The answer is (-?[\d.,]+)")
# Numeric tokens: optional minus, digits, optional decimal part. No comma
# handling here (we strip commas from the haystack before findall).
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _normalize(s: str) -> str:
    """Strip cosmetic decorations gsm8k normally regexes-to-ignore.

    Drop commas/dollars (US-locale thousands separator + currency), trailing
    period (sentence terminator), surrounding whitespace.
    """
    return s.replace(",", "").replace("$", "").rstrip(".").strip()


def _gold(doc: dict) -> str:
    # openai/gsm8k stores the gold inside `answer` after the "####" marker.
    # Fewshot demo dicts only have `target` -- not relevant at scoring time
    # but harmless to handle.
    if "answer" in doc:
        return _normalize(doc["answer"].split("####")[-1])
    return _normalize(doc.get("target", ""))


def process_results(doc, results):
    pred = results[0] if results else ""
    gold = _gold(doc)

    # ---- exact_match: literal "The answer is N" ----
    m = _STRICT_RE.search(pred)
    strict_pred = _normalize(m.group(1)) if m else ""
    exact_match = 1.0 if strict_pred == gold else 0.0

    # Normalize the haystack once for both flexible / loose checks: strip
    # commas (so "1,000" -> "1000") and dollar signs.
    pred_clean = pred.replace(",", "").replace("$", "")
    nums = _NUM_RE.findall(pred_clean)

    # ---- flexible_match: LAST numeric token equals gold ----
    flex_pred = _normalize(nums[-1]) if nums else ""
    flexible_match = 1.0 if flex_pred == gold else 0.0

    # ---- loose_contains: ANY extracted numeric token equals gold ----
    # Using findall + equality (vs. substring match) automatically gives us
    # the right token boundaries: "7" in "the value is 17" is NOT a match
    # because findall extracts "17", not "7"; "7" in "answer is 7." IS a
    # match (the trailing "." isn't part of the captured number).
    loose_contains = 1.0 if gold in (_normalize(n) for n in nums) else 0.0

    return {
        "exact_match": exact_match,
        "flexible_match": flexible_match,
        "loose_contains": loose_contains,
    }
