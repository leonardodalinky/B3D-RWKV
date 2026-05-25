"""Helpers for mbpp_passk task. Upstream lm-eval's mbpp/utils.py only
exposes ``pass_at_1`` (hardcoded k=[1]); for pass@k we need a generalized
``pass_at_k(references, predictions, k=[K])``. Everything else
(build_predictions / extract_code_blocks / list_fewshot_samples) is
re-exported from upstream so prompt layout stays byte-identical.
"""

from typing import Union

# Upstream binds `pass_at_k` to the evaluate-module object returned by
# `hf_evaluate.load("code_eval")` (yes, confusing naming). Import that object
# under a clearer name to use its `.compute(...)` method.
# Re-export so `!function utils.build_predictions` / `utils.list_fewshot_samples`
# resolve to the exact upstream functions (no drift in prompt format / fewshot).
from lm_eval.tasks.mbpp.utils import (  # noqa: F401
    build_predictions,
    extract_code_blocks,
    list_fewshot_samples,
)
from lm_eval.tasks.mbpp.utils import pass_at_k as _code_eval_module


def pass_at_k(
    references: Union[str, list[str]],
    predictions: Union[str, list[list[str]]],
    k: list[int] | None = None,
) -> dict:
    """Mirror humaneval's pass_at_k contract: return the FULL dict
    {"pass@k1": ..., "pass@k2": ...} from evaluate's code_eval module.
    lm-eval's metric aggregator inspects the dict and splits each key into
    its own metric column (e.g. yaml `k: [5]` -> column "pass@5"). Returning
    a single float here would crash lm-eval downstream (it indexes
    result[f"pass@{k}"]).
    """
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]
    if k is None:
        k = [1]
    if isinstance(k, int):
        k = [k]
    return _code_eval_module.compute(references=references, predictions=predictions, k=k)[0]
