"""Helpers for race_chat.yaml. Reuses the AST-eval pattern from the stock
lm-eval race task (preprocess_race.py) but emits a chat-API-friendly prompt:
the article + the last problem's question/options, ending with the
"answer letter" instruction the grader expects.
"""

import ast

_LETTERS = "ABCD"
_LETTER_TO_NUM = {ch: i for i, ch in enumerate(_LETTERS)}


def _last_problem(doc):
    return ast.literal_eval(doc["problems"])[-1]


def doc_to_text(doc):
    problem = _last_problem(doc)
    options = problem["options"]
    return (
        "Given the following passage, question, and four candidate answers "
        "(A, B, C and D), choose the best answer.\n"
        f"Passage: {doc['article'].strip()}\n\n"
        f"Question: {problem['question'].strip()}\n"
        f"A. {options[0]}\n"
        f"B. {options[1]}\n"
        f"C. {options[2]}\n"
        f"D. {options[3]}\n"
        'Your response should end with "The best answer is [the_answer_letter]" '
        "where the [the_answer_letter] is one of A, B, C or D."
    )


def doc_to_target(doc):
    return _LETTERS[_LETTER_TO_NUM[_last_problem(doc)["answer"]]]
