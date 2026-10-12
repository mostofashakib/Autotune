"""Score a model's answer against the reference answer."""
from __future__ import annotations

import string
from collections import Counter

PASS_F1 = 0.8


def reference_score(answer: str, reference: str) -> tuple[bool, float]:
    """(passed, score): passes on exact match or token F1 >= 0.8. The score is the F1."""
    if _normalize(answer) and _normalize(answer) == _normalize(reference):
        return True, 1.0
    f1 = token_f1(answer, reference)
    return f1 >= PASS_F1, f1


def token_f1(answer: str, reference: str) -> float:
    answer_tokens, reference_tokens = _normalize(answer).split(), _normalize(reference).split()
    if not answer_tokens or not reference_tokens:
        return 0.0
    common = sum((Counter(answer_tokens) & Counter(reference_tokens)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(answer_tokens), common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, and strip punctuation around the whole answer."""
    return " ".join(text.lower().split()).strip(string.punctuation + " ")
