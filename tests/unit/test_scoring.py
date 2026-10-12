"""Reference scoring: exact match or token F1 at 0.8."""
from __future__ import annotations

import pytest

from autotune.infrastructure.evaluators.scoring import reference_score, token_f1


def test_an_exact_match_passes_after_normalizing_case_space_and_punctuation():
    assert reference_score("  The answer is PARIS. ", "the answer is paris") == (True, 1.0)


def test_close_paraphrase_passes_on_token_f1():
    passed, score = reference_score("ls -la /var/log here", "ls -la /var/log")
    assert passed and score == pytest.approx(token_f1("ls -la /var/log here", "ls -la /var/log"))


def test_a_wrong_answer_fails_with_its_f1_as_the_score():
    passed, score = reference_score("rm -rf /tmp", "ls -la /var/log")
    assert not passed
    assert score == pytest.approx(0.0)


def test_partial_overlap_below_threshold_fails():
    passed, score = reference_score("ls /home", "ls -la /var/log")
    assert not passed
    assert 0.0 < score < 0.8


@pytest.mark.parametrize("answer", ["", "   ", "..."])
def test_an_empty_answer_scores_zero(answer):
    assert reference_score(answer, "ls -la") == (False, 0.0)


def test_repeated_tokens_are_counted_once_per_match():
    assert token_f1("yes yes yes", "yes") == pytest.approx(0.5)
