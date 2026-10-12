"""Dataset mode: k answers per held-out prompt, scored against the reference."""
from __future__ import annotations

import pytest

from autotune.core.domain.models import ModelEndpoint, SFTRecord
from autotune.infrastructure.evaluators.dataset_benchmark import DatasetBenchmark
from autotune.infrastructure.evaluators.openai_client import ModelCallError

MODEL = ModelEndpoint(base_url="http://m/v1", model="m")
RECORDS = [SFTRecord(prompt="List files", completion="ls -la"), SFTRecord(prompt="Where am I", completion="pwd")]


class ScriptedClient:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def complete(self, endpoint, messages, *, temperature=0.7, max_tokens=1024):
        self.prompts.append((endpoint.model, messages[-1]["content"], temperature))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class FixedJudge:
    def __init__(self, score):
        self.value = score
        self.seen = []

    def score(self, prompt, reference, answer):
        self.seen.append((prompt, reference, answer))
        return self.value


def test_each_prompt_is_answered_k_times_and_scored_against_its_reference():
    client = ScriptedClient(["ls -la", "rm -rf /", "pwd", "pwd"])

    samples = DatasetBenchmark(RECORDS, client, k=2, temperature=0.4).run(MODEL)

    assert [(s.task_id, s.passed) for s in samples] == [
        ("item_0000", True), ("item_0000", False), ("item_0001", True), ("item_0001", True),
    ]
    assert samples[1].score == 0.0
    assert client.prompts[0] == ("m", "List files", 0.4)


def test_the_judge_adds_a_second_score_without_changing_pass():
    judge = FixedJudge(0.3)

    samples = DatasetBenchmark(RECORDS[:1], ScriptedClient(["ls -la"]), k=1, judge=judge).run(MODEL)

    assert samples[0].passed and samples[0].judge_score == 0.3
    assert judge.seen == [("List files", "ls -la", "ls -la")]


def test_a_model_that_stops_answering_fails_the_run():
    client = ScriptedClient(["ls -la", ModelCallError("m at http://m/v1 did not answer")])

    with pytest.raises(ModelCallError):
        DatasetBenchmark(RECORDS[:1], client, k=2).run(MODEL)


def test_an_empty_dataset_fails_before_any_model_call():
    with pytest.raises(ValueError, match="no held-out records"):
        DatasetBenchmark([], ScriptedClient([]), k=1)


def test_k_must_be_positive():
    with pytest.raises(ValueError, match="k"):
        DatasetBenchmark(RECORDS, ScriptedClient([]), k=0)
