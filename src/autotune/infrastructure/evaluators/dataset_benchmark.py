"""Held-out dataset mode: answer each prompt k times and score against the reference."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from autotune.core.domain.models import ModelEndpoint, SampleResult, SFTRecord
from autotune.infrastructure.evaluators.judge import ChatClient
from autotune.infrastructure.evaluators.scoring import reference_score


class Judge(Protocol):
    def score(self, prompt: str, reference: str, answer: str) -> float: ...


class DatasetBenchmark:
    def __init__(
        self, records: Sequence[SFTRecord], client: ChatClient, k: int,
        *, judge: Judge | None = None, temperature: float = 0.7, max_tokens: int = 1024,
    ) -> None:
        if not records:
            raise ValueError("the dataset has no held-out records to benchmark")
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        self._records, self._client, self._k = list(records), client, k
        self._judge, self._temperature, self._max_tokens = judge, temperature, max_tokens

    def run(self, model: ModelEndpoint) -> list[SampleResult]:
        """k graded samples per record. Any model or judge failure stops the run."""
        samples = []
        for index, record in enumerate(self._records):
            for _ in range(self._k):
                answer = self._client.complete(
                    model, [{"role": "user", "content": record.prompt}],
                    temperature=self._temperature, max_tokens=self._max_tokens,
                )
                passed, score = reference_score(answer, record.completion)
                judge_score = self._judge.score(record.prompt, record.completion, answer) if self._judge else None
                samples.append(SampleResult(
                    task_id=f"item_{index:04d}", passed=passed, score=score, judge_score=judge_score,
                ))
        return samples
