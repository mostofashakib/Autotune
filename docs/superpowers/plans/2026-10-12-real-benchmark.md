# Real Benchmark Implementation Plan

> Native execution, chosen by the user.

**Goal:** Two real models, measured on a held-out dataset or Harbor tasks, compared with pass@k.
**Spec:** `docs/superpowers/specs/2026-10-12-real-benchmark-design.md`
**Tech:** urllib, subprocess, pydantic, typer, FastAPI, pytest.

## Global Constraints
- No new dependencies. Harbor stays an external, optional executable.
- Nothing is reported that a sample did not produce. Failures raise.
- API keys are read from the env var named on the endpoint and never written to the report.

## Review Focus
- An endpoint that is down fails the run instead of scoring zero.
- Harbor trials that crash count as errors, not silent failures.
- A judge that fails mid-run discards the run.
- An empty dataset or task dir fails before model calls.
- Base and trained run on the same tasks with the same k.

### Task 1: domain models and comparison service (`core/domain/models.py`, `core/services/benchmark.py`)
`ModelEndpoint`, `SampleResult`, `ModelScore`, `TaskComparison`, `ComparisonReport`, and `compare(mode, k, base, trained, base_samples, trained_samples)`.

### Task 2: OpenAI-compatible client and reference scorer (`infrastructure/evaluators/openai_client.py`, `scoring.py`)

### Task 3: LLM judge (`infrastructure/evaluators/judge.py`)

### Task 4: dataset runner (`infrastructure/evaluators/dataset_benchmark.py`)

### Task 5: Harbor runner (`infrastructure/evaluators/harbor_benchmark.py`)

### Task 6: port, CLI, delete SideBySideEvaluator (`core/ports/evaluator.py`, `presentation/cli/main.py`)

### Task 7: API job and UI card (`presentation/api/app.py`, `static/`)

### Task 8: README, full suite, real dataset run, real Harbor run

### Task 9: Forge cleanup (Forge repo, branch `refactor/remove-training`)
