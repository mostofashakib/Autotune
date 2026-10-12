# Real Benchmark

Date: 2026-10-12
Status: Approved in chat

## Goal

Replace `SideBySideEvaluator`, which reports hardcoded scores (base 0.65, trained 0.92), with a benchmark that measures two real models and reports only what it measured.

## Boundary

Autotune never imports or calls Forge. It reads files:

- exported datasets (`sft_pairs.jsonl`, `messages` JSONL, and the other formats `load_sft` already reads)
- Harbor task directories, written by Forge, taken from the Harbor hub, or authored by hand

## Models

A run compares two models, base and trained. Each is a `ModelEndpoint`: an OpenAI-compatible base URL, a model name, and an optional API key env var name. vLLM, RunPod, or Ollama can serve them. Model calls use `urllib`, with no new dependencies.

## Mode 1: held-out dataset

- Each record gives a prompt and a reference answer (`load_sft`).
- Each model answers every prompt `k` times (`/chat/completions`).
- The reference scorer normalizes both texts (lowercase, collapse whitespace, strip surrounding punctuation) and computes exact match and token F1. A sample passes on exact match or F1 >= 0.8. Its score is the F1.
- Optional LLM judge: a third endpoint scores each answer against the reference from 0 to 1 and returns JSON. It is a second score, reported beside the reference score. If the judge is on and any judge call fails or returns malformed output, the run fails with no report.

## Mode 2: Harbor tasks

- For each model, run `harbor run -p <tasks> -a terminus-2 -m openai/<model> --ak api_base=<url> -k <k> -o <jobs_dir> --job-name <name> -y` as a subprocess. Harbor is an optional tool. A missing `harbor` executable fails with an install hint.
- Each trial's `result.json` is parsed. A trial passes when `verifier_result.rewards.reward >= 1`. Its score is that reward.
- A trial with `exception_info` or no verifier result counts as a failure and is also counted as an error.
- A Harbor exit with no trial results fails the run.

## Report

`ComparisonReport`, written to `<output>/benchmark_report.json`:

- mode, k, the two endpoints (no keys), timestamps
- per model: samples, passed, errors, pass@1, pass@k, mean score, mean judge score (dataset mode with a judge)
- per task: the same numbers for each model, plus trained minus base deltas for pass@1 and mean score
- overall deltas for pass@1, pass@k, and mean score

pass@1 and pass@k are the unbiased estimator (`compute_pass_at_k`) averaged over tasks. Every number comes from recorded samples.

## Entry points

- CLI: `autotune benchmark dataset` and `autotune benchmark harbor` replace `autotune evaluate`.
- API: `POST /api/benchmark` starts a background job and `GET /api/benchmark/{job_id}` reports status, logs, and the report. `/api/evaluate` and its hardcoded fallback are deleted.
- UI: the Evaluate card takes the mode, both endpoints, the data or task path, and `k`, and shows the measured report.

## Errors

- A model endpoint that refuses or times out fails the run, naming the endpoint.
- An empty dataset or task directory fails before any model call.
- `k` must be at least 1.

## Testing

- Unit: scorer, judge parsing and failure, OpenAI client (fake transport), dataset runner (fake client), Harbor runner (fake subprocess writing real-shaped result files), comparison math, CLI, API.
- Real: dataset mode against two local Ollama models, and Harbor mode against one real task in Docker.

## After it works: Forge cleanup

Delete Forge's benchmark: `forge/benchmark/`, the benchmark API, worker, CLI, and `/benchmark` UI with its nav and home entries, `InferenceModePicker`, `experiments.py` and its YAML, the checkpoint contract and `policy_loader`, the GPU inference contract and gateway with their `/api/distributed` hardware and inference endpoints, and `env_quality.py`. `compiled_tasks`' loader moves to the task factory.
