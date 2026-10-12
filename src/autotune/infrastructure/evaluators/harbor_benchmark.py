"""Harbor mode: run the model as an agent on Harbor tasks and read each trial's verdict.

Harbor is an external tool (`uv tool install harbor`) that runs each task in Docker
and grades it with the task's own verifier. Autotune only invokes it and reads the
`<jobs_dir>/<job_name>/<trial>/result.json` files it writes.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from autotune.core.domain.models import ModelEndpoint, SampleResult

logger = logging.getLogger(__name__)

# (command, environment) -> finished process
Runner = Callable[[list[str], dict[str, str]], subprocess.CompletedProcess]


class HarborError(RuntimeError):
    """Harbor could not run, or ran without producing trial results."""


def _run(command: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, env=env, capture_output=True, text=True, check=False)


class HarborBenchmark:
    def __init__(
        self, task_path: Path, k: int, jobs_dir: Path,
        *, agent: str = "terminus-2", n_concurrent: int = 4,
        executable: str | None = None, runner: Runner | None = None,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if not any(Path(task_path).rglob("task.toml")):
            raise ValueError(f"{task_path} has no Harbor tasks (no task.toml found)")
        resolved = executable or shutil.which("harbor")
        if not resolved:
            raise HarborError("harbor is not installed. Install it with `uv tool install harbor`.")
        self._task_path, self._k, self._jobs_dir = Path(task_path), k, Path(jobs_dir)
        self._agent, self._n_concurrent = agent, n_concurrent
        self._executable, self._runner = resolved, runner or _run

    def run(self, model: ModelEndpoint) -> list[SampleResult]:
        """One sample per trial: k attempts at every task under `task_path`."""
        env = {**os.environ, **_key_env(model)}
        job_name = f"{re.sub(r'[^A-Za-z0-9_.-]+', '-', model.model)}-{uuid.uuid4().hex[:8]}"
        command = [
            self._executable, "run", "-p", str(self._task_path), "-a", self._agent,
            "-m", f"openai/{model.model}", "--ak", f"api_base={model.base_url}",
            "-k", str(self._k), "-n", str(self._n_concurrent),
            "-o", str(self._jobs_dir), "--job-name", job_name, "-y", "-q",
        ]
        logger.info("running harbor job %s for %s", job_name, model.model)
        finished = self._runner(command, env)
        samples = [_sample(path) for path in sorted((self._jobs_dir / job_name).glob("*/result.json"))]
        if not samples:
            raise HarborError(
                f"harbor job {job_name} exited {finished.returncode} with no trial results: "
                f"{(finished.stderr or finished.stdout or '')[-2000:]}"
            )
        if finished.returncode != 0:
            logger.warning("harbor job %s exited %s but wrote %d trials", job_name, finished.returncode, len(samples))
        return samples


def _key_env(model: ModelEndpoint) -> dict[str, str]:
    """Harbor's LiteLLM `openai/` provider reads its key from OPENAI_API_KEY."""
    if not model.api_key_env:
        return {}
    key = os.environ.get(model.api_key_env, "").strip()
    if not key:
        raise HarborError(f"{model.api_key_env} is not set, and {model.model} needs it as its API key")
    return {"OPENAI_API_KEY": key}


def _sample(result_path: Path) -> SampleResult:
    trial: dict[str, Any] = json.loads(result_path.read_text())
    task_id = str(trial.get("task_name") or result_path.parent.name)
    exception = trial.get("exception_info")
    if exception:
        message = exception.get("exception_message") or exception.get("exception_type") or "trial failed"
        return SampleResult(task_id=task_id, passed=False, score=0.0, error=str(message))
    reward = ((trial.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    if not isinstance(reward, (int, float)) or isinstance(reward, bool):
        return SampleResult(task_id=task_id, passed=False, score=0.0, error="no verifier result")
    return SampleResult(task_id=task_id, passed=reward >= 1, score=float(reward))
