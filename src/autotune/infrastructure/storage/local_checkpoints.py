"""Checkpoints on the local disk, with GGUF export through a llama.cpp checkout."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from autotune.core.domain.errors import MissingExtraError, ProviderError
from autotune.core.ports.storage import CheckpointStorage
from autotune.infrastructure.storage.checkpoint import MANIFEST_NAME
from autotune.settings import llama_cpp_dir

TRAINING_EXTRA_MESSAGE = "Training dependencies missing. Run: uv sync --extra training"


def run_converter(args: list[str]) -> None:
    """Run llama.cpp's converter with this interpreter."""
    try:
        subprocess.run([sys.executable, *args], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise ProviderError(f"llama.cpp conversion failed: {exc.stderr.strip() or exc}") from exc


def merge_adapter(adapter_dir: Path, merged_dir: Path) -> None:
    """Merge the LoRA adapter into its base model and save the full model and tokenizer."""
    try:
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise MissingExtraError(TRAINING_EXTRA_MESSAGE) from exc
    AutoPeftModelForCausalLM.from_pretrained(str(adapter_dir)).merge_and_unload().save_pretrained(str(merged_dir))
    AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(str(merged_dir))


class LocalCheckpointStorage(CheckpointStorage):
    """Keeps checkpoints in local directories. The converter and merge step are injectable for tests."""

    def __init__(
        self,
        convert: Callable[[list[str]], None] | None = None,
        merge: Callable[[Path, Path], None] | None = None,
    ) -> None:
        self._convert = convert or run_converter
        self._merge = merge or merge_adapter

    def save(self, job_id: str, source_dir: Path, target_dir: Path) -> Path:
        """Copy the checkpoint to `target_dir` unless it is already there. Returns the manifest path."""
        source, target = Path(source_dir), Path(target_dir)
        if source.resolve() != target.resolve():
            shutil.copytree(source, target, dirs_exist_ok=True)
        return target / MANIFEST_NAME

    def export_gguf(self, checkpoint_dir: Path, output_file: Path) -> Path:
        script = llama_cpp_dir() / "convert_hf_to_gguf.py"
        output = Path(output_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="autotune-merged-") as merged:
            self._merge(Path(checkpoint_dir), Path(merged))
            self._convert([str(script), merged, "--outfile", str(output)])
        return output
