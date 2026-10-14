"""Compose sanitizers and persist generated data for reuse by training."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from autotune.core.domain.models import CleanedDataset
from autotune.infrastructure.sanitizers.document_ingestor import DOCUMENT_SUFFIXES, DocumentIngestor, DocumentOptions
from autotune.infrastructure.sanitizers.format_normalizer import AutoFormatSanitizer
from autotune.settings import document_endpoint, setting_value


def load_dataset(path: Path, on_progress: Callable[[dict[str, int]], None] | None = None) -> CleanedDataset:
    """Reuse generated records only when their source bytes have not changed."""
    if path.suffix.lower() not in DOCUMENT_SUFFIXES:
        return AutoFormatSanitizer().sanitize(path)
    cache = path.with_name(path.name + ".cleaned.json")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved["sha256"] == digest:
            if on_progress:
                on_progress(saved["progress"])
            return CleanedDataset.model_validate(saved["dataset"])
    values = {}
    for field in DocumentOptions.model_fields:
        name = "AUTOTUNE_DOCUMENT_" + field.upper()
        value = setting_value(name)
        if value:
            values[field] = int(value)
    progress = {}

    def report(update: dict[str, int]) -> None:
        progress.update(update)
        if on_progress:
            on_progress(update)

    dataset = DocumentIngestor(document_endpoint(), options=DocumentOptions(**values), on_progress=report).sanitize(path)
    payload = {"sha256": digest, "dataset": dataset.model_dump(mode="json"), "progress": progress}
    temporary = cache.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(cache)
    heldout = path.with_name(path.name + ".heldout.jsonl")
    heldout.write_text("".join(json.dumps(r.model_dump(mode="json")) + "\n" for r in dataset.val_records), encoding="utf-8")
    return dataset
