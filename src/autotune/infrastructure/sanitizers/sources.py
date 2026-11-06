"""Read any upload into ready records plus raw texts, and assemble the final dataset."""
from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from autotune.core.domain.limits import MAX_FILE_BYTES, MAX_UPLOAD_BYTES
from autotune.core.domain.models import CleanedDataset, DatasetRecord, TaskType
from autotune.core.ports.sanitizer import IngestionError
from autotune.infrastructure.sanitizers.document_text import DOCUMENT_SUFFIXES, extract_text
from autotune.infrastructure.sanitizers.format_normalizer import STRUCTURED_SUFFIXES, normalize_file, tail_split

logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = DOCUMENT_SUFFIXES | STRUCTURED_SUFFIXES | {".zip"}
MAX_DOCUMENT_BYTES = MAX_FILE_BYTES
MAX_ARCHIVE_BYTES = MAX_UPLOAD_BYTES
_SYSTEM_FILES = {"thumbs.db", "desktop.ini"}


@dataclass(frozen=True)
class Sources:
    """Records ready for training, texts that need pair generation, and what was dropped on the way."""
    task_type: TaskType | None = None
    records: list[DatasetRecord] = field(default_factory=list)
    documents: list[tuple[str, str]] = field(default_factory=list)
    dropped: int = 0
    warnings: list[str] = field(default_factory=list)


def collect_sources(path: Path) -> Sources:
    """Read one file or a zip archive. Archive members that fail are skipped with a warning."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise IngestionError(f"Unsupported file format: {suffix or path.name}. Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}")
    if path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise IngestionError("Upload exceeds the 100 MiB limit")
    if suffix != ".zip":
        if path.stat().st_size > MAX_DOCUMENT_BYTES:
            raise IngestionError(f"{path.name} exceeds the 20 MiB per-file limit")
        return _merge([_read_file(path.name, path.read_bytes(), in_archive=False)], [])
    members, warnings = _archive_members(path)
    parts = []
    for name, data in members:
        try:
            parts.append(_read_file(name, data, in_archive=True))
        except IngestionError as exc:
            warnings.append(f"Skipped archive member: {exc}")
    if not parts:
        raise IngestionError(f"No usable files in {path.name}. " + " ".join(warnings))
    return _merge(parts, warnings)


def _archive_members(path: Path) -> tuple[list[tuple[str, bytes]], list[str]]:
    """Safe, supported members in name order. Unsafe paths fail the whole upload."""
    with zipfile.ZipFile(path) as archive:
        files = sorted((i for i in archive.infolist() if not i.is_dir()), key=lambda i: i.filename)
        if not files or len(files) > 1000 or sum(i.file_size for i in files) > MAX_ARCHIVE_BYTES:
            raise IngestionError("Archive must contain 1 to 1000 files totaling at most 100 MiB")
        seen = set()
        for item in files:
            name = PurePosixPath(item.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in item.filename or item.filename in seen:
                raise IngestionError(f"Unsafe archive path: {item.filename}")
            seen.add(item.filename)
        system = [i for i in files if _is_system_file(PurePosixPath(i.filename))]
        warnings = [f"Skipped {len(system)} system files such as .DS_Store and __MACOSX entries."] if system else []
        members = []
        for item in files:
            suffix = PurePosixPath(item.filename).suffix.lower()
            if item in system:
                continue
            if suffix not in DOCUMENT_SUFFIXES | STRUCTURED_SUFFIXES:
                warnings.append(f"Skipped archive member: {item.filename} has unsupported type {suffix or '(none)'}.")
            elif item.file_size > MAX_DOCUMENT_BYTES:
                warnings.append(f"Skipped archive member: {item.filename} exceeds the 20 MiB per-file limit.")
            else:
                members.append((item.filename, archive.read(item)))
        return members, warnings


def _is_system_file(name: PurePosixPath) -> bool:
    return "__MACOSX" in name.parts or name.name.startswith(".") or name.name.lower() in _SYSTEM_FILES


def _read_file(name: str, data: bytes, *, in_archive: bool) -> Sources:
    if PurePosixPath(name).suffix.lower() in DOCUMENT_SUFFIXES:
        return Sources(documents=[(name, extract_text(name, data))])
    parsed = normalize_file(name, data)
    records = parsed.records
    if in_archive:
        records = [r.model_copy(update={"metadata": {**r.metadata, "source_file": name}}) for r in records]
    return Sources(parsed.task_type if records else None, records, parsed.texts, parsed.dropped, parsed.warnings)


def _merge(parts: list[Sources], warnings: list[str]) -> Sources:
    types = {p.task_type for p in parts if p.task_type is not None}
    if any(p.documents for p in parts):
        types.add(TaskType.CHAT)
    return Sources(
        task_type=_combined_type(types),
        records=[r for p in parts for r in p.records],
        documents=[d for p in parts for d in p.documents],
        dropped=sum(p.dropped for p in parts),
        warnings=[*warnings, *(w for p in parts for w in p.warnings)],
    )


def _combined_type(types: set[TaskType]) -> TaskType | None:
    """Chat and tool-use data train together. Any other mix needs separate uploads."""
    if types <= {TaskType.CHAT, TaskType.TOOL_USE}:
        return TaskType.TOOL_USE if TaskType.TOOL_USE in types else (TaskType.CHAT if types else None)
    if len(types) == 1:
        return next(iter(types))
    names = ", ".join(sorted(t.value for t in types))
    raise IngestionError(f"The upload mixes {names} data. Upload each kind of data separately.")


def assemble_dataset(sources: Sources, generated: list[DatasetRecord], dropped: int, warnings: list[str],
                     val_split_ratio: float = 0.1) -> CleanedDataset:
    """One CleanedDataset from every surviving record, split by source file when sources are known."""
    records = [*sources.records, *generated]
    if not records:
        raise IngestionError("No usable training records survived. " + " ".join([*sources.warnings, *warnings]))
    warnings = [*sources.warnings, *warnings]
    if all("source_file" in r.metadata for r in records):
        train, val, split_warnings = _split_by_source(records, val_split_ratio)
        warnings += split_warnings
    else:
        train, val = tail_split(records, val_split_ratio)
    return CleanedDataset(task_type=sources.task_type or TaskType.CHAT, total_records=len(records), train_records=train,
                          val_records=val, dropped_count=sources.dropped + dropped, warnings=warnings)


def _split_by_source(records: list[DatasetRecord], ratio: float) -> tuple[list[DatasetRecord], list[DatasetRecord], list[str]]:
    warnings = []
    key = "source_file"
    if len({r.metadata["source_file"] for r in records}) == 1:
        key = "chunk_id"
        warning = "Only one document was supplied. Splitting by chunk instead of source document. Overlapping chunks might share text."
        logger.warning(warning)
        warnings.append(warning)
    groups = list(dict.fromkeys(r.metadata.get(key, r.id) for r in records))
    if ratio == 0 or len(groups) < 2:
        warnings.append("No held-out split: at least two surviving source groups are required.")
        return records, [], warnings
    count = min(len(groups) - 1, max(1, int(len(groups) * ratio)))
    heldout = set(groups[-count:])
    return ([r for r in records if r.metadata.get(key, r.id) not in heldout],
            [r for r in records if r.metadata.get(key, r.id) in heldout], warnings)
