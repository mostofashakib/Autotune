"""Any upload becomes CleanedDataset records. Bad parts are skipped and reported, never fatal alone."""
import io
import json
import zipfile
from unittest.mock import Mock

import pytest

from autotune.core.domain.models import ModelEndpoint, TaskType
from autotune.core.ports.sanitizer import IngestionError

SUPPORTED = '{"supported": true}'


def pair(topic):
    return json.dumps([{"question": f"Describe the {topic} policy", "answer": f"The {topic} policy requires daily review."}])


def policy(topic):
    return f"The {topic} policy requires daily review."


def ingestor(replies):
    from autotune.infrastructure.sanitizers.document_ingestor import DocumentIngestor
    client = Mock()
    client.complete.side_effect = replies
    return DocumentIngestor(ModelEndpoint(base_url="http://localhost:11434/v1", model="local"), client=client)


def archive(tmp_path, members, name="upload.zip"):
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as z:
        for member, data in members.items():
            z.writestr(member, data)
    return path


def write(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    return path


# Structured files report every malformed line.

def test_malformed_json_lines_are_counted_and_reported(tmp_path):
    from autotune.infrastructure.sanitizers.format_normalizer import AutoFormatSanitizer
    rows = '{"prompt": "a", "completion": "b"}\n{bad\n{"prompt": "c", "completion": "d"}\nnope\n'
    dataset = AutoFormatSanitizer().sanitize(write(tmp_path, "rows.jsonl", rows))
    assert dataset.total_records == 2
    assert dataset.dropped_count == 2
    assert any("2 malformed JSON lines" in w and "2, 4" in w for w in dataset.warnings)


# The generator step retries malformed replies once, then skips the chunk.

def test_fenced_json_reply_is_accepted(tmp_path):
    fenced = "```json\n" + pair("safety") + "\n```"
    result = ingestor([fenced, "```\n" + SUPPORTED + "\n```"]).sanitize(write(tmp_path, "p.md", policy("safety")))
    assert result.total_records == 1


def test_malformed_generator_reply_is_retried_once(tmp_path):
    result = ingestor(["not json", pair("safety"), SUPPORTED]).sanitize(write(tmp_path, "p.md", policy("safety")))
    assert result.total_records == 1


def test_chunk_is_skipped_after_two_malformed_replies(tmp_path):
    path = archive(tmp_path, {"a.md": policy("access"), "b.md": policy("billing")})
    result = ingestor(["not json", "[1]", pair("billing"), SUPPORTED]).sanitize(path)
    assert result.total_records == 1
    assert result.dropped_count == 1
    assert any("a.md:0" in w for w in result.warnings)


def test_grounding_reply_is_retried_once(tmp_path):
    result = ingestor([pair("safety"), "not json", SUPPORTED]).sanitize(write(tmp_path, "p.md", policy("safety")))
    assert result.total_records == 1


def test_upload_fails_only_when_nothing_survives(tmp_path):
    with pytest.raises(IngestionError, match="No grounded"):
        ingestor(["bad", "bad"]).sanitize(write(tmp_path, "p.md", policy("safety")))


# Archives skip junk, empty, and unsupported members, and accept structured files.

def test_zip_skips_system_files_with_a_warning(tmp_path):
    path = archive(tmp_path, {"__MACOSX/._a.md": b"\x00\x05", ".DS_Store": b"x", "docs/Thumbs.db": b"x", "a.md": policy("safety")})
    result = ingestor([pair("safety"), SUPPORTED]).sanitize(path)
    assert result.total_records == 1
    assert any("Skipped 3 system files" in w for w in result.warnings)


@pytest.mark.parametrize("bad_member, data", [("empty.txt", b""), ("photo.png", b"\x89PNG"), ("broken.pdf", b"%PDF garbage")])
def test_zip_skips_unusable_member_with_a_warning(tmp_path, bad_member, data):
    path = archive(tmp_path, {"a.md": policy("safety"), bad_member: data})
    result = ingestor([pair("safety"), SUPPORTED]).sanitize(path)
    assert result.total_records == 1
    assert any(bad_member in w for w in result.warnings)


def test_zip_with_no_usable_member_fails_with_reasons(tmp_path):
    path = archive(tmp_path, {"empty.txt": b"", "photo.png": b"\x89PNG"})
    with pytest.raises(IngestionError, match="empty.txt"):
        ingestor([]).sanitize(path)


def test_mixed_zip_combines_structured_rows_and_generated_pairs(tmp_path):
    path = archive(tmp_path, {"rows.csv": "prompt,completion\nhi,there\n", "policy.md": policy("safety")})
    result = ingestor([pair("safety"), SUPPORTED]).sanitize(path)
    records = result.train_records + result.val_records
    assert result.task_type == TaskType.CHAT
    assert sorted(r.metadata["source_file"] for r in records) == ["policy.md", "rows.csv"]


def test_zip_of_structured_files_needs_no_generator(tmp_path, monkeypatch):
    from autotune.presentation.api.dataset_ingestion import load_dataset
    monkeypatch.setenv("AUTOTUNE_DOCUMENT_MODEL", "")
    path = archive(tmp_path, {"a.jsonl": '{"prompt": "a", "completion": "b"}\n', "b.csv": "question,answer\nc,d\n"})
    assert load_dataset(path).total_records == 2


def test_zip_mixing_preference_and_chat_data_is_rejected(tmp_path):
    preference = json.dumps({"prompt": "p", "chosen": "good", "rejected": "bad"})
    path = archive(tmp_path, {"prefs.jsonl": preference, "policy.md": policy("safety")})
    with pytest.raises(IngestionError, match="mix"):
        ingestor([pair("safety"), SUPPORTED]).sanitize(path)


# Structured files holding raw text go to the generator.

@pytest.mark.parametrize("column", ["text", "content", "body", "document"])
def test_text_column_rows_become_generated_pairs(tmp_path, column):
    rows = f"{column}\n{policy('safety')}\n{policy('billing')}\n"
    result = ingestor([pair("safety"), SUPPORTED, pair("billing"), SUPPORTED]).sanitize(write(tmp_path, "notes.csv", rows))
    records = result.train_records + result.val_records
    assert result.total_records == 2
    assert {r.metadata["source_file"] for r in records} == {"notes.csv#row1", "notes.csv#row2"}


def test_jsonl_text_rows_become_generated_pairs(tmp_path):
    rows = json.dumps({"text": policy("safety")}) + "\n"
    assert ingestor([pair("safety"), SUPPORTED]).sanitize(write(tmp_path, "notes.jsonl", rows)).total_records == 1


def test_conversation_rows_win_over_a_text_column(tmp_path, monkeypatch):
    from autotune.presentation.api.dataset_ingestion import load_dataset
    monkeypatch.setenv("AUTOTUNE_DOCUMENT_MODEL", "")
    path = write(tmp_path, "rows.csv", "prompt,completion,text\nhi,there,extra\n")
    assert load_dataset(path).train_records[0].messages[1].content == "there"


def test_text_rows_without_generator_name_the_setting(tmp_path, monkeypatch):
    from autotune.presentation.api.dataset_ingestion import load_dataset
    monkeypatch.setenv("AUTOTUNE_DOCUMENT_MODEL", "")
    with pytest.raises(ValueError, match="AUTOTUNE_DOCUMENT_MODEL"):
        load_dataset(write(tmp_path, "notes.csv", f"text\n{policy('safety')}\n"))


def test_htm_is_a_document(tmp_path):
    result = ingestor([pair("safety"), SUPPORTED]).sanitize(write(tmp_path, "p.htm", f"<p>{policy('safety')}</p>"))
    assert result.total_records == 1


# Extraction keeps reading order and drops markup.

def test_docx_tables_stay_in_document_order():
    from docx import Document
    from autotune.infrastructure.sanitizers.document_text import extract_text
    document = Document()
    document.add_paragraph("Intro")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Name", "Value"
    document.add_paragraph("Outro")
    buffer = io.BytesIO()
    document.save(buffer)
    assert extract_text("order.docx", buffer.getvalue()) == "Intro\nName | Value\nOutro"


def test_markdown_markup_is_stripped():
    from autotune.infrastructure.sanitizers.document_text import extract_text
    source = ("# Title\n\nSome **bold** and _italic_ text with a [link](http://example.com) and ![a chart](chart.png).\n"
              "> Quoted `inline` code\n\n- item one\n\n---\n\n```python\nprint(1)\n```\n")
    assert extract_text("notes.md", source.encode()) == (
        "Title\n\nSome bold and italic text with a link and a chart.\nQuoted inline code\n\n- item one\n\nprint(1)")


# Forge export bundles report environment quality.

def forge_trajectories():
    episodes = [
        {"episode_id": "e1", "env": "mail", "task": "t", "seed": 1, "steps": [
            {"step_index": 0, "action": {"endpoint": "/a"}, "reward": 0.0, "terminated": False, "truncated": False},
            {"step_index": 1, "action": {"endpoint": "/a"}, "reward": 1.0, "terminated": True, "truncated": False}]},
        {"episode_id": "e2", "env": "mail", "task": "t", "seed": 2, "steps": [
            {"step_index": 0, "action": {"endpoint": "/b"}, "reward": 0.0, "terminated": False, "truncated": False},
            {"step_index": 1, "action": {"endpoint": "/c"}, "reward": 0.0, "terminated": True, "truncated": False}]},
    ]
    return "".join(json.dumps(e) + "\n" for e in episodes)


def forge_rollouts():
    import pandas as pd
    rows = [
        {"episode_id": "e1", "env_name": "mail", "task_name": "t", "prompt": "p", "completion": "c", "total_reward": 1.0,
         "passed": True, "termination_reason": "success", "per_step_rewards": [0.0, 1.0]},
        {"episode_id": "e2", "env_name": "mail", "task_name": "t", "prompt": "p", "completion": "c", "total_reward": 0.0,
         "passed": False, "termination_reason": "dead_end", "per_step_rewards": [0.0, 0.0]},
    ]
    buffer = io.BytesIO()
    pd.DataFrame(rows).to_parquet(buffer)
    return buffer.getvalue()


SFT_ROWS = '{"prompt": "a", "completion": "b"}\n{"prompt": "c", "completion": "d"}\n'


def test_forge_bundle_upload_reports_environment_quality(tmp_path):
    from autotune.infrastructure.sanitizers.sources import collect_sources
    bundle = archive(tmp_path, {"trajectories.jsonl": forge_trajectories(), "grpo_rollouts.parquet": forge_rollouts()})
    sources = collect_sources(bundle)
    [metrics] = sources.environment_quality
    assert (metrics.env_name, metrics.num_episodes, metrics.num_steps) == ("mail", 2, 4)
    assert (metrics.reward_density, metrics.action_diversity, metrics.dead_end_rate) == (0.25, 0.75, 0.5)
    assert metrics.state_coverage_score is None
    assert len(sources.records) == 2  # the reward rows in the rollouts file still train


def test_trajectories_beside_sft_pairs_still_trains(tmp_path):
    from autotune.presentation.api.dataset_ingestion import load_dataset
    dataset = load_dataset(archive(tmp_path, {"sft_pairs.jsonl": SFT_ROWS, "trajectories.jsonl": forge_trajectories()}))
    assert dataset.total_records == 2
    assert dataset.environment_quality[0].num_steps == 4


def test_lone_trajectories_upload_has_nothing_to_train_on(tmp_path):
    from autotune.presentation.api.dataset_ingestion import load_dataset
    with pytest.raises(IngestionError, match="environment quality only"):
        load_dataset(write(tmp_path, "trajectories.jsonl", forge_trajectories()))


def test_rollouts_upload_trains_and_reports_environment_quality(tmp_path):
    from autotune.presentation.api.dataset_ingestion import load_dataset
    dataset = load_dataset(write(tmp_path, "grpo_rollouts.parquet", forge_rollouts()))
    assert dataset.task_type == TaskType.RL_GRPO
    [metrics] = dataset.environment_quality
    assert (metrics.num_episodes, metrics.num_steps, metrics.dead_end_rate) == (2, 0, 0.5)


def forge_preferences():
    row = {"task": "t", "env_name": "mail", "chosen": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "good"}],
           "rejected": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "bad"}],
           "chosen_reward": 1.0, "rejected_reward": 0.0}
    return json.dumps(row) + "\n"


def test_full_forge_bundle_trains_on_one_member_and_reports_quality(tmp_path):
    from autotune.presentation.api.dataset_ingestion import load_dataset
    bundle = archive(tmp_path, {"sft_pairs.jsonl": SFT_ROWS, "preference_pairs.jsonl": forge_preferences(),
                                "grpo_rollouts.parquet": forge_rollouts(), "trajectories.jsonl": forge_trajectories()})
    dataset = load_dataset(bundle)
    assert dataset.task_type == TaskType.CHAT
    assert dataset.total_records == 2
    [metrics] = dataset.environment_quality
    assert (metrics.num_steps, metrics.dead_end_rate) == (4, 0.5)
    assert any("Trained on sft_pairs.jsonl from this Forge bundle. Skipped" in w
               and "preference_pairs.jsonl" in w and "grpo_rollouts.parquet" in w for w in dataset.warnings)


def test_forge_bundle_without_sft_trains_on_the_next_member_in_priority(tmp_path):
    from autotune.infrastructure.sanitizers.sources import collect_sources
    bundle = archive(tmp_path, {"preference_pairs.jsonl": forge_preferences(), "grpo_rollouts.parquet": forge_rollouts(),
                                "trajectories.jsonl": forge_trajectories()})
    sources = collect_sources(bundle)
    assert sources.task_type == TaskType.PREFERENCE
    assert any("Trained on preference_pairs.jsonl" in w for w in sources.warnings)
    assert sources.environment_quality


def test_zip_that_is_not_a_forge_bundle_still_rejects_mixed_data(tmp_path):
    from autotune.infrastructure.sanitizers.sources import collect_sources
    bundle = archive(tmp_path, {"chat.jsonl": SFT_ROWS, "grpo_rollouts.parquet": forge_rollouts()})
    with pytest.raises(IngestionError, match="mixes"):
        collect_sources(bundle)
