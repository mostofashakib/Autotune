"""One decoder reads text bytes for every ingestion format."""
import json

import pytest

from autotune.core.domain.models import TaskType


@pytest.mark.parametrize("data, expected", [
    ("café".encode("utf-8"), "café"),
    (b"\xef\xbb\xbfcaf\xc3\xa9", "café"),
    ("café".encode("utf-16"), "café"),
    ("café".encode("latin-1"), "café"),
])
def test_decode_text_tries_utf8_then_utf16_bom_then_latin1(data, expected):
    from autotune.infrastructure.sanitizers.text_decoding import decode_text
    assert decode_text(data) == expected


def test_declared_charset_wins_over_utf8_guess():
    from autotune.infrastructure.sanitizers.text_decoding import decode_text
    assert decode_text("Привет".encode("cp1251"), declared="windows-1251") == "Привет"


def test_unknown_declared_charset_falls_back():
    from autotune.infrastructure.sanitizers.text_decoding import decode_text
    assert decode_text(b"plain", declared="not-a-codec") == "plain"


@pytest.mark.parametrize("markup, charset", [
    (b'<meta charset="iso-8859-1"><p>x</p>', "iso-8859-1"),
    (b"<meta http-equiv='Content-Type' content='text/html; charset=windows-1251'>", "windows-1251"),
    (b"<p>no charset</p>", None),
])
def test_html_charset_reads_meta_tag(markup, charset):
    from autotune.infrastructure.sanitizers.text_decoding import html_charset
    assert html_charset(markup) == charset


@pytest.mark.parametrize("name, data", [
    ("notes.txt", "café menu".encode("latin-1")),
    ("notes.txt", "café menu".encode("utf-16")),
    ("page.html", b'<meta charset="iso-8859-1"><p>caf\xe9 menu</p>'),
])
def test_documents_decode_non_utf8_text(name, data):
    from autotune.infrastructure.sanitizers.document_text import extract_text
    assert "café menu" in extract_text(name, data)


@pytest.mark.parametrize("name, data", [
    ("rows.csv", "prompt,completion\ncafé,olé\n".encode("latin-1")),
    ("rows.jsonl", b'\xef\xbb\xbf{"prompt": "caf\xc3\xa9", "completion": "ol\xc3\xa9"}\n'),
    ("rows.json", json.dumps([{"prompt": "café", "completion": "olé"}]).encode("utf-16")),
])
def test_structured_files_decode_non_utf8_text(tmp_path, name, data):
    from autotune.infrastructure.sanitizers.format_normalizer import AutoFormatSanitizer
    path = tmp_path / name
    path.write_bytes(data)
    dataset = AutoFormatSanitizer().sanitize(path, val_split_ratio=0)
    assert dataset.task_type == TaskType.CHAT
    assert [m.content for m in dataset.train_records[0].messages] == ["café", "olé"]
