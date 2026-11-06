import pytest

from autotune.infrastructure.llm_json import MalformedReply, parse_json_reply


def test_a_plain_object_parses():
    assert parse_json_reply('{"score": 1}') == {"score": 1}


def test_a_fenced_array_parses():
    assert parse_json_reply('```\n[{"a": 1}]\n```') == [{"a": 1}]


def test_an_object_inside_a_tagged_fence_parses():
    assert parse_json_reply('Here:\n```json\n{"score": 0.5}\n```\nbye') == {"score": 0.5}


def test_an_object_inside_prose_parses():
    assert parse_json_reply('Sure: {"score": 1} done') == {"score": 1}


def test_an_array_inside_prose_parses():
    assert parse_json_reply('Pairs: [{"a": 1}] ok') == [{"a": 1}]


def test_text_without_json_is_malformed():
    with pytest.raises(MalformedReply):
        parse_json_reply("no json here")


def test_a_non_string_reply_is_malformed():
    with pytest.raises(MalformedReply):
        parse_json_reply(None)  # type: ignore[arg-type]
