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


def test_an_array_followed_by_a_bracket_in_prose_keeps_the_array():
    assert parse_json_reply('[{"a":1}] see [1]') == [{"a": 1}]


def test_the_first_of_two_objects_in_prose_is_returned():
    assert parse_json_reply('Result: {"a": 1} and {"b": 2}') == {"a": 1}


def test_a_fence_followed_by_prose_returns_the_fenced_value():
    assert parse_json_reply('```json\n{"a": 1}\n```\nSee {braces} here') == {"a": 1}


def test_nested_braces_parse():
    assert parse_json_reply('x {"a": {"b": [1, {"c": 2}]}} y') == {"a": {"b": [1, {"c": 2}]}}


def test_braces_inside_strings_parse():
    assert parse_json_reply('x {"a": "}{"} y') == {"a": "}{"}


def test_a_non_string_reply_is_malformed():
    with pytest.raises(MalformedReply):
        parse_json_reply(None)  # type: ignore[arg-type]
