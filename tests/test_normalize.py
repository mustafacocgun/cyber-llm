import json

import pytest

from prep.normalize import to_template_messages, validate_tool_calls
from prep.schema import message, record

SYS = "You are a cybersecurity assistant."


def _rec(msgs):
    return record(msgs, source="s", bucket="b", license="apache-2.0")


def test_prepends_system_prompt_when_missing():
    out = to_template_messages(_rec([message("user", "hi"), message("assistant", "hello")]), SYS)
    assert out == [{"role": "system", "content": SYS},
                   {"role": "user", "content": "hi"},
                   {"role": "assistant", "content": "hello"}]


def test_existing_system_is_appended_to_ours():
    out = to_template_messages(_rec([message("system", "Tools: [nmap]"), message("user", "go"),
                                     message("assistant", "ok")]), SYS)
    assert out[0] == {"role": "system", "content": SYS + "\n\nTools: [nmap]"}
    assert sum(m["role"] == "system" for m in out) == 1


def test_tool_calls_become_explicit_text_including_parallel_calls():
    rec = _rec([message("user", "scan"),
                message("assistant", "", tool_calls=[{"name": "nmap", "arguments": {"host": "x"}},
                                                    {"name": "whois", "arguments": '{"d": "y"}'}])])
    a = to_template_messages(rec, SYS)[-1]
    assert a["role"] == "assistant" and "tool_calls" not in a
    calls = [json.loads(s.split("</tool_call>")[0]) for s in a["content"].split("<tool_call>")[1:]]
    assert calls == [{"name": "nmap", "arguments": {"host": "x"}},
                     {"name": "whois", "arguments": {"d": "y"}}]


def test_tool_results_become_user_turns():
    rec = _rec([message("user", "scan"),
                message("assistant", "", tool_calls=[{"name": "nmap", "arguments": {}}]),
                message("tool", "22/tcp open"), message("assistant", "SSH is open.")])
    out = to_template_messages(rec, SYS)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user", "assistant"]
    assert out[3]["content"] == "<tool_response>\n22/tcp open\n</tool_response>"


def test_tool_call_without_arguments_key_renders():
    rec = _rec([message("user", "go"), message("assistant", "", tool_calls=[{"name": "ping"}])])
    assert '"arguments": {}' in to_template_messages(rec, SYS)[-1]["content"]


def test_empty_assistant_turn_is_rejected():
    with pytest.raises(ValueError):
        to_template_messages(_rec([message("user", "q"), message("assistant", "  ")]), SYS)


def test_validate_rejects_nameless_call():
    rec = _rec([message("assistant", "", tool_calls=[{"arguments": {}}])])
    with pytest.raises(ValueError):
        validate_tool_calls(rec)


def test_validate_rejects_non_json_string_args():
    rec = _rec([message("assistant", "", tool_calls=[{"name": "x", "arguments": "{not json"}])])
    with pytest.raises(ValueError):
        validate_tool_calls(rec)
