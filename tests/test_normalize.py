import pytest

from prep.normalize import to_llama31_messages, validate_tool_calls
from prep.schema import message, record


def _rec(msgs):
    return record(msgs, source="s", bucket="b", license="apache-2.0")


def test_to_llama31_wraps_tool_calls():
    rec = _rec([message("user", "go"),
                message("assistant", "", tool_calls=[{"name": "nmap", "arguments": {"t": "x"}}])])
    out = to_llama31_messages(rec)
    fn = out[-1]["tool_calls"][0]
    assert fn["type"] == "function"
    assert fn["function"]["name"] == "nmap"
    assert fn["function"]["arguments"] == {"t": "x"}


def test_to_llama31_preserves_plain_messages():
    rec = _rec([message("user", "hi"), message("assistant", "hello")])
    out = to_llama31_messages(rec)
    assert out == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


def test_validate_rejects_nameless_call():
    rec = _rec([message("assistant", "", tool_calls=[{"arguments": {}}])])
    with pytest.raises(ValueError):
        validate_tool_calls(rec)


def test_validate_rejects_non_json_string_args():
    rec = _rec([message("assistant", "", tool_calls=[{"name": "x", "arguments": "{not json"}])])
    with pytest.raises(ValueError):
        validate_tool_calls(rec)
