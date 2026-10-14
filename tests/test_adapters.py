import pytest

from prep.adapters import ADAPTERS

META = {"source": "s", "bucket": "b", "license": "apache-2.0"}


def test_instruction_adapter_builds_user_assistant():
    row = {"instruction": "Explain SQLi", "context": "", "response": "It is injection."}
    fm = {"user_fields": ["instruction", "context"], "assistant_field": "response"}
    rec = ADAPTERS["instruction"](row, fm, META)
    assert [m["role"] for m in rec["messages"]] == ["user", "assistant"]
    assert rec["messages"][1]["content"] == "It is injection."
    assert rec["meta"]["source"] == "s"


def test_instruction_adapter_prepends_system():
    row = {"system": "You are a SOC analyst.", "user": "triage this", "assistant": "ok"}
    fm = {"system_field": "system", "user_fields": ["user"], "assistant_field": "assistant"}
    rec = ADAPTERS["instruction"](row, fm, META)
    assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]


def test_reasoning_adapter_wraps_thought():
    row = {"instruction": "Q", "input": "", "thought": "step1", "output": "ans"}
    fm = {"user_fields": ["instruction", "input"], "reasoning_field": "thought", "assistant_field": "output"}
    rec = ADAPTERS["reasoning"](row, fm, META)
    a = rec["messages"][1]["content"]
    assert "step1" in a and a.rstrip().endswith("ans")


def test_messages_passthrough_maps_roles():
    row = {"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]}
    fm = {"conversations": "conversations", "role_key": "from", "content_key": "value",
          "role_map": {"human": "user", "gpt": "assistant"}}
    rec = ADAPTERS["messages_passthrough"](row, fm, META)
    assert [m["role"] for m in rec["messages"]] == ["user", "assistant"]


def test_tool_call_adapter_extracts_calls():
    row = {"query": "scan example.com",
           "tools": [{"name": "nmap", "description": "scan", "parameters": {}}],
           "answers": [{"name": "nmap", "arguments": {"target": "example.com"}}]}
    fm = {"query": "query", "tools": "tools", "answers": "answers"}
    rec = ADAPTERS["tool_call"](row, fm, META)
    assert rec["messages"][-1]["tool_calls"][0]["name"] == "nmap"
    assert rec["messages"][0]["role"] == "system"


def test_vuln_code_adapter_flags_vulnerable():
    row = {"func": "strcpy(a,b);", "target": 1, "cwe": "CWE-120", "message": "use strncpy"}
    fm = {"func": "func", "label": "target", "cwe": "cwe", "commit_msg": "message"}
    rec = ADAPTERS["vuln_code"](row, fm, META)
    assert "CWE-120" in rec["messages"][1]["content"]


def test_adapter_raises_on_schema_drift():
    fm = {"query": "query", "tools": "tools", "answers": "answers"}
    with pytest.raises((KeyError, ValueError)):
        ADAPTERS["tool_call"]({"unexpected": 1}, fm, META)
