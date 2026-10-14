"""Render canonical records into the Llama-3.1 chat/tool-call shape, and validate.

This is the ONE place tool calls are shaped, so every tool-use dataset ends up
in a single format regardless of how its source wrote them.
"""
import json


def validate_tool_calls(rec):
    """Raise ValueError if any tool call is malformed (missing name / bad args)."""
    for m in rec["messages"]:
        for tc in m.get("tool_calls") or []:
            if not tc.get("name"):
                raise ValueError("tool call missing 'name'")
            args = tc.get("arguments", {})
            if isinstance(args, str):
                try:
                    json.loads(args)
                except json.JSONDecodeError as e:
                    raise ValueError(f"tool call arguments not JSON: {args!r}") from e
            elif not isinstance(args, dict):
                raise ValueError(f"tool call arguments must be dict or JSON string, got {type(args)}")


def to_llama31_messages(rec):
    """Return the record's messages in the structure Llama-3.1's chat template expects."""
    validate_tool_calls(rec)
    out = []
    for m in rec["messages"]:
        msg = {"role": m["role"], "content": m["content"]}
        if m.get("tool_calls"):
            msg["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"] if isinstance(tc["arguments"], dict)
                        else json.loads(tc["arguments"]),
                    },
                }
                for tc in m["tool_calls"]
            ]
        out.append(msg)
    return out
