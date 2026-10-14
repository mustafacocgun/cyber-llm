"""Render canonical records into the format Foundation-Sec's chat template understands.

That template (chat_template.jinja in fdtn-ai/Foundation-Sec-8B-Instruct) knows only
the system/user/assistant roles and silently ignores `tool_calls` and the `tool` role.
So this is the ONE place tool use is shaped:
  - assistant tool calls  -> <tool_call>{"name": ..., "arguments": {...}}</tool_call> text
  - tool output           -> a user turn wrapped in <tool_response>...</tool_response>
  - every row             -> one system prompt we control (the model's built-in default
                             is ~600 tokens and tells it to refuse non-security questions)
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


def _call_text(tc):
    args = tc.get("arguments", {})
    if isinstance(args, str):
        args = json.loads(args)
    return "<tool_call>" + json.dumps({"name": tc["name"], "arguments": args}) + "</tool_call>"


def to_template_messages(rec, system_prompt):
    """Return [system, user/assistant...] messages with plain-string content only.

    Raises ValueError for malformed tool calls or an empty turn (training on an empty
    reply would teach the model to answer with nothing).
    """
    validate_tool_calls(rec)
    msgs = rec["messages"]
    system = system_prompt
    if msgs and msgs[0]["role"] == "system":
        system += "\n\n" + msgs[0]["content"]
        msgs = msgs[1:]

    out = [{"role": "system", "content": system}]
    for m in msgs:
        if m["role"] == "tool":
            out.append({"role": "user", "content": f"<tool_response>\n{m['content']}\n</tool_response>"})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            calls = "\n".join(_call_text(tc) for tc in m["tool_calls"])
            text = m["content"].strip()
            out.append({"role": "assistant", "content": f"{text}\n{calls}" if text else calls})
        else:
            out.append({"role": m["role"], "content": m["content"]})

    for m in out[1:]:
        if not str(m["content"]).strip():
            raise ValueError(f"empty {m['role']} turn")
    return out
