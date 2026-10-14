"""Format-family adapters: map one dataset's rows into a canonical Record.

Chosen per-source by ``adapter:`` in mix_config.yaml; column names come from
that source's ``field_map``. An adapter raises on a missing expected column,
so a schema drift fails loudly instead of silently emitting empty records.
(The ``mcq`` adapter lives in prep.mcq — it is coupled to the MCQ transform.)
"""
import json

from prep.schema import message, record


def _require(row, key, adapter):
    if key not in row:
        raise KeyError(f"{adapter}: expected column {key!r}, got {sorted(row)}")
    return row[key]


def _join(row, fields):
    missing = [f for f in fields if f not in row]
    if missing:          # an absent column is schema drift, not an empty optional value
        raise KeyError(f"_join: missing columns {missing}, got {sorted(row)}")
    parts = [str(row[f]).strip() for f in fields if row.get(f) not in (None, "")]
    if not parts:
        raise ValueError(f"no content in fields {fields}")
    return "\n\n".join(parts)


def adapt_instruction(row, fm, meta):
    msgs = []
    sys_f = fm.get("system_field")
    if sys_f and row.get(sys_f):
        msgs.append(message("system", str(row[sys_f])))
    msgs.append(message("user", _join(row, fm["user_fields"])))
    msgs.append(message("assistant", str(_require(row, fm["assistant_field"], "instruction"))))
    return record(msgs, **meta)


def adapt_reasoning(row, fm, meta):
    user = _join(row, fm["user_fields"])
    reasoning = str(_require(row, fm["reasoning_field"], "reasoning")).strip()
    answer = str(_require(row, fm["assistant_field"], "reasoning")).strip()
    assistant = f"<think>\n{reasoning}\n</think>\n\n{answer}"
    return record([message("user", user), message("assistant", assistant)], **meta)


def adapt_messages_passthrough(row, fm, meta):
    conv = _require(row, fm["conversations"], "messages_passthrough")
    role_key, content_key = fm["role_key"], fm["content_key"]
    role_map = fm.get("role_map", {})
    msgs = [message(role_map.get(t[role_key], t[role_key]), str(t[content_key])) for t in conv]
    return record(msgs, **meta)


def _maybe_json(value):
    """Some datasets (e.g. xLAM) store list columns as JSON text — parse those."""
    return json.loads(value) if isinstance(value, str) else value


def adapt_tool_call(row, fm, meta):
    query = str(_require(row, fm["query"], "tool_call"))
    tools = _maybe_json(_require(row, fm["tools"], "tool_call"))
    answers = _maybe_json(_require(row, fm["answers"], "tool_call"))
    tool_calls = [{"name": a["name"], "arguments": a.get("arguments", {})} for a in answers]
    msgs = [
        message("system", "You can call these tools:\n" + json.dumps(tools)),
        message("user", query),
        message("assistant", "", tool_calls=tool_calls),
    ]
    return record(msgs, **meta)


def adapt_vuln_code(row, fm, meta):
    func = str(_require(row, fm["func"], "vuln_code"))
    label = _require(row, fm["label"], "vuln_code")
    vulnerable = bool(int(label)) if str(label).isdigit() else bool(label)
    cwe = str(row.get(fm.get("cwe", ""), "")).strip()
    commit = str(row.get(fm.get("commit_msg", ""), "")).strip()
    user = ("Is the following function vulnerable? If so, name the CWE and explain.\n\n"
            f"```c\n{func}\n```")
    if vulnerable:
        answer = f"Yes — this function is vulnerable ({cwe or 'unspecified CWE'})."
        if commit:
            answer += f" The fix: {commit}"
    else:
        answer = "No — no vulnerability is evident in this function."
    return record([message("user", user), message("assistant", answer)], **meta)


ADAPTERS = {
    "instruction": adapt_instruction,
    "reasoning": adapt_reasoning,
    "messages_passthrough": adapt_messages_passthrough,
    "tool_call": adapt_tool_call,
    "vuln_code": adapt_vuln_code,
}

# mcq lives in prep.mcq (coupled to the MCQ->reasoning transform); register it here.
from prep.mcq import adapt_mcq  # noqa: E402

ADAPTERS["mcq"] = adapt_mcq
