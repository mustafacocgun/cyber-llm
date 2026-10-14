"""Canonical record types shared by every adapter.

A Record is ``{"messages": [...], "meta": {...}}``. A Message is
``{"role", "content"}`` plus an optional ``"tool_calls"`` list of
``{"name", "arguments"}``. Everything downstream (normalize, dedup, mixer)
speaks this shape, so each messy source is handled in exactly one adapter.
"""

ROLES = {"system", "user", "assistant", "tool"}


def message(role, content, tool_calls=None):
    if role not in ROLES:
        raise ValueError(f"bad role: {role!r}")
    m = {"role": role, "content": content}
    if tool_calls is not None:
        m["tool_calls"] = tool_calls
    return m


def record(messages, source, bucket, license, ids=None):
    if not messages:
        raise ValueError(f"empty messages from source {source!r}")
    return {
        "messages": messages,
        "meta": {"source": source, "bucket": bucket, "license": license, "ids": ids or {}},
    }
