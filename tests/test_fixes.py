"""Regression tests for the final-review findings (#1 ids, #2 _join drift, #3 MCQ base)."""
import pytest

from prep import load_config
from prep.adapters import ADAPTERS
from prepare_data import build_pools

META = {"source": "s", "bucket": "b", "license": "l"}


# Finding #2 — a renamed/absent column must raise, not drop content silently
def test_reasoning_adapter_raises_on_renamed_column():
    row = {"prompt": "real question", "input": "ctx", "thought": "t", "output": "a"}
    fm = {"user_fields": ["instruction", "input"], "reasoning_field": "thought",
          "assistant_field": "output"}
    with pytest.raises(KeyError):
        ADAPTERS["reasoning"](row, fm, META)


# Finding #3 — answer format is explicit; letter and 1-based are handled, not mis-indexed
def test_mcq_adapter_letter_answer():
    row = {"question": "Q", "choices": ["a", "b", "c", "d"], "answer": "B", "explanation": "x"}
    fm = {"question": "question", "choices": "choices", "answer_idx": "answer",
          "explanation": "explanation", "answer_format": "letter"}
    rec = ADAPTERS["mcq"](row, fm, META)
    assert rec["messages"][-1]["content"].endswith("b")


def test_mcq_adapter_one_based_index():
    row = {"question": "Q", "choices": ["a", "b"], "answer": 1, "explanation": "x"}
    fm = {"question": "question", "choices": "choices", "answer_idx": "answer",
          "explanation": "explanation", "answer_format": "index1"}
    rec = ADAPTERS["mcq"](row, fm, META)
    assert rec["messages"][-1]["content"].endswith("a")   # 1-based 1 -> choices[0]


# Finding #1 — ids are actually populated so strong-key dedup/decontam fire
def test_extract_ids_from_id_fields():
    from prep.dedup import extract_ids
    assert extract_ids({"commit_id": "abc123"}, {"id_fields": {"commit": "commit_id"}}) == {"commit": "abc123"}


def test_extract_ids_cve_from_text():
    from prep.dedup import extract_ids
    assert extract_ids({}, {"extract_cve": True}, "fixes CVE-2021-1234 now") == {"cve": "CVE-2021-1234"}


def test_pipeline_populates_ids_for_vuln_source():
    cfg = load_config("mix_config.yaml")
    pools = build_pools(cfg, smoke=True)
    assert any(r["meta"]["ids"].get("commit") for r in pools["diversevul"])


# Real-data robustness: a few corrupt rows must be skipped, not crash the source
SRC = {"name": "x", "bucket": "b", "license": "l", "adapter": "instruction",
       "field_map": {"user_fields": ["instruction"], "assistant_field": "response"}}


def test_adapt_rows_skips_bad_rows_keeps_good():
    from prepare_data import adapt_rows
    recs, skipped = adapt_rows([{"instruction": "q", "response": "a"}, {"nope": 1}], SRC)
    assert len(recs) == 1 and skipped == 1


def test_adapt_rows_raises_when_all_rows_fail():
    from prepare_data import adapt_rows
    with pytest.raises(RuntimeError):
        adapt_rows([{"nope": 1}, {"nope": 2}], SRC)


def test_mcq_adapter_choices_dict_with_letter():
    row = {"question": "Q", "answers": {"A": "wrong", "B": "right", "C": "x", "D": "y"}, "solution": "B"}
    fm = {"question": "question", "choices_dict": "answers", "answer_idx": "solution",
          "answer_format": "letter"}
    rec = ADAPTERS["mcq"](row, fm, META)
    assert rec["messages"][-1]["content"].endswith("right")


def test_rows_from_json_finds_list_payload():
    from prepare_data import _rows_from_json
    assert _rows_from_json({"questions": [1, 2, 3]}) == [1, 2, 3]
    assert _rows_from_json([{"a": 1}]) == [{"a": 1}]


def test_tool_call_adapter_parses_json_string_fields():
    import json
    row = {"query": "scan host",
           "tools": json.dumps([{"name": "nmap", "parameters": {}}]),
           "answers": json.dumps([{"name": "nmap", "arguments": {"host": "10.0.0.5"}}])}
    fm = {"query": "query", "tools": "tools", "answers": "answers"}
    rec = ADAPTERS["tool_call"](row, fm, META)
    assert rec["messages"][-1]["tool_calls"][0]["arguments"] == {"host": "10.0.0.5"}


def test_build_pools_continues_past_a_failing_source(monkeypatch):
    import prepare_data as pd
    cfg = {"sources": [
        {"name": "good", "adapter": "instruction", "bucket": "b", "license": "l",
         "field_map": {"user_fields": ["instruction"], "assistant_field": "response"}},
        {"name": "bad", "adapter": "instruction", "bucket": "b", "license": "l",
         "field_map": {"user_fields": ["instruction"], "assistant_field": "response"}},
    ], "eval": {"holdout_sources": [], "size": 0}}
    rows = {"good": [{"instruction": "q", "response": "a"}], "bad": [{"nope": 1}]}
    monkeypatch.setattr(pd, "load_source_rows", lambda src, smoke: rows[src["name"]])
    pools = pd.build_pools(cfg, smoke=False)   # must NOT raise
    assert len(pools["good"]) == 1 and pools["bad"] == []
