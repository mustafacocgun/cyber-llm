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
