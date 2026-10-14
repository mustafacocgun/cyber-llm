import pytest

from prep.adapters import ADAPTERS
from prep.mcq import mcq_to_record

META = {"source": "s", "bucket": "b", "license": "apache-2.0"}


def test_mcq_maps_to_answer_text_not_letter():
    rec = mcq_to_record("Q?", ["wrong", "right", "no", "nope"], 1, "because reasons", META)
    ans = rec["messages"][-1]["content"]
    assert "right" in ans
    assert "because reasons" in ans
    assert "B)" not in ans and "answer_idx" not in ans


def test_mcq_rejects_out_of_range_index():
    with pytest.raises(ValueError):
        mcq_to_record("Q?", ["a", "b"], 5, "x", META)


def test_mcq_adapter_registered_and_works():
    row = {"question": "What is XSS?", "choices": ["a", "cross-site scripting", "c", "d"],
           "answer_idx": 1, "explanation": "it injects script"}
    fm = {"question": "question", "choices": "choices", "answer_idx": "answer_idx",
          "explanation": "explanation"}
    rec = ADAPTERS["mcq"](row, fm, META)
    assert "cross-site scripting" in rec["messages"][-1]["content"]
