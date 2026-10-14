"""End-to-end: the files prepare_data writes must render cleanly with Foundation-Sec's
chat template, which only knows system/user/assistant and ignores tool_calls."""
import json

from prep import load_config
from prepare_data import main

ROLES = {"system", "user", "assistant"}


def test_config_has_system_prompt():
    prompt = load_config("mix_config.yaml").get("system_prompt")
    assert isinstance(prompt, str) and len(prompt) > 20


def test_smoke_output_is_template_safe(tmp_path):
    main(["--smoke", "--out", str(tmp_path)])
    prompt = load_config("mix_config.yaml")["system_prompt"]
    for name in ("train.jsonl", "eval.jsonl"):
        rows = [json.loads(line) for line in open(tmp_path / name)]
        assert rows, name
        for row in rows:
            msgs = row["messages"]
            assert msgs[0]["role"] == "system" and msgs[0]["content"].startswith(prompt)
            assert all(m["role"] in ROLES for m in msgs)
            assert all(isinstance(m["content"], str) and m["content"].strip() for m in msgs)
            assert all("tool_calls" not in m for m in msgs)


def test_smoke_tool_rows_carry_explicit_calls(tmp_path):
    main(["--smoke", "--out", str(tmp_path)])
    rows = [json.loads(line) for line in open(tmp_path / "train.jsonl")]
    xlam = [r for r in rows if r["source"] == "xlam"]
    assert xlam and all("<tool_call>" in r["messages"][-1]["content"] for r in xlam)


def test_prepare_data_writes_system_prompt_file(tmp_path):
    main(["--smoke", "--out", str(tmp_path)])
    assert (tmp_path / "system_prompt.txt").read_text() == load_config("mix_config.yaml")["system_prompt"]
