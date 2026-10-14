import json
import pathlib

NB = pathlib.Path(__file__).resolve().parent.parent / "train_unsloth.ipynb"


def _src():
    nb = json.loads(NB.read_text())
    assert nb.get("nbformat") == 4
    return "\n".join(
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in nb["cells"]
    )


def test_notebook_targets_the_base_model():
    assert "fdtn-ai/Foundation-Sec-8B-Instruct" in _src()


def test_notebook_has_smoke_switch_and_data_paths():
    s = _src()
    assert "SMOKE" in s
    assert "data/train.jsonl" in s and "data/eval.jsonl" in s


def test_notebook_merges_before_gguf_export():
    s = _src()
    assert "q4_k_m" in s
    assert s.index("save_pretrained_merged") < s.index("save_pretrained_gguf")


def test_notebook_stages_attached_kaggle_data():
    s = _src()
    assert "/kaggle/input" in s and "shutil.copy" in s


def test_notebook_strips_duplicate_bos_and_has_no_stale_format_claim():
    s = _src()
    assert "removeprefix(tokenizer.bos_token" in s
    assert "Llama-3.1)" not in s


def test_notebook_points_to_local_modelfile_script():
    assert "make_modelfile.py" in _src()
