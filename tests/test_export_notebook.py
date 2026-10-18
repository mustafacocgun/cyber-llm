"""export_gguf.ipynb: turn a saved LoRA adapter into a GGUF in a short session (no training)."""
import json
import pathlib

NB = pathlib.Path(__file__).resolve().parent.parent / "export_gguf.ipynb"


def _src():
    nb = json.loads(NB.read_text())
    assert nb.get("nbformat") == 4
    return "\n".join(
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in nb["cells"]
    )


def test_finds_the_saved_adapter_in_attached_input():
    s = _src()
    assert "/kaggle/input" in s and "lora_adapter" in s and "adapter_config.json" in s


def test_does_not_train():
    s = _src()
    assert "SFTTrainer" not in s and ".train(" not in s


def test_uses_foundation_sec_chat_template():
    assert "chat_template.jinja" in _src()


def test_exports_q4_k_m_via_tmp_and_copies_to_output():
    s = _src()
    assert 'OUT_GGUF = "/tmp' in s and "q4_k_m" in s
    assert s.index("save_pretrained_gguf") < s.index('shutil.copy')
    assert "/kaggle/working" in s and "make_modelfile.py" in s
