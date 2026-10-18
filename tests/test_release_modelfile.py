"""release/Modelfile is a convenience copy for the exported GGUF; it must match make_modelfile.py."""
import pathlib

from make_modelfile import build_modelfile
from prep import load_config

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_release_modelfile_is_in_sync_with_config():
    expected = build_modelfile("Foundation-Sec-8B-Instruct.Q4_K_M.gguf",
                               load_config(ROOT / "mix_config.yaml")["system_prompt"])
    assert (ROOT / "release" / "Modelfile").read_text() == expected
