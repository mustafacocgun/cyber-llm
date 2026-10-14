from make_modelfile import build_modelfile, main
from prep import load_config


def test_modelfile_matches_foundation_sec_chat_format():
    mf = build_modelfile("m.gguf", "SYS PROMPT")
    assert mf.startswith("FROM ./m.gguf\n")
    for needle in ("<|system|>", "<|user|>", "<|assistant|>", "<|end_of_text|>",
                   'SYSTEM """SYS PROMPT"""', 'PARAMETER stop "<|end_of_text|>"'):
        assert needle in mf, needle


def test_main_writes_modelfile_next_to_gguf_with_training_prompt(tmp_path):
    gguf = tmp_path / "cyber.Q4_K_M.gguf"
    gguf.write_bytes(b"")
    main([str(gguf)])
    mf = (tmp_path / "Modelfile").read_text()
    assert "FROM ./cyber.Q4_K_M.gguf" in mf
    assert load_config("mix_config.yaml")["system_prompt"] in mf
