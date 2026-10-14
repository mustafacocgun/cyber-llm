#!/usr/bin/env python3
"""Write an Ollama Modelfile for the fine-tuned GGUF (run on your own machine).

  python3 make_modelfile.py ~/Downloads/<model>.Q4_K_M.gguf
  ollama create cyber-sec -f ~/Downloads/Modelfile && ollama run cyber-sec

TEMPLATE reproduces Foundation-Sec's chat format (<|system|> / <|user|> / <|assistant|>,
with <|end_of_text|> closing each reply), and SYSTEM is the same system_prompt from
mix_config.yaml that every training row used, so inference sees what training saw.
"""
import argparse
import pathlib

from prep import load_config

TEMPLATE = (
    "<|system|>\n{{ .System }}\n"
    '{{ range .Messages }}{{ if eq .Role "user" }}<|user|>\n{{ .Content }}\n'
    '{{ else if eq .Role "assistant" }}<|assistant|>\n{{ .Content }}<|end_of_text|>\n'
    "{{ end }}{{ end }}<|assistant|>\n"
)


def build_modelfile(gguf_name, system_prompt):
    return (
        f"FROM ./{gguf_name}\n"
        f'TEMPLATE """{TEMPLATE}"""\n'
        f'SYSTEM """{system_prompt}"""\n'
        'PARAMETER stop "<|end_of_text|>"\n'
        'PARAMETER stop "<|user|>"\n'
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", help="path to the downloaded .gguf file")
    ap.add_argument("--config", default="mix_config.yaml")
    args = ap.parse_args(argv)

    gguf = pathlib.Path(args.gguf)
    out = gguf.parent / "Modelfile"
    out.write_text(build_modelfile(gguf.name, load_config(args.config)["system_prompt"]))
    print(f"wrote {out}\n  ollama create cyber-sec -f {out} && ollama run cyber-sec")


if __name__ == "__main__":
    main()
