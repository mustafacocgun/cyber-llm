# cyber-llm

Build a local, security-specialized LLM: curate a ~25k-example training mix, QLoRA fine-tune
**Foundation-Sec-8B-Instruct** on Kaggle's free T4, and run the result on an RTX 4060 (8 GB)
via Ollama. See [`DATA_PLAN.md`](DATA_PLAN.md) for the datasets, mix, licenses, and pipeline.

## Layout

```
mix_config.yaml       the dataset-mix recipe (edit this to rebalance)
prep/                 offline data engineering (stdlib + pyyaml only)
prepare_data.py       CLI: build data/train.jsonl + data/eval.jsonl
train_unsloth.ipynb   Kaggle QLoRA notebook -> GGUF
make_modelfile.py     writes the Ollama Modelfile locally (matches the training format)
tests/                pytest; uses inline fixtures, no downloads
```

## Local dev (works offline, on a weak box)

```bash
python3 -m pytest -q          # runs the whole suite; no dataset downloads
python3 prepare_data.py --smoke --out data/   # end-to-end on bundled fixtures
```

Only `pytest` + `pyyaml` are needed locally — both already present. The heavy libraries
(`datasets`, `unsloth`, ...) are used only on Kaggle.

## Real run (on Kaggle — fast network, free GPU)

1. **Prep (CPU notebook):** run `prepare_data.py` with internet on; "Save & Run All" to publish
   `data/` as a Kaggle Dataset. (HF token as a Kaggle Secret for gated datasets.)
2. **Train (GPU T4 notebook):** `train_unsloth.ipynb` reads that Dataset, fine-tunes, and exports
   a GGUF. Download the ~5 GB GGUF once.
3. **Local:** `python3 make_modelfile.py ~/Downloads/<file>.gguf`, then
   `ollama create cyber-sec -f ~/Downloads/Modelfile && ollama run cyber-sec`.

## Scope

Personal / local use. Standard dataset-engineering and fine-tuning — nothing operational.
