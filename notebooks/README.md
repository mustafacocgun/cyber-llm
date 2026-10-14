# Running on Kaggle (free T4)

Two notebooks: a **CPU** prep notebook (downloads + builds the data) and the **GPU** training
notebook. Keeping prep on CPU saves your ~30 GPU-hrs/week for actual training.

## 1. Prep (CPU notebook) → a Kaggle Dataset

1. New Kaggle notebook, accelerator **None (CPU)**, **Internet: On**.
2. Add your HuggingFace token as a **Kaggle Secret** (for gated datasets); `huggingface_hub login`.
3. `pip install datasets huggingface_hub pyyaml`, then run `prepare_data.py` (the repo's files
   pasted in or added as a dataset). It writes `data/train.jsonl` + `data/eval.jsonl`.
4. **Save & Run All** → the output becomes a Kaggle Dataset you can attach to the training notebook.

## 2. Train (GPU T4 notebook) → GGUF

1. New notebook, accelerator **GPU T4**, **Internet: On**.
2. Attach the Dataset from step 1 (so `data/train.jsonl` is available).
3. Open `train_unsloth.ipynb`. Run with `SMOKE = True` **first** — it does 50 rows / 10 steps and
   exercises the whole path (load → train → merge → GGUF) in minutes. If that produces a `.gguf`,
   set `SMOKE = False` and run the full ~25k training.
4. Download the `.gguf` + `Modelfile` from the output (~5 GB, one time, resumable).

## 3. Local (your RTX 4060)

```bash
ollama create cyber-sec -f Modelfile
ollama run cyber-sec
```

## Gotchas (already handled in the notebook)

- **Merge before GGUF** (Unsloth #11698) — the notebook merges explicitly first.
- **Chat-template match** — `prepare_data.py` normalized every row to Llama-3.1, and the notebook
  applies the base model's own template, so they agree.
- **Schema drift** — if an adapter raises "expected column X", fix that source's `field_map` in
  `mix_config.yaml` and re-run prep. (Expected on first contact with live data.)
