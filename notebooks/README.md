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
2. **Add Input** → your prep notebook's output. The staging cell copies it from
   `/kaggle/input/...` into `data/`.
3. Open `train_unsloth.ipynb`. Run with `SMOKE = True` **first** — it does 50 rows / 10 steps and
   exercises the whole path (load → train → merge → GGUF) in minutes. If that produces a `.gguf`,
   set `SMOKE = False` and run the full ~25k training.
4. Download the `.gguf` from the output (~5 GB, one time, resumable).

## 3. Local (your RTX 4060)

```bash
python3 make_modelfile.py ~/Downloads/<file>.gguf   # inside the cyber-llm repo
ollama create cyber-sec -f ~/Downloads/Modelfile
ollama run cyber-sec
```

## Gotchas (already handled in the notebook)

- **Merge before GGUF** (Unsloth #11698) — the notebook merges explicitly first.
- **Chat-template match** — `prepare_data.py` writes only system/user/assistant turns (tool calls as
  `<tool_call>` text), which is all Foundation-Sec's own template understands.
- **Schema drift** — if an adapter raises "expected column X", fix that source's `field_map` in
  `mix_config.yaml` and re-run prep. (Expected on first contact with live data.)
