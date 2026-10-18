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
2. Data: optionally **Add Input** → your prep notebook's output. If nothing is attached, the
   notebook builds the data itself (attach the `HF_TOKEN` secret so the two gated datasets load).
3. Open `train_unsloth.ipynb`. Run with `SMOKE = True` **first** — 50 rows / 10 steps, a few
   minutes. Then set `SMOKE = False` and use **Save & Run All**: the full run (~20.8k rows,
   ~9–10.5 h on a T4) continues in the background, so you can close the browser.
4. From the run's **Output**: download the `.gguf` (~5 GB, one time, resumable). Keep
   `lora_adapter/` too — it is the training result, and the GGUF can be re-exported from it.

## If the run stops before the GGUF is saved

Training the full mix on a T4 takes ~11.7 h (~32.5 s/step), close to Kaggle's 12-hour limit, so
the export can get cut off. The adapter is saved right after training, so nothing is lost:
open `export_gguf.ipynb` in a new notebook, **Add Input** → the training run's output (with
`lora_adapter`), GPU T4, Internet On, then **Save & Run All (Commit)** (~30–40 min). It loads
the adapter, exports Q4_K_M, and puts the `.gguf` in its Output tab.

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
