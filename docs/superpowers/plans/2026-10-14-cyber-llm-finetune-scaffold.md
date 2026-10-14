# Cyber-LLM Fine-tune Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local project that turns the researched dataset shortlist into a clean ~25k-example training mix and a Kaggle-ready QLoRA notebook that produces a GGUF model runnable on an RTX 4060 8GB.

**Architecture:** A small Python package (`prep/`) does the data engineering — per-source adapters normalize every dataset into one canonical `messages`/tool-call record, then dedup → MCQ-conversion → sampling → train/eval split → JSONL. A separate Kaggle notebook (`train_unsloth.ipynb`) fine-tunes Foundation-Sec-8B-Instruct with Unsloth QLoRA and exports a merged GGUF + Ollama Modelfile. The mix recipe lives in `mix_config.yaml` (data), never hardcoded in Python.

**Tech Stack:** Python 3.10+, HuggingFace `datasets` + `huggingface_hub`, `pyyaml`, `pandas`, `pytest` (local); `unsloth`, `transformers`, `trl`, `bitsandbytes` (notebook/Kaggle only).

**Spec:** This conversation's research synthesis, captured as `DATA_PLAN.md` in Task 1 (the plan argues from it; executors read both).

## Global Constraints

- Base model: `fdtn-ai/Foundation-Sec-8B-Instruct` (Llama-3.1-8B lineage).
- Tool-call format: **Llama-3.1** — every tool-use dataset normalizes to it; no mixed syntaxes in the output.
- Train target: **~25,000** train examples + a disjoint held-out eval set; recipe/counts defined only in `mix_config.yaml`.
- Train platform: Kaggle free **T4 (16GB)**, ~30 GPU-hrs/week. Local inference: RTX 4060 **8GB**, GGUF **Q4_K_M** via Ollama.
- Merge step runs on **CPU** (T4 VRAM too small for fp16 8B merge) and **before** GGUF export (works around Unsloth issue #11698, which silently exports the un-merged base).
- Use case: **personal/local** — non-commercial datasets are permitted; `meta.license` is recorded per record so a commercial subset can be filtered later.
- Commits are **local only** (no push), and commit messages carry **no Claude attribution** lines (standing user rule overrides the attribution reminder).
- Numbers (fees, counts, sample sizes, licenses) live in config/data files, not in `.py` files.
- **All large downloads happen on Kaggle's network, not the user's** (poor home connection): `prepare_data.py` runs in a CPU-only Kaggle notebook that downloads the raw datasets and saves `train/eval.jsonl` as a Kaggle Dataset; the GPU training notebook pulls the ~16 GB base model on Kaggle too. The only file crossing the user's own connection is the final ~5 GB GGUF, once (resumable). Unit tests use inline samples and need no network, so code can be built and tested locally.

## Review Focus

- **Upstream schema drift:** a dataset's HF columns differ from what its adapter expects → the adapter must raise a clear error, never emit a record with empty `messages`. (Test in Task 2.)
- **Malformed tool calls:** a tool-use row with missing/non-JSON arguments → `validate_tool_calls` raises and the row is dropped, not rendered into a broken Llama-3.1 structure. (Test in Task 2.)
- **Cross-source key extraction:** CVE-id lives in one source, commit-hash in another → `strong_keys` must pull the right key per source or duplicates leak into eval. (Test in Task 3.)
- **MCQ off-by-one / shuffled options:** `answer_idx` out of range or options reordered → conversion maps to the correct answer *text*, never a wrong letter. (Test in Task 3.)
- **Eval contamination:** a train record sharing a CVE/commit id with an eval record → `build_splits` removes it from train; paraphrase-level overlap (no shared id) is a documented limitation, not silently assumed solved. (Test in Task 4.)

---

### Task 1: Project skeleton, mix config, and DATA_PLAN.md

**Files:**
- Create: `/home/thegg/cyber-llm/README.md`, `requirements.txt`, `.gitignore`, `DATA_PLAN.md`, `mix_config.yaml`, `prep/__init__.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `mix_config.yaml` with keys `train_target: int`, `seed: int`, `sources: list[{name, hf_id, split, adapter, bucket, license, sample}]`, `eval: {holdout_sources: list[str], size: int}`. Consumed by Tasks 2–4.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
def test_mix_config_sums_to_target():
    cfg = load_config("mix_config.yaml")
    total = sum(s["sample"] for s in cfg["sources"] if s["name"] not in cfg["eval"]["holdout_sources"])
    assert abs(total - cfg["train_target"]) <= 2000      # within tolerance of ~25k
    assert len({s["name"] for s in cfg["sources"]}) == len(cfg["sources"])   # unique names
    assert every source has a non-empty license string
```

- [ ] **Step 2: Run test, verify it fails** — `pytest tests/test_config.py -v` → FAIL (`mix_config.yaml` missing).
- [ ] **Step 3: Write `mix_config.yaml`** with the shortlist from `DATA_PLAN.md` (SEvenLLM 4000, xLAM 3000, cyberdata-2-large 3000, pentesting-explanations 3000, security-agentic 2500, Fenrir 2500, ToolACE 1500, vuln-detect 1500, Primus-Reasoning 1000, CyberMetric 1000, general-nonsec 1000, Primus-Instruct 835, pentest-checklist 160), each with `hf_id/split/adapter/bucket/license`; `eval.holdout_sources` = CyberMetric-80 split, SecQA, PrimeVul paired-test.
- [ ] **Step 4: Write `DATA_PLAN.md`** (reference doc: shortlist table, mix, licenses, 4 hard rules, train→local pipeline) and `README.md` (quickstart), `requirements.txt`, `.gitignore` (ignore `data/`, `models/`, `*.gguf`, `.venv`). Add `load_config` to `prep/__init__.py`.
- [ ] **Step 5: Run test, verify it passes.**
- [ ] **Step 6: Commit** — `git init && git add -A && git commit -m "scaffold: project, mix config, data plan"` (local, no attribution line).

---

### Task 2: Canonical schema, per-source adapters, Llama-3.1 normalization

**Files:**
- Create: `prep/schema.py`, `prep/adapters.py`, `prep/normalize.py`
- Test: `tests/test_adapters.py`, `tests/test_normalize.py`

**Interfaces:**
- Produces:
  - `schema.Record = TypedDict("Record", {"messages": list[Message], "meta": Meta})`; `Message = {"role": str, "content": str, "tool_calls": list | None}`.
  - `adapters.ADAPTERS: dict[str, Callable[[dict], Record]]` — one `adapt_<name>(row) -> Record` per source; each sets `meta.source/bucket/license` and strong-key ids where present.
  - `normalize.to_llama31_messages(rec: Record) -> list[dict]`; `normalize.validate_tool_calls(rec: Record) -> None` (raises `ValueError` on malformed calls).
- Consumes: `mix_config.yaml` source rows (adapter names must match `ADAPTERS` keys).

- [ ] **Step 1: Write failing tests** — for a representative row of each adapter, assert the Record has non-empty `messages`, correct roles, and populated `meta`; assert a row with a missing expected column raises `KeyError`/`ValueError`. For normalize: a tool-call Record renders to Llama-3.1 tool syntax; a malformed `tool_calls` (non-dict args) makes `validate_tool_calls` raise.

```python
# tests/test_adapters.py
def test_xlam_adapter_produces_tool_call_record():
    rec = ADAPTERS["xlam"](SAMPLE_XLAM_ROW)
    assert rec["messages"][0]["role"] == "user"
    assert rec["messages"][-1]["tool_calls"][0]["name"] == "search_cve"
    assert rec["meta"]["bucket"] == "tool_use"

def test_adapter_raises_on_schema_drift():
    with pytest.raises((KeyError, ValueError)):
        ADAPTERS["xlam"]({"unexpected": "shape"})
```

- [ ] **Step 2: Run tests, verify they fail.**
- [ ] **Step 3: Implement** `schema.py`, each `adapt_*` in `adapters.py`, and `normalize.py`. Adapters map each source's documented columns → canonical Record; `to_llama31_messages` converts canonical `tool_calls` into Llama-3.1's expected message structure.
- [ ] **Step 4: Run tests, verify they pass.**
- [ ] **Step 5: Commit** — `feat: canonical schema, source adapters, llama-3.1 normalization`.

---

### Task 3: Dedup and MCQ-to-reasoning conversion

**Files:**
- Create: `prep/dedup.py`, `prep/mcq.py`
- Test: `tests/test_dedup.py`, `tests/test_mcq.py`

**Interfaces:**
- Produces:
  - `dedup.strong_keys(rec: Record) -> set[str]` (CVE-ids, commit-hashes from `meta.ids`); `dedup.text_hash(rec: Record) -> str`; `dedup.dedup(records: list[Record]) -> list[Record]` (drops later records sharing any strong key or an identical text hash).
  - `mcq.mcq_to_record(question, choices, answer_idx, explanation, source) -> Record` — assistant turn is short reasoning then the answer *text*.
- Consumes: `schema.Record`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dedup.py
def test_dedup_removes_shared_cve_across_sources():
    out = dedup([rec_with_cve("CVE-2021-1"), rec_with_cve("CVE-2021-1")])
    assert len(out) == 1

# tests/test_mcq.py
def test_mcq_maps_to_answer_text_not_letter():
    rec = mcq_to_record("Q?", ["wrong","right","no","nope"], answer_idx=1, explanation="because", source="cybermetric")
    ans = rec["messages"][-1]["content"]
    assert "right" in ans and "B)" not in ans and "answer_idx" not in ans

def test_mcq_rejects_out_of_range_index():
    with pytest.raises(ValueError):
        mcq_to_record("Q?", ["a","b"], answer_idx=5, explanation="x", source="s")
```

- [ ] **Step 2: Run tests, verify they fail.**
- [ ] **Step 3: Implement** `dedup.py` and `mcq.py`.
- [ ] **Step 4: Run tests, verify they pass.**
- [ ] **Step 5: Commit** — `feat: dedup by strong key + text hash, mcq-to-reasoning conversion`.

---

### Task 4: Mixer, eval split, and the prepare_data CLI

**Files:**
- Create: `prep/mixer.py`, `prepare_data.py`
- Test: `tests/test_mixer.py`

**Interfaces:**
- Produces:
  - `mixer.build_splits(config: dict, pools: dict[str, list[Record]], seed: int) -> tuple[list[Record], list[Record]]` — samples each source to its `sample` count, assembles train, builds eval from `holdout_sources`, then removes any train record sharing a strong key with eval; deterministic for a fixed seed.
  - `prepare_data.main(argv)` — CLI: `--config`, `--out`, `--seed`, `--smoke`; loads/downloads via `datasets`, routes rows through `ADAPTERS` → `validate_tool_calls` → `to_llama31_messages`, runs `dedup`, then `build_splits`, writing `data/train.jsonl` and `data/eval.jsonl`.
- Consumes: Tasks 1–3.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mixer.py
def test_build_splits_counts_and_determinism():
    train, ev = build_splits(CFG, POOLS, seed=42)
    assert abs(len(train) - CFG["train_target"]) <= 2000
    train2, _ = build_splits(CFG, POOLS, seed=42)
    assert [r["meta"]["ids"] for r in train] == [r["meta"]["ids"] for r in train2]

def test_eval_disjoint_from_train_by_strong_key():
    train, ev = build_splits(CFG, POOLS, seed=42)
    tk = set().union(*(strong_keys(r) for r in train))
    ek = set().union(*(strong_keys(r) for r in ev))
    assert tk.isdisjoint(ek)
```

- [ ] **Step 2: Run tests, verify they fail.**
- [ ] **Step 3: Implement** `mixer.py` and `prepare_data.py` (the `--smoke` path samples ~50 rows/source for a fast end-to-end check).
- [ ] **Step 4: Run tests, verify they pass** (unit tests use inline sample rows — no dataset downloads, so this works locally on a poor connection). Run `python prepare_data.py --smoke --out data/` locally to confirm valid JSONL. The **full** run (real `load_dataset` downloads) executes in a CPU-only Kaggle notebook, saving `data/` as a Kaggle Dataset.
- [ ] **Step 5: Commit** — `feat: mixer with decontaminated eval split + prepare_data CLI`.

---

### Task 5: Kaggle QLoRA notebook (train → CPU-merge → GGUF → Modelfile)

**Files:**
- Create: `train_unsloth.ipynb`, `notebooks/README.md` (how to run on Kaggle)
- Test: smoke run (manual, documented) — no pytest unit test (GPU/notebook step)

**Interfaces:**
- Consumes: `data/train.jsonl`, `data/eval.jsonl` — the Kaggle Dataset produced by the CPU prep notebook (no upload from the laptop needed).
- Produces: `model.gguf` (Q4_K_M) + `Modelfile` artifacts for download.

- [ ] **Step 1: Write the notebook cells** — (1) install Unsloth + deps; (2) load `fdtn-ai/Foundation-Sec-8B-Instruct` 4-bit; (3) attach LoRA (r=32, alpha=32, all attn+MLP target modules, gradient checkpointing); (4) load JSONL, apply the model's chat template to the Llama-3.1 `messages`; (5) `SFTTrainer` (seq-len 2048, batch 2 × grad-accum 8, LR 2e-4 cosine, 2 epochs, `paged_adamw_8bit`, bf16/fp16); (6) quick eval loss on `data/eval.jsonl`; (7) `merge_and_unload()` on **CPU** then `save_pretrained_gguf(..., quantization_method="q4_k_m")`; (8) write/verify the Ollama `Modelfile` (chat template embedded).
- [ ] **Step 2: Add a `SMOKE = True` switch** at the top — 50 examples, `max_steps=10` — so the whole path (load → train → merge → GGUF) can be validated in minutes before the full run.
- [ ] **Step 3: Document the Kaggle run** in `notebooks/README.md`: enable GPU T4, phone-verify, upload `data/` as a dataset, run SMOKE first, then full run; note the CPU-merge rationale and the #11698 merge-before-export gotcha.
- [ ] **Step 4: Verify** — record that a SMOKE run completes end-to-end and emits a non-empty `.gguf` + `Modelfile`.
- [ ] **Step 5: Commit** — `feat: kaggle unsloth qlora notebook with cpu-merge gguf export`.

---

## Self-Review

**1. Spec coverage:** DATA_PLAN.md + mix_config (Task 1) ✓ deliverable 1; adapters/normalize/dedup/mcq/mixer/CLI (Tasks 2–4) ✓ deliverable 2; notebook (Task 5) ✓ deliverable 3. Kaggle-T4 / CPU-merge / Llama-3.1 format / 8GB-local constraints all pinned in Global Constraints and Task 5. No gaps found.

**2. Step scan:** Each code step names a signature + file; bodies shown only for test assertions and the notebook's cell list (which is the decision, not boilerplate). No "handle edge cases" filler.

**3. Type consistency:** `Record`/`Message`/`Meta` defined in Task 2 and reused verbatim in Tasks 3–4; `strong_keys`, `dedup`, `build_splits`, `ADAPTERS`, `to_llama31_messages`, `validate_tool_calls` names match across tasks.

**4. Review Focus:** schema drift → Task 2 test; malformed tool calls → Task 2 test; cross-source keys → Task 3 test; MCQ off-by-one → Task 3 test; eval contamination → Task 4 test. Paraphrase-level contamination recorded as a known limitation.

**5. Proportion:** Plan is near the size of the code it specifies; most length is in interface/config decisions, not transcribed bodies.
