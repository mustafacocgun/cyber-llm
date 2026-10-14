#!/usr/bin/env python3
"""Build data/train.jsonl + data/eval.jsonl from the mix recipe.

  python3 prepare_data.py --smoke --out data/   # offline, bundled fixtures
  python3 prepare_data.py --out data/           # real run (Kaggle: needs `datasets`)

Smoke mode reads fixtures/smoke.json (no network). The real run calls
`datasets.load_dataset`, which is why that library is Kaggle-only.
"""
import argparse
import json
import pathlib

from prep import load_config
from prep.adapters import ADAPTERS
from prep.dedup import extract_ids
from prep.mixer import build_splits
from prep.normalize import to_llama31_messages

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "smoke.json"


def load_source_rows(src, smoke):
    if smoke:
        return json.loads(FIXTURES.read_text()).get(src["name"], [])
    from datasets import load_dataset  # Kaggle-only; absent locally by design
    try:
        return list(load_dataset(src["hf_id"], split=src["split"]))
    except Exception as e:  # messy JSONL repos break HF's whole-file schema cast
        print(f"  {src['name']}: load_dataset failed ({type(e).__name__}); parsing raw files")
        return _load_hf_jsonl(src["hf_id"], src["split"])


def _load_hf_jsonl(hf_id, split):
    """Fallback: download the repo's data files and parse them tolerantly."""
    from huggingface_hub import hf_hub_download, list_repo_files
    exts = (".jsonl", ".json", ".parquet")
    files = list_repo_files(hf_id, repo_type="dataset")
    cands = [f for f in files if split in f.lower() and f.endswith(exts)] \
        or [f for f in files if f.endswith(exts)]
    rows = []
    for f in cands:
        path = hf_hub_download(hf_id, f, repo_type="dataset")
        if f.endswith(".parquet"):
            import pandas as pd
            rows.extend(pd.read_parquet(path).to_dict("records"))
        else:
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass  # skip a corrupt line, keep the rest
    return rows


def _user_text(rec):
    return " ".join(m["content"] for m in rec["messages"] if m["role"] == "user")


def adapt_rows(rows, src):
    """Adapt rows to records, skipping individually-bad rows; raise only if ALL fail."""
    meta = {"source": src["name"], "bucket": src["bucket"], "license": src["license"]}
    recs, skipped, last_err = [], 0, None
    for row in rows:
        try:
            r = dict(row)
            rec = ADAPTERS[src["adapter"]](r, src["field_map"], meta)
            rec["meta"]["ids"] = extract_ids(r, src["field_map"], _user_text(rec))
            recs.append(rec)
        except (KeyError, ValueError, TypeError, IndexError) as e:
            skipped, last_err = skipped + 1, e
    if rows and not recs:
        raise RuntimeError(
            f"source {src['name']!r}: all {len(rows)} rows failed adaptation — "
            f"fix its field_map in mix_config.yaml. Last error: {last_err!r}"
        )
    return recs, skipped


def build_pools(config, smoke):
    pools = {}
    for src in config["sources"]:
        rows = list(load_source_rows(src, smoke))
        recs, skipped = adapt_rows(rows, src)
        if skipped:
            print(f"  {src['name']}: adapted {len(recs)}, skipped {skipped} bad rows")
        pools[src["name"]] = recs
    return pools


def write_jsonl(path, records):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps({"messages": to_llama31_messages(rec), **rec["meta"]}) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="mix_config.yaml")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.get("seed", 0)
    pools = build_pools(config, args.smoke)
    train, eval_ = build_splits(config, pools, seed)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "eval.jsonl", eval_)
    print(f"train={len(train)} eval={len(eval_)} -> {out}/")


if __name__ == "__main__":
    main()
