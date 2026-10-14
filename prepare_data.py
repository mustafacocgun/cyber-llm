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
    return load_dataset(src["hf_id"], split=src["split"])


def _user_text(rec):
    return " ".join(m["content"] for m in rec["messages"] if m["role"] == "user")


def build_pools(config, smoke):
    pools = {}
    for src in config["sources"]:
        meta = {"source": src["name"], "bucket": src["bucket"], "license": src["license"]}
        recs = []
        for row in load_source_rows(src, smoke):
            rec = ADAPTERS[src["adapter"]](row, src["field_map"], meta)
            rec["meta"]["ids"] = extract_ids(row, src["field_map"], _user_text(rec))
            recs.append(rec)
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
