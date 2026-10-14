"""Sample each source to its configured size, split train vs held-out eval,
and decontaminate train against eval.

Decontamination removes any training record that shares a strong id (CVE/commit)
OR an exact text hash with an eval record — so the "is it good?" measurement
can't be inflated by leakage. Deterministic for a fixed seed.
"""
import random

from prep.dedup import dedup, strong_keys, text_hash


def _sample(records, n, rng):
    if n >= len(records):
        return list(records)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    return [records[i] for i in sorted(idx[:n])]


def build_splits(config, pools, seed):
    rng = random.Random(seed)
    holdout = set(config["eval"]["holdout_sources"])
    train, eval_ = [], []
    for src in config["sources"]:
        picked = _sample(pools.get(src["name"], []), src["sample"], rng)
        (eval_ if src["name"] in holdout else train).extend(picked)

    eval_ = dedup(eval_)
    train = dedup(train)

    ekeys = set().union(*(strong_keys(r) for r in eval_)) if eval_ else set()
    etext = {text_hash(r) for r in eval_}
    train = [r for r in train
             if not (strong_keys(r) & ekeys) and text_hash(r) not in etext]
    return train, eval_
