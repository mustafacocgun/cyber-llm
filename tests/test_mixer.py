from prep.mixer import build_splits
from prep.dedup import strong_keys
from prep.schema import message, record


def _rec(text, src, ids=None):
    return record([message("user", text), message("assistant", "a")],
                  source=src, bucket="b", license="l", ids=ids or {})


CFG = {
    "train_target": 100,
    "sources": [
        {"name": "A", "sample": 50},
        {"name": "B", "sample": 50},
        {"name": "evalset", "sample": 10},
    ],
    "eval": {"holdout_sources": ["evalset"], "size": 10},
}


def _pools():
    return {
        "A": [_rec(f"a{i}", "A") for i in range(80)],
        "B": [_rec(f"b{i}", "B", ids={"cve": f"CVE-{i}"}) for i in range(80)],
        "evalset": [_rec(f"e{i}", "evalset", ids={"cve": f"CVE-{i}"}) for i in range(20)],
    }


def test_build_splits_counts_and_determinism():
    train, ev = build_splits(CFG, _pools(), seed=42)
    assert len(ev) == 10
    assert 90 <= len(train) <= 100           # B rows sharing an eval CVE are removed
    train2, _ = build_splits(CFG, _pools(), seed=42)
    assert [r["messages"] for r in train] == [r["messages"] for r in train2]


def test_eval_disjoint_from_train_by_strong_key():
    train, ev = build_splits(CFG, _pools(), seed=42)
    tk = set().union(*(strong_keys(r) for r in train)) if train else set()
    ek = set().union(*(strong_keys(r) for r in ev)) if ev else set()
    assert tk.isdisjoint(ek)


def test_eval_disjoint_from_train_by_text():
    pools = {"A": [_rec("dup", "A")], "B": [], "evalset": [_rec("dup", "evalset")]}
    cfg = {
        "train_target": 1,
        "sources": [{"name": "A", "sample": 1}, {"name": "B", "sample": 0},
                    {"name": "evalset", "sample": 1}],
        "eval": {"holdout_sources": ["evalset"], "size": 1},
    }
    train, ev = build_splits(cfg, pools, seed=1)
    assert len(ev) == 1 and len(train) == 0
