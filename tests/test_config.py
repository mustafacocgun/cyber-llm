from pathlib import Path
from prep import load_config

CFG_PATH = Path(__file__).resolve().parent.parent / "mix_config.yaml"


def test_mix_config_sums_to_target():
    cfg = load_config(CFG_PATH)
    holdout = set(cfg["eval"]["holdout_sources"])
    train_total = sum(s["sample"] for s in cfg["sources"] if s["name"] not in holdout)
    assert abs(train_total - cfg["train_target"]) <= 2000, train_total


def test_source_names_unique():
    cfg = load_config(CFG_PATH)
    names = [s["name"] for s in cfg["sources"]]
    assert len(names) == len(set(names))


def test_every_source_has_license():
    cfg = load_config(CFG_PATH)
    assert all(s.get("license") for s in cfg["sources"])
