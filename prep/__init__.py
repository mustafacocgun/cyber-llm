"""Data-prep package for the cyber-LLM fine-tune scaffold.

Pure standard-library + PyYAML only, so the whole build/test loop runs
offline on a weak machine. The heavy libraries (`datasets`, `huggingface_hub`)
are imported only inside the dataset-loading path, which runs on Kaggle.
"""
from pathlib import Path

import yaml


def load_config(path="mix_config.yaml"):
    """Load the dataset-mix recipe (YAML) into a plain dict."""
    with open(Path(path)) as f:
        return yaml.safe_load(f)


__all__ = ["load_config"]
