"""データセット・モデルパスの共通定義。"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
DATA_YAML = DATA_DIR / "data.yaml"
BEST_MODEL = MODELS_DIR / "best.pt"


def load_class_names() -> list[str]:
    with DATA_YAML.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["names"]
