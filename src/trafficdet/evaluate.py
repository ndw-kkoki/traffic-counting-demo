"""学習済みモデルをval分割で評価し、mAP・クラス別APを表示する。

使い方:
    PYTHONPATH=src python -m trafficdet.evaluate
"""
from __future__ import annotations

from ultralytics import YOLO

from trafficdet.config import BEST_MODEL, DATA_YAML


def evaluate(model_path=BEST_MODEL):
    model = YOLO(str(model_path))
    metrics = model.val(data=str(DATA_YAML))

    print(f"mAP50-95(B): {metrics.box.map:.4f}")
    print(f"mAP50(B)   : {metrics.box.map50:.4f}")
    print(f"mAP75(B)   : {metrics.box.map75:.4f}")
    return metrics


if __name__ == "__main__":
    evaluate()
