"""YOLOの学習スクリプト(Ultralytics)。

使い方:
    PYTHONPATH=src python -m trafficdet.train --epochs 50 --model yolo11n.pt
"""
from __future__ import annotations

import argparse
import shutil

from ultralytics import YOLO

from trafficdet.config import DATA_YAML, MODELS_DIR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--model", type=str, default="yolo11n.pt", help="COCO事前学習済みのベースモデル")
    p.add_argument("--patience", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)

    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=str(MODELS_DIR),
        name="run",
        exist_ok=True,
    )

    best = MODELS_DIR / "run" / "weights" / "best.pt"
    MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy(best, MODELS_DIR / "best.pt")
    print(f"best model copied to {MODELS_DIR / 'best.pt'}")
    print(results)


if __name__ == "__main__":
    main()
