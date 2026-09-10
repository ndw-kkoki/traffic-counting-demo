"""valid分割からランダムに数枚選び、検出結果を可視化してmodels/demo_outputs/に保存する。

使い方:
    PYTHONPATH=src python -m trafficdet.demo
"""
from __future__ import annotations

import random

import cv2

from trafficdet.config import DATA_DIR, MODELS_DIR
from trafficdet.infer import Detector

OUT_DIR = MODELS_DIR / "demo_outputs"


def main(n: int = 8, conf: float = 0.4):
    images_dir = DATA_DIR / "valid" / "images"
    all_images = sorted(images_dir.glob("*.jpg"))
    sample = random.sample(all_images, min(n, len(all_images)))

    detector = Detector()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for image_path in sample:
        drawn = detector.predict_and_draw(str(image_path), conf=conf)
        out_path = OUT_DIR / f"{image_path.stem}_detected.jpg"
        cv2.imwrite(str(out_path), drawn)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
