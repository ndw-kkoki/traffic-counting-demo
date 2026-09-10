"""1枚の画像に対して推論する薄いラッパー。"""
from __future__ import annotations

from ultralytics import YOLO

from trafficdet.config import BEST_MODEL


class Detector:
    def __init__(self, model_path=BEST_MODEL):
        self.model = YOLO(str(model_path))

    def predict(self, image_path: str, conf: float = 0.4):
        result = self.model.predict(image_path, conf=conf, verbose=False)[0]
        names = result.names
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class": names[int(box.cls[0])],
                "score": round(float(box.conf[0]), 4),
                "box": [round(v, 1) for v in (x1, y1, x2, y2)],
            })
        return detections

    def predict_and_draw(self, image_path: str, conf: float = 0.4):
        result = self.model.predict(image_path, conf=conf, verbose=False)[0]
        return result.plot()  # BGR numpy array
