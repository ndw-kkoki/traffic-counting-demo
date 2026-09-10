"""推論API。画像をアップロードすると検出結果(JSON)を返す。

起動:
    PYTHONPATH=src uvicorn trafficdet.api:app --reload
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from PIL import Image

from trafficdet.config import load_class_names
from trafficdet.infer import Detector

app = FastAPI(title="Traffic Vehicle Detection API")
_detector: Detector | None = None


def get_detector() -> Detector:
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector


@app.get("/")
def root():
    return {"status": "ok", "classes": load_class_names()}


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.4):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    try:
        detections = get_detector().predict(tmp_path, conf=conf)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"count": len(detections), "detections": detections}
