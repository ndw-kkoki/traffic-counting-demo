"""動画に対してYOLO+ByteTrackでトラッキングし、指定ラインの通過台数を
クラス別にカウントする。注釈付き動画とカウント結果(JSON)を出力する。

使い方:
    PYTHONPATH=src python -m trafficdet.count_video \
        --source data/sample_traffic.mp4 --line 0,360,1280,360
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from trafficdet.config import BEST_MODEL, MODELS_DIR
from trafficdet.counter import LineCounter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, required=True)
    p.add_argument("--model", type=str, default=str(BEST_MODEL))
    p.add_argument("--line", type=str, default=None,
                    help="x1,y1,x2,y2。省略時は画面中央の水平線")
    p.add_argument("--out", type=str, default=str(MODELS_DIR / "counted_output.mp4"))
    p.add_argument("--conf", type=float, default=0.4)
    return p.parse_args()


def run(source: str, model_path: str, line: tuple[float, float, float, float] | None,
        out_path: str, conf: float = 0.4) -> dict:
    model = YOLO(model_path)

    cap = cv2.VideoCapture(source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()

    if line is None:
        line = (0, height // 2, width, height // 2)

    counter = LineCounter(line=line)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    results_stream = model.track(source=source, conf=conf, persist=True, stream=True, verbose=False)
    for result in results_stream:
        frame = result.plot()
        cv2.line(frame, (int(line[0]), int(line[1])), (int(line[2]), int(line[3])), (0, 0, 255), 2)

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            names = result.names
            for box, track_id, cls in zip(boxes.xywh, boxes.id, boxes.cls):
                cx, cy = float(box[0]), float(box[1])
                class_name = names[int(cls)]
                counter.update(int(track_id), class_name, (cx, cy))

        y = 30
        cv2.putText(frame, f"total: {counter.total}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        for cls_name, n in sorted(counter.counts.items()):
            y += 26
            cv2.putText(frame, f"{cls_name}: {n}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(frame)

    writer.release()

    report = {"total": counter.total, "counts": dict(counter.counts)}
    report_path = Path(out_path).with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    args = parse_args()
    line = tuple(float(v) for v in args.line.split(",")) if args.line else None
    run(args.source, args.model, line, args.out, args.conf)


if __name__ == "__main__":
    main()
