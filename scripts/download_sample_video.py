"""通過カウントのデモ用に、Pexels提供のフリー素材動画をdata/にダウンロードする。

動画: "Traffic Flow In The Highway" by Mike Bird (Pexels)
配布元: https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/
ライセンス: Pexels License(商用利用可・改変可・クレジット表記不要)
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

VIDEO_URL = "https://videos.pexels.com/video-files/2103099/2103099-sd_960_540_30fps.mp4"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEST = DATA_DIR / "sample_traffic.mp4"


def download() -> None:
    if DEST.exists() and DEST.stat().st_size > 0:
        print("already downloaded, skipping.")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {VIDEO_URL} -> {DEST}")
    request = urllib.request.Request(VIDEO_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as resp, DEST.open("wb") as f:
        f.write(resp.read())
    print("done.")


if __name__ == "__main__":
    download()
