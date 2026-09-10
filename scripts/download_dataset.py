"""Road Vehicle Images Dataset(Kaggle)をdata/にダウンロードし、
Ultralytics YOLOがそのまま読める配置(train/valid images+labels, data.yaml)に整える。

データセット: Road Vehicle Images Dataset
配布元: https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset
ライセンス: DbCL-1.0 (Database Contents License)

事前に `kaggle` CLI が認証済みであること(~/.kaggle/access_token 等)。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET_REF = "ashfakyeafi/road-vehicle-images-dataset"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def download() -> None:
    if (DATA_DIR / "data.yaml").exists():
        print("already downloaded, skipping.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "road-vehicle-images-dataset.zip"

    subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download", DATASET_REF,
         "-p", str(DATA_DIR)],
        check=True,
    )

    print("extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    zip_path.unlink()

    src_root = DATA_DIR / "trafic_data"
    for name in ("train", "valid"):
        dest = DATA_DIR / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src_root / name), str(dest))

    # Ultralytics用にパスを絶対パス相当(data.yaml基準の相対)へ書き換え
    yaml_text = (src_root / "data_1.yaml").read_text(encoding="utf-8")
    yaml_text = yaml_text.replace("../train/images", "train/images")
    yaml_text = yaml_text.replace("../valid/images", "valid/images")
    (DATA_DIR / "data.yaml").write_text(yaml_text, encoding="utf-8")

    shutil.rmtree(src_root)
    print("done.")


if __name__ == "__main__":
    download()
