"""Kaggle Kernel(script)上で traffic-counting-demo を
clone -> 依存関係インストール -> 学習 -> 評価 -> デモ画像生成 まで一気通貫で実行する。

データセットはkernel-metadata.jsonのdataset_sourcesで宣言済みのため、
/kaggle/input/road-vehicle-images-dataset/ に読み取り専用でマウントされる
(ダウンロード不要)。

出力(models/配下)は /kaggle/working/ 以下に残るため、
ローカルからは `kaggle kernels output` で回収する。
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/ndw-kkoki/traffic-counting-demo.git"
REPO_DIR = Path("/kaggle/working/traffic-counting-demo")
KAGGLE_INPUT = Path("/kaggle/input/road-vehicle-images-dataset/trafic_data")


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main() -> None:
    if REPO_DIR.exists():
        run(["git", "pull"], cwd=REPO_DIR)
    else:
        run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])

    # KaggleのベースイメージのPyTorchは新しいCUDAツールキットでビルドされており、
    # 割り当てGPUがP100(Pascal, sm_60)だと動かないことがある(satellite-object-detection-demoで
    # 確認済み)。ローカルで動作確認済みのcu121ビルドを明示的に入れ直す。
    run([sys.executable, "-m", "pip", "install", "-q",
         "torch==2.5.1", "torchvision==0.20.1",
         "--index-url", "https://download.pytorch.org/whl/cu121"])
    run([sys.executable, "-m", "pip", "install", "-q",
         "ultralytics", "lap", "pyyaml", "opencv-python-headless"])

    data_dir = REPO_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "data.yaml").write_text(
        "train: {0}/train/images\n"
        "val: {0}/valid/images\n"
        "\n"
        "nc: 21\n"
        "names: ['ambulance', 'army vehicle', 'auto rickshaw', 'bicycle', 'bus', 'car', "
        "'garbagevan', 'human hauler', 'minibus', 'minivan', 'motorbike', 'pickup', "
        "'policecar', 'rickshaw', 'scooter', 'suv', 'taxi', 'three wheelers -CNG-', "
        "'truck', 'van', 'wheelbarrow']\n".format(KAGGLE_INPUT),
        encoding="utf-8",
    )

    env = dict(os.environ, PYTHONPATH="src")
    run([
        sys.executable, "-m", "trafficdet.train",
        "--epochs", "80", "--batch", "32", "--imgsz", "640", "--patience", "15",
    ], cwd=REPO_DIR, env=env)

    run([sys.executable, "-m", "trafficdet.evaluate"], cwd=REPO_DIR, env=env)

    # デモ画像生成にはvalid画像を参照するため、KaggleマウントのパスをDATA_DIR配下に見せる
    (data_dir / "valid").symlink_to(KAGGLE_INPUT / "valid", target_is_directory=True)
    run([sys.executable, "-m", "trafficdet.demo"], cwd=REPO_DIR, env=env)

    print("=== done. outputs under:", REPO_DIR / "models", "===")


if __name__ == "__main__":
    main()
