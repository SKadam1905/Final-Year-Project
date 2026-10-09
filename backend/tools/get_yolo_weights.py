"""Download COCO-pretrained YOLOv8 weights (from the ultralytics GitHub releases) for person detection.

    cd backend
    ..\\venv\\Scripts\\python.exe -m tools.get_yolo_weights yolov8n yolov8s   # -> ml/weights/yolo/<name>.pt

The files are git-ignored like every other weights file.
"""

from __future__ import annotations

import argparse

from app.config import WEIGHTS_DIR

YOLO_DIR = WEIGHTS_DIR / "yolo"
SIZES = ("yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x")


def main(argv=None) -> None:
    from ultralytics.utils.downloads import attempt_download_asset

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", choices=SIZES)
    args = p.parse_args(argv)
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        path = YOLO_DIR / f"{name}.pt"
        attempt_download_asset(path)
        print(f"{name}: {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
