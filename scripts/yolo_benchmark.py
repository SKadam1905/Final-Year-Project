"""P1: YOLOv8 person-detection time per frame for each model size, next to CSRNet, on the demo clips.

    venv\\Scripts\\python.exe scripts\\yolo_benchmark.py            # yolov8n, s, m from backend/ml/weights/yolo

Reads every clip in data/india_dataset/raw_videos, takes `--frames` evenly spaced frames from each
and times (after a warm-up) YOLO detection (person class, imgsz 640, half precision on the GPU)
and CSRNet (the weights in .env) on each frame, one at a time as the live worker does. Also reports
how many people each model finds, so the speed can be weighed against what is detected.
Writes results/yolo_benchmark.csv / .md. A live cycle runs every zone, so its GPU time is about
(YOLO + CSRNet) x number of zones; the measured live cycle time is in /health (last_cycle_s).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.config import get_settings  # noqa: E402
from app.csrnet_model import DensityEstimator  # noqa: E402
from app.detector import PersonDetector  # noqa: E402
from app.worker import RAW_VIDEOS, VIDEO_EXTENSIONS  # noqa: E402
from report_utils import write_table  # noqa: E402

YOLO_DIR = ROOT / "backend" / "ml" / "weights" / "yolo"
CYCLE_BUDGET_S = 2.0


def sample_frames(path: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * total / n))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def time_calls(fn, frames: list[np.ndarray]) -> tuple[list[float], list[float]]:
    """(milliseconds per call, result count per call)."""
    times, counts = [], []
    for frame in frames:
        started = time.perf_counter()
        result = fn(frame)
        times.append((time.perf_counter() - started) * 1e3)
        counts.append(float(result.count))
    return times, counts


def p95(values: list[float]) -> float:
    return float(np.percentile(values, 95))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["yolov8n", "yolov8s", "yolov8m"])
    p.add_argument("--frames", type=int, default=40, help="frames per clip")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "yolo_benchmark")
    args = p.parse_args()

    clips = sorted(c for c in RAW_VIDEOS.iterdir() if c.suffix.lower() in VIDEO_EXTENSIONS)
    if not clips:
        raise SystemExit(f"no clips in {RAW_VIDEOS}")
    frames = {c.stem: sample_frames(c, args.frames) for c in clips}
    every = [f for fs in frames.values() for f in fs]
    zones = len(clips)
    s = get_settings()

    csrnet = DensityEstimator(s.csrnet_weights, s.device, s.infer_max_width, allow_pickle=s.csrnet_allow_pickle)
    for f in every[:5]:
        csrnet.predict(f)  # warm-up
    csr_ms, _ = time_calls(csrnet.predict, every)
    csr_counts = {name: statistics.mean(time_calls(csrnet.predict, fs)[1]) for name, fs in frames.items()}
    csr_median = statistics.median(csr_ms)

    rows = [{"model": f"CSRNet ({s.csrnet_weights.name})", "frames": len(every),
             "median_ms": round(csr_median, 1), "p95_ms": round(p95(csr_ms), 1),
             **{f"mean_count_{k}": round(v, 1) for k, v in csr_counts.items()},
             f"est_cycle_s_{zones}_zones": "", "fits_2s": ""}]
    for name in args.models:
        detector = PersonDetector(YOLO_DIR / f"{name}.pt", s.device, args.imgsz)
        for f in every[:5]:
            detector.detect(f)  # warm-up
        ms, _ = time_calls(detector.detect, every)
        counts = {clip: statistics.mean(time_calls(detector.detect, fs)[1]) for clip, fs in frames.items()}
        cycle = zones * (statistics.median(ms) + csr_median) / 1e3
        rows.append({"model": f"{name} (imgsz {args.imgsz})", "frames": len(every),
                     "median_ms": round(statistics.median(ms), 1), "p95_ms": round(p95(ms), 1),
                     **{f"mean_count_{k}": round(v, 1) for k, v in counts.items()},
                     f"est_cycle_s_{zones}_zones": round(cycle, 2), "fits_2s": cycle < CYCLE_BUDGET_S})
        del detector
        print(rows[-1], flush=True)

    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    write_table(args.out, rows, "YOLOv8 person detection vs CSRNet: time per frame on the demo clips", notes=[
        f"Device: {gpu}. {args.frames} evenly spaced frames from each of {zones} clips in "
        f"data/india_dataset/raw_videos ({', '.join(frames)}), 640x360. One frame at a time, after 5 warm-up calls.",
        "YOLO: COCO-pretrained, person class only, conf 0.25, half precision on the GPU. "
        f"CSRNet: INFER_MAX_WIDTH={s.infer_max_width}, AMP.",
        f"est_cycle_s = zones x (YOLO median + CSRNet median): GPU time of one live cycle, before optical flow "
        "(which runs on the CPU, in parallel across zones). The measured cycle time is /health last_cycle_s.",
        "mean_count_* = mean number of people found per frame in that clip (no ground truth for these clips).",
        "Command: venv\\Scripts\\python.exe scripts\\yolo_benchmark.py",
    ])
    print(f"wrote {args.out}.csv / .md")


if __name__ == "__main__":
    main()
