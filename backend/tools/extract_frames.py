"""Extract frames from videos at a fixed rate (OpenCV, so FFmpeg is not required).

    python -m tools.extract_frames ../data/india_dataset/raw_videos/pandharpur_01.mp4 --fps 0.2
    python -m tools.extract_frames ../data/india_dataset/raw_videos --fps 0.2    # every video in a folder

Frames go to data/india_dataset/frames/<video name>/frame_0001.jpg ..., and frames.csv
records each frame's time in the video. For annotation, sample sparsely (one frame every
5-10 s, i.e. --fps 0.1-0.2): frames 1 s apart are near-duplicates.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from app.config import DATA_DIR
from app.image_io import write_image

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}


def extract_frames(
    video: Path,
    out_dir: Path,
    fps: float = 1.0,
    start: float = 0.0,
    end: float | None = None,
    max_frames: int | None = None,
    quality: int = 95,
) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not 0 < src_fps <= 240:
        src_fps = 25.0
    out_dir.mkdir(parents=True, exist_ok=True)
    next_t, index, rows = start, 0, []
    try:
        while cap.grab():
            t = index / src_fps
            index += 1
            if end is not None and t > end:
                break
            if t + 1e-9 < next_t:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            name = f"frame_{len(rows) + 1:04d}.jpg"
            write_image(out_dir / name, frame, quality)
            rows.append((name, f"{t:.2f}"))
            next_t += 1.0 / fps
            if max_frames and len(rows) >= max_frames:
                break
    finally:
        cap.release()
    with open(out_dir / "frames.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s"])
        writer.writerows(rows)
    return len(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", type=Path, nargs="+", help="video files or folders of videos")
    p.add_argument("--fps", type=float, default=1.0, help="frames per second to keep (0.2 = one every 5 s)")
    p.add_argument("--out", type=Path, default=DATA_DIR / "india_dataset" / "frames")
    p.add_argument("--start", type=float, default=0.0, help="seconds to skip at the start")
    p.add_argument("--end", type=float, help="stop at this time (seconds)")
    p.add_argument("--max-frames", type=int)
    args = p.parse_args(argv)

    videos = []
    for path in args.inputs:
        videos += sorted(v for v in path.iterdir() if v.suffix.lower() in VIDEO_EXTS) if path.is_dir() else [path]
    if not videos:
        raise SystemExit("no videos found")
    for video in videos:
        n = extract_frames(video, args.out / video.stem, args.fps, args.start, args.end, args.max_frames)
        print(f"{video.name}: {n} frames -> {args.out / video.stem}")


if __name__ == "__main__":
    main()
