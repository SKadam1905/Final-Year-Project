"""Extract the fixed 15-frame Kumbh counting-evaluation set (see results/counting_accuracy.md).

    python -m tools.make_kumbh_eval            # -> data/eval/frames/kumbh_eval/frame_01.png ... + frames.csv

Frames come from the two distinct Kumbh clips in data/uploads/videos (the other uploads and
the data/india_dataset/raw_videos clips are copies or cuts of them), at full source
resolution as lossless PNG. Each time was picked inside a shot (cuts found with an
HS-histogram detector) away from captions, interviews, end cards and transitions; only the
channel watermark in the top-right corner remains. Shot types: aerial (drone, top-down),
wide (crowd to the horizon), medium (people at mid-distance), close (a few large people).

The set lives outside data/india_dataset on purpose, so prepare_dataset never trains on it.
Ground truth, either way (see ml.kumbh_eval):
    python -m tools.count_eval          # count heads per cell of a 3x3 grid (counts.csv)
    python -m tools.annotate_points ../data/eval/frames/kumbh_eval --out ../data/eval/annotations/kumbh_eval

At 640x360 not every part of every frame can be read (checked by eye with contrast enhancement
and zoom, 2026-10-12): COUNT_REGIONS lists the part of a frame to count where the rest is too
small, dense or hazy to make out heads; frames not listed are counted whole.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from app.config import DATA_DIR
from app.image_io import write_image

SOURCES = {  # name -> upload file name prefix (sha-based id from the upload API)
    "news9_aerial": "4f63c49ac47c__",
    "pti_prayagraj": "1a939a9e0743__",
}
RAW_CUTS = {  # data/india_dataset/raw_videos clips cut from the sources: (source, start_s, end_s)
    "kumbh_aerial_news9": ("news9_aerial", 0.0, 120.0),
    "prayagraj_ghat_pti": ("pti_prayagraj", 0.0, 24.0),
    "prayagraj_sangam_pti": ("pti_prayagraj", 99.9, 109.7),
}
PICKS = [  # (source, time_s, enclosing shot "start-end" in s, shot type)
    ("news9_aerial", 7.1, "0.0-14.2", "aerial"),
    ("news9_aerial", 26.5, "14.2-63.2", "aerial"),
    ("news9_aerial", 51.0, "14.2-63.2", "aerial"),
    ("news9_aerial", 77.5, "63.2-120.2", "aerial"),
    ("pti_prayagraj", 16.4, "14.3-18.4", "wide"),
    ("pti_prayagraj", 90.5, "89.4-91.6", "wide"),
    ("pti_prayagraj", 104.8, "99.8-109.8", "wide"),
    ("pti_prayagraj", 21.2, "18.4-24.0", "medium"),
    ("pti_prayagraj", 41.2, "39.0-43.4", "medium"),
    ("pti_prayagraj", 68.2, "66.1-70.4", "medium"),
    ("pti_prayagraj", 96.0, "94.5-97.5", "medium"),
    ("pti_prayagraj", 57.6, "55.5-59.7", "close"),
    ("pti_prayagraj", 60.9, "59.7-62.1", "close"),
    ("pti_prayagraj", 64.1, "62.1-66.1", "close"),
    ("pti_prayagraj", 93.0, "91.6-94.5", "close"),
]

COUNT_REGIONS = {  # frame -> (x0, y0, x1, y1) in source pixels: the readable part to count
    "frame_01.png": (100, 0, 205, 360),  # aerial: sand strip between the pontoon and the dense crowd
    "frame_02.png": (0, 0, 250, 85),  # aerial: sand at the top left, clear of the crowd's edge
    "frame_04.png": (300, 120, 410, 360),  # aerial: bathers in the water between the crowd and the boats
    "frame_05.png": (0, 230, 640, 360),  # wide: near and middle crowd; the band at the horizon is unreadable
    "frame_06.png": (0, 200, 640, 360),  # wide: foreground and bathers; the far boats and bank excluded
    "frame_07.png": (0, 230, 640, 360),  # wide: lower band, as frame 05
    "frame_08.png": (0, 185, 640, 360),  # medium: near crowd; the hazy background excluded
    "frame_09.png": (280, 140, 640, 360),  # medium, night: the standing crowd right of the ambulance
}
# frame_03.png (dense aerial crowd in haze) is unreadable throughout: mark it so (s) rather than count it.


def find_source(videos_dir: Path, prefix: str) -> Path:
    matches = sorted(videos_dir.glob(f"{prefix}*.mp4"))
    if not matches:
        raise SystemExit(f"source video {prefix}*.mp4 not found in {videos_dir}")
    return matches[0]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", type=Path, default=DATA_DIR / "uploads" / "videos")
    p.add_argument("--out", type=Path, default=DATA_DIR / "eval" / "frames" / "kumbh_eval")
    p.add_argument("--force", action="store_true", help="overwrite existing frames (annotations refer to them)")
    args = p.parse_args(argv)
    if any(args.out.glob("*.png")) and not args.force:
        raise SystemExit(f"{args.out} already has frames; not overwriting (use --force)")
    sources = {name: find_source(args.videos, prefix) for name, prefix in SOURCES.items()}
    rows = []
    for i, (source, t, shot, kind) in enumerate(PICKS, start=1):
        cap = cv2.VideoCapture(str(sources[source]))
        index = round(t * cap.get(cv2.CAP_PROP_FPS))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"could not read frame {index} of {sources[source]}")
        name = f"frame_{i:02d}.png"
        write_image(args.out / name, frame)
        raw = next((clip for clip, (src, a, b) in RAW_CUTS.items() if src == source and a <= t < b), "")
        rows.append({"frame": name, "shot_type": kind, "source": source, "source_file": sources[source].name,
                     "time_s": t, "frame_index": index, "shot_span_s": shot, "raw_videos_clip": raw,
                     "width": frame.shape[1], "height": frame.shape[0]})
        print(f"{name}: {kind:<6} {source} t={t}s (frame {index}) {frame.shape[1]}x{frame.shape[0]}")
    with open(args.out / "frames.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} frames + frames.csv -> {args.out}")


if __name__ == "__main__":
    main()
