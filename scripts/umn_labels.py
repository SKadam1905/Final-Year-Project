"""Phase 6B, step 1: data/umn/labels.csv from the UMN video's own ground truth.

    venv\\Scripts\\python.exe scripts\\umn_labels.py

The UMN Unusual Crowd Activity video (Crowd-Activity-All.avi from https://mha.cs.umn.edu/Movies/,
linked from http://mha.cs.umn.edu/proj_events.shtml; 11 clips, 3 scenes, 7739 frames, 30 fps, 320x240)
carries its ground truth burnt into the frames: from the moment the authors mark the crowd as
panicking, a red "Abnormal Crowd Activity" caption shows in the top-left corner until shortly
before the clip ends. This script reads that caption frame by frame (no judgement involved):

  panic_frame      first frame of each caption interval (the published normal -> panic transition)
  label_end_frame  last frame with the caption
  start/end_frame  clip boundaries: the clip after a caption interval starts at the largest
                   frame-to-frame change within 2 s after the caption ends (clips are joined by
                   hard cuts); the first clip starts at 0, the last ends at the last frame
  scene            lawn (green field), indoor (grey-scale foyer) or plaza, from the clip's colours

Columns: clip, scene, start_frame, end_frame, panic_frame, label_end_frame, source (frames 0-based).
It refuses to write unless it finds 11 caption intervals and 2 lawn, 6 indoor and 3 plaza clips, the
published layout of the dataset.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAPTION_BOX = np.s_[2:16, 2:130]  # where the red caption sits, in 320x240 pixels
SOURCE = "red 'Abnormal Crowd Activity' caption in Crowd-Activity-All.avi (mha.cs.umn.edu)"
EXPECTED_SCENES = {"lawn": 2, "indoor": 6, "plaza": 3}


def caption_present(frame: np.ndarray) -> bool:
    box = frame[CAPTION_BOX].astype(int)
    b, g, r = box[..., 0], box[..., 1], box[..., 2]
    return int(((r > 150) & (g < 90) & (b < 90)).sum()) > 30  # the caption's red text: ~38 pixels; 0 without


def intervals(flags: np.ndarray) -> list[tuple[int, int]]:
    """(first, last) index of every run of True."""
    edges = np.diff(np.concatenate([[0], flags.astype(int), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def clip_starts(diffs: np.ndarray, caption_ends: list[int], fps: float) -> list[int]:
    """Start frame of every clip: 0, then the biggest frame change within 2 s after each caption ends."""
    starts = [0]
    for end in caption_ends[:-1]:
        window = diffs[end + 1:min(len(diffs), end + 1 + round(2 * fps))]
        starts.append(end + 1 + int(np.argmax(window)))
    return starts


def scene_of(mean_bgr: np.ndarray, saturation: float) -> str:
    if saturation < 20:
        return "indoor"  # the foyer clips are grey-scale
    b, g, r = mean_bgr
    return "lawn" if g > r and g > b else "plaza"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, default=ROOT / "data" / "umn" / "Crowd-Activity-All.avi")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "umn" / "labels.csv")
    args = p.parse_args(argv)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    flags, diffs, colours, sats, prev = [], [], [], [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        flags.append(caption_present(frame))
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grey[:18, :135] = 0  # the caption itself is not a scene change
        diffs.append(0.0 if prev is None else float(np.abs(grey - prev).mean()))
        prev = grey
        colours.append(frame.reshape(-1, 3).mean(axis=0))
        sats.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 1].mean()))
    cap.release()
    captions = intervals(np.array(flags))
    if len(captions) != 11:
        raise SystemExit(f"found {len(captions)} caption intervals, expected 11: {captions}")
    starts = clip_starts(np.array(diffs), [end for _, end in captions], fps)
    ends = [s - 1 for s in starts[1:]] + [len(flags) - 1]
    rows = []
    for k, ((on, off), start, end) in enumerate(zip(captions, starts, ends), start=1):
        if not start < on <= off <= end:
            raise SystemExit(f"clip {k}: caption {on}-{off} outside the clip {start}-{end}")
        scene = scene_of(np.mean(colours[start:end + 1], axis=0), float(np.mean(sats[start:end + 1])))
        rows.append({"clip": k, "scene": scene, "start_frame": start, "end_frame": end, "panic_frame": int(on),
                     "label_end_frame": int(off), "source": SOURCE})
    found = {s: sum(r["scene"] == s for r in rows) for s in EXPECTED_SCENES}
    if found != EXPECTED_SCENES:
        raise SystemExit(f"scenes {found} differ from the published layout {EXPECTED_SCENES}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"clip {r['clip']:2d} {r['scene']:<6} frames {r['start_frame']:4d}-{r['end_frame']:4d}  panic from "
              f"{r['panic_frame']} ({(r['panic_frame'] - r['start_frame']) / fps:.1f} s in), caption until "
              f"{r['label_end_frame']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
