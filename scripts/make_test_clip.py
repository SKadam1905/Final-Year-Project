"""Generate a reproducible synthetic crowd clip for the optical-flow and stress tests.

    venv\\Scripts\\python.exe scripts\\make_test_clip.py        # -> data\\test_clips\\synthetic_crowd_1280.mp4

About 600 "heads" (shaded discs) move over a static textured ground through eight 8-second
regimes, so turbulence covers the whole 0..1 range:
    0 still (sub-pixel jitter)   1 orderly stream        2 two opposite lanes
    3 interleaved counter-flow   4 chaotic               5 half orderly, half chaotic
    6 converging on the centre   7 swirling around the centre
Speeds are 1-3 px/frame at 1280 px wide (walking pace for a distant camera). This is a test
signal for comparing settings, not a stand-in for real crowd footage.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

REGIMES = ["still", "orderly", "two_lanes", "counter_flow", "chaotic", "half_chaotic", "converging", "swirl"]


def velocities(regime: str, pos: np.ndarray, rng: np.random.Generator, chaos: np.ndarray, size) -> np.ndarray:
    w, h = size
    n = len(pos)
    centre = np.array([w / 2, h / 2])
    if regime == "still":
        return rng.normal(0, 0.15, (n, 2))
    if regime == "orderly":
        return np.tile([2.0, 0.0], (n, 1))
    if regime == "two_lanes":
        return np.where(pos[:, 1:2] < h / 2, [[2.0, 0.0]], [[-2.0, 0.0]])
    if regime == "counter_flow":
        band = (pos[:, 1] // 24).astype(int) % 2  # alternate direction every 24 px band
        return np.where(band[:, None] == 0, [[2.0, 0.0]], [[-2.0, 0.0]])
    if regime == "chaotic":
        return chaos
    if regime == "half_chaotic":
        return np.where(pos[:, 0:1] < w / 2, [[2.0, 0.0]], chaos)
    to_centre = centre - pos
    dist = np.linalg.norm(to_centre, axis=1, keepdims=True) + 1e-6
    if regime == "converging":
        return 2.0 * to_centre / dist
    tangent = np.column_stack([-to_centre[:, 1], to_centre[:, 0]]) / dist  # swirl
    return 2.0 * tangent


def main(argv=None) -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=root / "data" / "test_clips" / "synthetic_crowd_1280.mp4")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--regime-seconds", type=float, default=8.0)
    p.add_argument("--people", type=int, default=600)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    size = (args.width, args.height)
    noise = cv2.GaussianBlur(rng.uniform(0, 255, (args.height, args.width)).astype(np.float32), (0, 0), 3)
    ground = cv2.normalize(noise, None, 120, 190, cv2.NORM_MINMAX).astype(np.uint8)
    ground = cv2.cvtColor(ground, cv2.COLOR_GRAY2BGR)
    pos = rng.uniform((0, 0), size, (args.people, 2))
    radius = rng.integers(7, 12, args.people)
    shade = rng.integers(20, 90, args.people)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
    frames_per_regime = round(args.regime_seconds * args.fps)
    chaos = np.zeros((args.people, 2))
    timeline = []
    for index, regime in enumerate(REGIMES):
        timeline.append((regime, index * args.regime_seconds, (index + 1) * args.regime_seconds))
        for k in range(frames_per_regime):
            if k % (args.fps // 2) == 0:  # chaotic headings change every 0.5 s
                angle = rng.uniform(0, 2 * np.pi, args.people)
                speed = rng.uniform(1.0, 3.0, args.people)
                chaos = np.column_stack([np.cos(angle), np.sin(angle)]) * speed[:, None]
            pos = (pos + velocities(regime, pos, rng, chaos, size)) % size
            frame = ground.copy()
            for (x, y), r, s in zip(pos.astype(int), radius, shade):
                cv2.circle(frame, (int(x), int(y)), int(r), (int(s), int(s) + 10, int(s) + 20), -1, cv2.LINE_AA)
                cv2.circle(frame, (int(x) - 2, int(y) - 2), max(2, int(r) // 3), (int(s) + 90,) * 3, -1, cv2.LINE_AA)
            writer.write(frame)
    writer.release()

    with open(args.out.with_suffix(".regimes.csv"), "w", newline="") as f:
        csv.writer(f).writerows([("regime", "start_s", "end_s"), *timeline])
    print(f"{args.out}: {len(REGIMES) * frames_per_regime} frames, {size[0]}x{size[1]} @ {args.fps} fps")


if __name__ == "__main__":
    main()
