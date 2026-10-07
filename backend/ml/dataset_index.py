"""Index of a processed dataset: data/processed/<name>/index.csv, one row per image.

Layout written by ml.prepare_dataset:
    images/<id>.jpg    resized so both sides are multiples of 8
    density/<id>.npy   density map at 1/8 resolution (float32); sums to the head count
    index.csv          id, split (train/val/test), paths, count, source video, size
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

INDEX_NAME = "index.csv"
FIELDS = ["id", "split", "image", "density", "count", "video", "width", "height"]


@dataclass(frozen=True)
class Sample:
    id: str
    split: str
    image: Path
    density: Path
    count: float  # number of annotated heads
    video: str  # source video ("" for image datasets)
    width: int
    height: int


def write_index(root: str | Path, samples: Iterable[Sample]) -> Path:
    root = Path(root)
    path = root / INDEX_NAME
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for s in samples:
            writer.writerow({
                "id": s.id, "split": s.split,
                "image": s.image.relative_to(root).as_posix(),
                "density": s.density.relative_to(root).as_posix(),
                "count": f"{s.count:g}", "video": s.video, "width": s.width, "height": s.height,
            })
    return path


def read_index(root: str | Path, splits: Sequence[str] | None = None) -> list[Sample]:
    root = Path(root)
    path = root / INDEX_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m ml.prepare_dataset` first")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    samples = [
        Sample(r["id"], r["split"], root / r["image"], root / r["density"], float(r["count"]),
               r["video"], int(r["width"]), int(r["height"]))
        for r in rows
    ]
    return [s for s in samples if splits is None or s.split in splits]


def choose_val_ids(ids: Iterable[str], fraction: float, seed: int = 42) -> set[str]:
    """Random validation subset of the training images (reproducible for a seed)."""
    ids = sorted(ids)
    if fraction <= 0 or len(ids) < 2:
        return set()
    n = min(len(ids) - 1, max(1, round(len(ids) * fraction)))
    return set(random.Random(seed).sample(ids, n))


def split_videos(
    videos: Iterable[str],
    test_videos: Sequence[str] = (),
    val_videos: Sequence[str] = (),
    test_fraction: float = 0.0,
    val_fraction: float = 0.0,
    seed: int = 42,
) -> dict[str, str]:
    """Assign whole videos to train/val/test.

    Frames from one video are near-duplicates, so they must never be split across train
    and test, or the test score is inflated (plan §5.3).
    """
    videos = sorted(set(videos))
    unknown = (set(test_videos) | set(val_videos)) - set(videos)
    if unknown:
        raise ValueError(f"unknown videos: {sorted(unknown)} (available: {videos})")
    if set(test_videos) & set(val_videos):
        raise ValueError("a video cannot be in both test and val")

    assignment = {v: "test" for v in test_videos} | {v: "val" for v in val_videos}
    rest = [v for v in videos if v not in assignment]
    random.Random(seed).shuffle(rest)
    for split, explicit, fraction in (("test", test_videos, test_fraction), ("val", val_videos, val_fraction)):
        if not explicit and fraction > 0:
            for v in rest[: max(1, round(len(videos) * fraction))]:
                assignment[v] = split
            rest = [v for v in rest if v not in assignment]
    if not rest:
        raise ValueError("no videos left for training; add videos or lower the test/val share")
    assignment |= {v: "train" for v in rest}
    return assignment
