"""Index of a processed dataset: data/processed/<name>/index.csv, one row per image.

Layout written by ml.prepare_dataset:
    images/<id>.jpg    resized so both sides are multiples of 8
    density/<id>.npy   density map at 1/8 resolution (float32); sums to the head count
    index.csv          id, split (train/val/test), paths, count, source video, size
"""

from __future__ import annotations

import csv
import itertools
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


def split_by_location(
    videos: dict[str, tuple[str, int]],
    test_share: float = 0.15,
    val_share: float = 0.15,
    min_test_locations: int = 2,
    seed: int = 42,
) -> dict[str, str]:
    """Assign whole videos to train/val/test, holding out whole locations for test.

    `videos` maps video id -> (location, number of images). Test takes at least
    `min_test_locations` complete locations (none of their videos is trained on), chosen so the
    test share of images is as close as possible to `test_share`; validation takes whole videos
    from the other locations, close to `val_share`; the rest is training.
    """
    locations: dict[str, int] = {}
    for location, n in videos.values():
        locations[location] = locations.get(location, 0) + n
    total = sum(locations.values())
    if len(locations) < min_test_locations + 1:
        raise ValueError(f"{len(locations)} locations: holding out {min_test_locations} for test needs at least "
                         f"{min_test_locations + 1} (or name the test videos yourself)")
    rng = random.Random(seed)
    names = sorted(locations)
    best: list[tuple[float, float, tuple[str, ...]]] = []
    for k in range(min_test_locations, min(len(names) - 1, min_test_locations + 4) + 1):
        for combo in itertools.combinations(names, k):
            best.append((abs(sum(locations[c] for c in combo) / total - test_share), rng.random(), combo))
    test_locations = set(min(best)[2])

    assignment = {v: "test" for v, (location, _) in videos.items() if location in test_locations}
    rest = sorted(v for v in videos if v not in assignment)
    rng.shuffle(rest)
    if len(rest) < 2:
        raise ValueError("the locations left after the test hold-out have fewer than 2 videos: "
                         "validation and training need at least one each")
    val, val_n = [], 0
    for v in rest:  # add a video when it brings the validation share closer to the target
        n = videos[v][1]
        if len(val) < len(rest) - 1 and abs((val_n + n) / total - val_share) < abs(val_n / total - val_share):
            val.append(v)
            val_n += n
    if not val:  # every video alone overshoots the target: take the smallest
        val = [min(rest, key=lambda v: (videos[v][1], v))]
    assignment |= {v: "val" for v in val}
    assignment |= {v: "train" for v in rest if v not in assignment}
    return assignment
