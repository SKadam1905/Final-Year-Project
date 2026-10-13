"""Annotation work on the India crops: balanced batches for the annotators, and checks of what they return.

Batches (step 5): whole videos are dealt to the annotators largest first, each going to the
batch with the least work so far (work = share of all crops + share of all suggested heads),
so no video is split between two annotators.

Checks (step 6): every crop must come back tagged "done" (or "bad_crop" to drop it). A done
crop is flagged for a second look when:
  * its corrected count differs from the suggestion far more than is typical for its video
    (the ratio corrected/suggested is compared with the video's median ratio, because the
    model's bias, e.g. undercounting dense crowds, is shared by all crops of a video);
  * every suggestion was deleted, or every suggestion was kept exactly as it was;
  * it has points outside the image or two points on the same spot (a double click).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

RATIO_TOLERANCE = 3.0  # flag when corrected/suggested is more than 3x away from the video's typical ratio
DUPLICATE_PX = 2.0


def balanced_batches(videos: dict[str, tuple[int, float]], n: int) -> list[list[str]]:
    """`videos`: id -> (crops, suggested heads). Returns n lists of whole videos with similar work."""
    total_crops = sum(c for c, _ in videos.values()) or 1
    total_heads = sum(h for _, h in videos.values()) or 1
    work = {v: c / total_crops + h / total_heads for v, (c, h) in videos.items()}
    batches: list[list[str]] = [[] for _ in range(n)]
    loads = [0.0] * n
    for v in sorted(videos, key=lambda v: (-work[v], v)):
        k = min(range(n), key=lambda k: (loads[k], k))
        batches[k].append(v)
        loads[k] += work[v]
    return batches


@dataclass
class CropCheck:
    crop: str  # image file name
    video: str
    batch: str
    status: str  # done / bad_crop / not marked done / missing from exports
    suggested: int
    corrected: int | None = None
    flags: list[str] = field(default_factory=list)


def point_flags(points: np.ndarray, width: int, height: int, suggested: np.ndarray) -> list[str]:
    points = np.asarray(points, np.float64).reshape(-1, 2)
    suggested = np.asarray(suggested, np.float64).reshape(-1, 2)
    flags = []
    outside = int(((points[:, 0] < 0) | (points[:, 0] >= width) | (points[:, 1] < 0) | (points[:, 1] >= height)).sum())
    if outside:
        flags.append(f"{outside} points outside the image")
    if len(points) > 1:
        pairs = len(cKDTree(points).query_pairs(DUPLICATE_PX - 1e-9))
        if pairs:
            flags.append(f"{pairs} pairs of points less than {DUPLICATE_PX:g} px apart (double clicks?)")
    if len(suggested) >= 10 and len(points) == len(suggested):
        if np.allclose(np.sort(points, axis=0), np.sort(suggested, axis=0), atol=0.5):
            flags.append("all suggestions kept exactly as they were: was this crop really checked?")
    if len(points) == 0 and len(suggested) >= 5:
        flags.append(f"all {len(suggested)} suggestions deleted and no head added")
    return flags


def ratio_flags(checks: list[CropCheck]) -> None:
    """Flag done crops whose corrected/suggested ratio is far from their video's median ratio."""
    done = [c for c in checks if c.status == "done" and c.corrected is not None]
    log_ratio = {id(c): math.log((c.corrected + 1) / (c.suggested + 1)) for c in done}
    overall = float(np.median(list(log_ratio.values()))) if done else 0.0
    for c in done:
        same_video = [log_ratio[id(o)] for o in done if o.video == c.video]
        typical = float(np.median(same_video)) if len(same_video) >= 4 else overall
        if abs(log_ratio[id(c)] - typical) > math.log(RATIO_TOLERANCE):
            c.flags.append(f"count {c.suggested} -> {c.corrected}: far from the usual change for this video "
                           f"(x{math.exp(typical):.2f})")
