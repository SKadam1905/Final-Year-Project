"""Ground truth and count error for the Kumbh evaluation frames (data/eval, tools.make_kumbh_eval).

A frame's ground truth is a head count inside a region (the whole frame when none). It comes from
one of two human annotations in the annotations folder:
  * counts_<annotator>.csv (tools.count_eval --annotator <name>): heads counted per cell of a 3x3
    grid over the region, one file per person. Only rows with all nine cells entered are used; the
    frame's count is the mean of every annotator's total (all must have counted the same region).
    No head positions, so with this ground truth only the count error is measured: point-level
    localisation is not evaluated.
  * <frame>.json (tools.annotate_points): clicked head points; the count is the points inside the
    optional ROI.
When a frame has both, the counts win and the point file is reported as ignored. Frames marked
unreadable in frames.csv are left out. Models are scored on the same region: density summed over
the map cells whose centres lie in it, detections counted by their head position.

Error measures: per frame the absolute error and the % error (|pred - true| / true); per shot type
the MAE and the relative error sum |pred - true| / sum true, which, unlike the mean of per-frame
percentages, is not dominated by frames with only a few people.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from ml.annotations import UNREADABLE, in_roi, load_points_json, read_frame_status

GRID = 3  # cells per side
CELLS = [f"cell_{k}" for k in range(1, GRID * GRID + 1)]  # row by row, cell_1 = top left
COUNT_FIELDS = ["frame", "x0", "y0", "x1", "y1", "grid", *CELLS, "total", "complete", "updated"]
LEGACY_COUNTS = "counts.csv"  # before per-annotator files; renamed to counts_shubham.csv on 2026-10-12
ANNOTATOR = re.compile(r"^[a-z0-9_-]+$")

Region = tuple[float, float, float, float]  # x0, y0, x1, y1 in source pixels


@dataclass(frozen=True)
class GroundTruth:
    frame: str
    shot_type: str
    count: float
    region: Region
    source: str  # "counts" or "points"
    width: int
    height: int
    annotators: tuple[tuple[str, float], ...] = ()  # (name, total) behind a "counts" count


def cell_rects(region: Region, grid: int = GRID) -> list[Region]:
    """The grid's cells over a region, row by row (left to right, top to bottom)."""
    x0, y0, x1, y1 = region
    xs = [x0 + (x1 - x0) * i / grid for i in range(grid + 1)]
    ys = [y0 + (y1 - y0) * j / grid for j in range(grid + 1)]
    return [(xs[i], ys[j], xs[i + 1], ys[j + 1]) for j in range(grid) for i in range(grid)]


def counts_path(annotations_dir: str | Path, annotator: str) -> Path:
    """data/eval/annotations/kumbh_eval/counts_<annotator>.csv; names are lowercase letters, digits, _ and -."""
    if not ANNOTATOR.match(annotator):
        raise ValueError(f"annotator name {annotator!r}: use lowercase letters, digits, _ and - (e.g. rohit)")
    return Path(annotations_dir) / f"counts_{annotator}.csv"


def read_annotator_counts(annotations_dir: str | Path) -> dict[str, dict[str, dict]]:
    """annotator -> frame -> counts row, from every counts_<annotator>.csv in the folder."""
    annotations_dir = Path(annotations_dir)
    if (annotations_dir / LEGACY_COUNTS).exists():
        raise ValueError(f"{annotations_dir / LEGACY_COUNTS} has no annotator: rename it to counts_<name>.csv")
    return {path.stem[len("counts_"):]: read_counts(path) for path in sorted(annotations_dir.glob("counts_*.csv"))}


def read_counts(path: str | Path) -> dict[str, dict]:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["frame"]: r for r in csv.DictReader(f)}


def count_row(frame: str, region: Region, cells: list[int | None]) -> dict:
    """A counts.csv row: the region, each cell's count ("" if not entered), the total and whether complete."""
    entered = [c for c in cells if c is not None]
    return {"frame": frame, **{k: round(float(v), 1) for k, v in zip(("x0", "y0", "x1", "y1"), region)},
            "grid": f"{GRID}x{GRID}", **{name: "" if c is None else c for name, c in zip(CELLS, cells)},
            "total": sum(entered), "complete": "yes" if len(entered) == len(cells) else "no",
            "updated": datetime.now().isoformat(timespec="seconds")}


def write_counts(path: str | Path, rows: dict[str, dict]) -> None:
    """Rewrite counts.csv atomically, rows in frame order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COUNT_FIELDS)
        writer.writeheader()
        writer.writerows({k: rows[name].get(k, "") for k in COUNT_FIELDS} for name in sorted(rows))
    os.replace(tmp, path)


def load_ground_truth(frames_dir: str | Path, annotations_dir: str | Path) -> tuple[list[GroundTruth], list[str]]:
    """Ground truth per usable frame, and notes on what was left out and why."""
    frames_dir, annotations_dir = Path(frames_dir), Path(annotations_dir)
    with open(frames_dir / "frames.csv", newline="", encoding="utf-8-sig") as f:
        manifest = {r["frame"]: r for r in csv.DictReader(f)}
    status = read_frame_status(frames_dir / "frames.csv")
    by_annotator = read_annotator_counts(annotations_dir)
    truth, notes = [], []
    for frame, info in sorted(manifest.items()):
        w, h = int(info.get("width") or 0), int(info.get("height") or 0)
        shot = info.get("shot_type", "")
        points_file = annotations_dir / f"{Path(frame).stem}.json"
        if status.get(frame, ("",))[0] == UNREADABLE:
            notes.append(f"{frame}: left out, marked unreadable ({status[frame][1]})")
            continue
        rows = {name: counts[frame] for name, counts in by_annotator.items() if frame in counts}
        complete = {name: row for name, row in rows.items() if row.get("complete") == "yes"}
        incomplete = sorted(set(rows) - set(complete))
        if incomplete:
            notes.append(f"{frame}: not finished by {', '.join(incomplete)} (not used)")
        regions = {name: tuple(round(float(row[k])) for k in ("x0", "y0", "x1", "y1")) for name, row in complete.items()}
        if len(set(regions.values())) > 1:
            notes.append(f"{frame}: left out, annotators counted different regions "
                         f"({'; '.join(f'{n}: {r}' for n, r in sorted(regions.items()))})")
            continue
        if complete:
            first = next(iter(complete.values()))
            region = tuple(float(first[k]) for k in ("x0", "y0", "x1", "y1"))
            totals = tuple((name, float(row["total"])) for name, row in sorted(complete.items()))
            count = sum(t for _, t in totals) / len(totals)
            truth.append(GroundTruth(frame, shot, count, region, "counts", w, h, totals))
            if points_file.exists():
                notes.append(f"{frame}: counts used; the point file {points_file.name} is ignored")
        elif points_file.exists():
            ann = load_points_json(points_file)
            region = ann.roi or (0.0, 0.0, float(ann.width), float(ann.height))
            count = float(in_roi(ann.points, ann.roi).sum())
            truth.append(GroundTruth(frame, shot, count, region, "points", ann.width, ann.height))
        else:
            notes.append(f"{frame}: left out, not annotated")
    return truth, notes


def agreement(truth: list[GroundTruth]) -> list[dict]:
    """Per counted frame: each annotator's total, their mean, and the spread (max - min) / mean in %
    ("" with a single annotator; for two people it is their % difference)."""
    names = sorted({name for gt in truth for name, _ in gt.annotators})
    rows = []
    for gt in truth:
        if gt.source != "counts":
            continue
        totals = dict(gt.annotators)
        values = list(totals.values())
        spread = 100 * (max(values) - min(values)) / gt.count if len(values) > 1 and gt.count else ""
        rows.append({"frame": gt.frame, "shot_type": gt.shot_type,
                     **{name: f"{totals[name]:g}" if name in totals else "-" for name in names},
                     "mean": round(gt.count, 1), "annotators": len(values),
                     "difference %": round(spread, 1) if spread != "" else "-"})
    return rows


def density_in_region(density: np.ndarray, frame_size: tuple[int, int], region: Region) -> float:
    """Sum of the density map's cells whose centres lie in `region` (frame pixels).

    `density` covers the whole frame at any resolution (CSRNet: 1/8 of its input size)."""
    gh, gw = density.shape
    w, h = frame_size
    cx = (np.arange(gw) + 0.5) * w / gw
    cy = (np.arange(gh) + 0.5) * h / gh
    x0, y0, x1, y1 = region
    mask = ((cy >= y0) & (cy < y1))[:, None] & ((cx >= x0) & (cx < x1))[None, :]
    return float(density[mask].sum())


def boxes_in_region(boxes: np.ndarray, region: Region) -> int:
    """Person boxes (x1, y1, x2, y2) whose estimated head position lies in `region`: horizontally
    centred, 0.2 x box width under the top (capped at mid-height), the point a person would click."""
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    heads = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 1] + np.minimum(0.2 * w, 0.5 * h)])
    return int(in_roi(heads, region).sum())


def score(truth: list[GroundTruth], predicted: dict[str, float]) -> tuple[list[dict], list[dict]]:
    """Per-frame rows and per-shot-type summary rows (plus "all") of the count error."""
    frames, errors = [], {}
    for gt in truth:
        pred = predicted[gt.frame]
        errors[gt.frame] = abs(pred - gt.count)
        frames.append({"frame": gt.frame, "shot_type": gt.shot_type, "true": gt.count, "predicted": round(pred, 1),
                       "abs_error": round(errors[gt.frame], 1),
                       "pct_error": round(100 * errors[gt.frame] / gt.count, 1) if gt.count else "",
                       "ground_truth": gt.source})
    summary = []
    for shot in sorted({gt.shot_type for gt in truth}) + ["all"]:
        group = [gt for gt in truth if shot in ("all", gt.shot_type)]
        true_sum = sum(gt.count for gt in group)
        abs_sum = sum(errors[gt.frame] for gt in group)
        summary.append({"shot_type": shot, "frames": len(group), "true_heads": round(true_sum, 1),
                        "mae": round(abs_sum / len(group), 1),
                        "relative_error_pct": round(100 * abs_sum / true_sum, 1) if true_sum else ""})
    return frames, summary
