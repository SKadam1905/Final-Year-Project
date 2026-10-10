"""India dataset intake: raw_videos/<location>/<video>.mp4 listed in sources.csv.

sources.csv has one row per video (UTF-8; Excel's "CSV UTF-8" is fine):
    video_file   path under raw_videos/, e.g. nashik_ramkund/ramkund_01.mp4
    location     the location folder name (must match video_file), e.g. nashik_ramkund
    event        e.g. Simhastha Kumbh, Pandharpur Wari
    year         e.g. 2015
    source_url   where the video came from (cited in the thesis)
    camera_type  fixed / handheld / drone
plus optional provenance columns (OPTIONAL below). A video's id is its file name without the
extension, so ids must be unique across locations.

Videos directly in raw_videos/ (not in a location folder) are the demo clips: they feed the live
demo and the counting-accuracy evaluation frames (data/eval), so they are never part of this
training dataset.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import cv2

REQUIRED = ["video_file", "location", "event", "year", "source_url", "camera_type"]
OPTIONAL = ["title", "channel", "upload_date", "accessed_date", "clip_start_s", "clip_end_s", "notes"]
CAMERA_TYPES = ("fixed", "handheld", "drone")
VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")  # ids become file names and CVAT image names


@dataclass(frozen=True)
class VideoInfo:
    rel: str  # path under raw_videos/, forward slashes
    location: str  # "" for a demo clip directly in raw_videos/
    video_id: str
    readable: bool
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frames: int = 0

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps else 0.0


@dataclass
class IntakeReport:
    videos: list[VideoInfo] = field(default_factory=list)  # dataset videos (in a location folder)
    demo_clips: list[VideoInfo] = field(default_factory=list)  # directly in raw_videos/: not in the dataset
    sources: dict[str, dict] = field(default_factory=dict)  # video_file -> row
    not_in_csv: list[str] = field(default_factory=list)  # dataset videos without a sources.csv row
    missing_files: list[str] = field(default_factory=list)  # sources.csv rows without the video file
    problems: list[str] = field(default_factory=list)  # row errors, duplicate ids, unreadable videos
    warnings: list[str] = field(default_factory=list)  # usable, but worth a look (e.g. low resolution)

    @property
    def ok(self) -> bool:
        return not (self.not_in_csv or self.missing_files or self.problems)

    def source_of(self, video: VideoInfo) -> dict:
        return self.sources.get(video.rel, {})


def read_sources(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:  # -sig: tolerate Excel's byte-order mark
        return [{k.strip(): (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(f)]


def write_sources(path: Path, rows: list[dict]) -> None:
    extra = [k for row in rows for k in row if k not in REQUIRED + OPTIONAL]
    fields = REQUIRED + OPTIONAL + list(dict.fromkeys(extra))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def probe(raw_root: Path, path: Path) -> VideoInfo:
    rel = path.relative_to(raw_root).as_posix()
    location = PurePosixPath(rel).parent.as_posix()
    location = "" if location == "." else location
    cap = cv2.VideoCapture(str(path))
    try:
        ok, _ = cap.read()
        if not ok:
            return VideoInfo(rel, location, path.stem, False)
        return VideoInfo(rel, location, path.stem, True, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), float(cap.get(cv2.CAP_PROP_FPS) or 0),
                         int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    finally:
        cap.release()


def check_intake(root: Path) -> IntakeReport:
    """Compare raw_videos/ with sources.csv; probe each video's resolution and duration."""
    raw_root = root / "raw_videos"
    report = IntakeReport()
    files = sorted(p for p in raw_root.rglob("*") if p.suffix.lower() in VIDEO_EXTS) if raw_root.exists() else []
    for path in files:
        info = probe(raw_root, path)
        (report.videos if info.location else report.demo_clips).append(info)
        if not info.readable:
            report.problems.append(f"{info.rel}: OpenCV cannot read it (re-download as H.264 .mp4)")
        elif info.location and min(info.width, info.height) < 512:
            report.warnings.append(f"{info.rel}: {info.width}x{info.height} is below 512 px on its short side, so its "
                                   "crops will be smaller than 512x512 (prefer 720p or higher)")
        if info.location and "/" in info.location:
            report.problems.append(f"{info.rel}: put videos exactly one folder deep, raw_videos/<location>/<video>")

    rows = read_sources(root / "sources.csv")
    if rows and (missing := [c for c in REQUIRED if c not in rows[0]]):
        report.problems.append(f"sources.csv lacks the columns {missing} (expected {', '.join(REQUIRED)})")
        return report
    demo = {v.rel for v in report.demo_clips}
    present = {v.rel for v in report.videos}
    for n, row in enumerate(rows, start=2):  # line 1 is the header
        rel = row["video_file"].replace("\\", "/").strip("/")
        if not rel:
            report.problems.append(f"sources.csv line {n}: empty video_file")
            continue
        report.sources[rel] = row
        if rel in demo:
            continue  # a demo clip's provenance row: kept for citing, not validated as dataset data
        if rel not in present:
            report.missing_files.append(rel)
        folder = PurePosixPath(rel).parent.as_posix()
        where = f"sources.csv line {n} ({rel})"
        if folder == ".":
            report.problems.append(f"{where}: dataset videos go in a location folder, raw_videos/<location>/")
        elif row["location"] != folder:
            report.problems.append(f"{where}: location '{row['location']}' does not match the folder '{folder}'")
        empty = [c for c in REQUIRED if c != "source_url" and not row.get(c)]
        if empty:
            report.problems.append(f"{where}: empty {', '.join(empty)}")
        if not row.get("source_url"):
            report.warnings.append(f"{where}: no source_url (needed to cite the video)")
        if row.get("camera_type") and row["camera_type"].lower() not in CAMERA_TYPES:
            report.problems.append(f"{where}: camera_type '{row['camera_type']}' is not one of {', '.join(CAMERA_TYPES)}")
        if row.get("year") and not (row["year"].isdigit() and 1990 <= int(row["year"]) <= 2100):
            report.problems.append(f"{where}: year '{row['year']}' is not a 4-digit year")
    report.not_in_csv = sorted(present - set(report.sources))

    seen: dict[str, str] = {}
    for v in report.videos:
        if not SAFE_ID.match(v.video_id) or not SAFE_ID.match(v.location):
            report.problems.append(f"{v.rel}: use only letters, digits, _ and - in folder and file names")
        if v.video_id in seen:
            report.problems.append(f"{v.rel}: the video id '{v.video_id}' is also used by {seen[v.video_id]}; "
                                   "rename one (ids must be unique across locations)")
        seen.setdefault(v.video_id, v.rel)
    return report


def dataset_videos(root: Path) -> list[tuple[VideoInfo, Path]]:
    """The readable dataset videos with their paths; raises if the intake check has errors."""
    report = check_intake(root)
    if not report.ok:
        raise SystemExit("fix the intake problems first (python -m tools.india_dataset check)")
    return [(v, root / "raw_videos" / v.rel) for v in report.videos]
