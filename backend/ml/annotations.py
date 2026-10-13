"""Head-point annotation formats, all loaded as (N, 2) float arrays of (x, y) pixels.

* ShanghaiTech - GT_IMG_<n>.mat holding image_info{1}.location
* UCF-QNRF     - img_<n>_ann.mat holding annPoints
* CVAT         - "CVAT for images 1.1" XML export with Points shapes
* Ours         - one JSON per frame, written by tools/annotate_points.py and
                 tools/cvat_to_points.py: annotations/<video>/<frame>.json. An optional
                 "roi" [x0, y0, x1, y1] means only that rectangle was annotated (used for
                 evaluation frames too dense to annotate completely).
"""

from __future__ import annotations

import csv
import json
import os
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from scipy.io import loadmat


def load_shanghaitech_mat(path: str | Path) -> np.ndarray:
    record = loadmat(str(path))["image_info"][0, 0][0, 0]
    names = record.dtype.names or ()
    location = record["location"] if "location" in names else record[0]
    return np.asarray(location, np.float64).reshape(-1, 2)


def load_qnrf_mat(path: str | Path) -> np.ndarray:
    return np.asarray(loadmat(str(path))["annPoints"], np.float64).reshape(-1, 2)


@dataclass
class PointAnnotation:
    image: str  # frame file name, e.g. frame_0001.jpg
    video: str  # source video id = folder name under frames/
    width: int
    height: int
    points: np.ndarray  # (N, 2) x, y in pixels
    roi: tuple[float, float, float, float] | None = None  # x0, y0, x1, y1; None = whole frame annotated


def in_roi(points: np.ndarray, roi: tuple[float, float, float, float] | None) -> np.ndarray:
    """Boolean mask of the points inside roi (half-open [x0, x1) x [y0, y1)); all True when roi is None."""
    points = np.asarray(points, np.float64).reshape(-1, 2)
    if roi is None:
        return np.ones(len(points), bool)
    x0, y0, x1, y1 = roi
    return (points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)


def save_points_json(path: str | Path, ann: PointAnnotation) -> None:
    """Write atomically, so a crash mid-save never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"image": ann.image, "video": ann.video, "width": ann.width, "height": ann.height}
    if ann.roi is not None:
        data["roi"] = [round(float(v), 2) for v in ann.roi]
    data["count"] = int(len(ann.points))
    data["points"] = [[round(float(x), 2), round(float(y), 2)] for x, y in np.asarray(ann.points).reshape(-1, 2)]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def load_points_json(path: str | Path) -> PointAnnotation:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return PointAnnotation(
        image=data["image"],
        video=data.get("video", Path(path).parent.name),
        width=int(data["width"]),
        height=int(data["height"]),
        points=np.asarray(data["points"], np.float64).reshape(-1, 2),
        roi=tuple(float(v) for v in data["roi"]) if data.get("roi") else None,
    )


def read_cvat_export(path: str | Path) -> bytes:
    """annotations.xml of a CVAT export: the .xml itself, or the one inside the downloaded .zip."""
    path = Path(path)
    if not zipfile.is_zipfile(path):
        return path.read_bytes()
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if PurePosixPath(n).name == "annotations.xml"]
        if not names:
            raise ValueError(f"{path} has no annotations.xml; export as 'CVAT for images 1.1'")
        return z.read(min(names, key=len))


def _cvat_images(source: str | Path | bytes) -> list[ET.Element]:
    root = ET.fromstring(source) if isinstance(source, bytes) else ET.parse(str(source)).getroot()
    images = root.findall("image")
    if not images and root.find("track") is not None:
        raise ValueError("this is a 'CVAT for video' export; export the task as 'CVAT for images 1.1'")
    return images


def parse_cvat_tags(source: str | Path | bytes) -> dict[str, set[str]]:
    """Image-level tags (e.g. "done", "bad_crop") of a CVAT for images 1.1 export, by image file name."""
    return {PurePosixPath(i.get("name", "").replace("\\", "/")).name: {t.get("label", "") for t in i.findall("tag")}
            for i in _cvat_images(source)}


def write_cvat_xml(path: str | Path, images: list[tuple[str, int, int, np.ndarray]], labels: dict[str, str],
                   point_label: str, task_name: str = "") -> None:
    """A "CVAT for images 1.1" file with one single-point shape per head, for CVAT's Upload annotations.

    `images` are (file name, width, height, points); ids follow sorted file names, CVAT's
    default order. One shape per head lets an annotator delete or move each suggestion alone.
    `labels` maps label name -> CVAT type ("points", "tag").
    """
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    task = ET.SubElement(ET.SubElement(root, "meta"), "task")
    ET.SubElement(task, "name").text = task_name
    ET.SubElement(task, "size").text = str(len(images))
    ET.SubElement(task, "mode").text = "annotation"
    label_list = ET.SubElement(task, "labels")
    for name, kind in labels.items():
        label = ET.SubElement(label_list, "label")
        ET.SubElement(label, "name").text = name
        ET.SubElement(label, "type").text = kind
        ET.SubElement(label, "attributes")
    for index, (name, width, height, points) in enumerate(sorted(images, key=lambda i: i[0])):
        image = ET.SubElement(root, "image", id=str(index), name=name, width=str(width), height=str(height))
        for x, y in np.asarray(points, np.float64).reshape(-1, 2):
            ET.SubElement(image, "points", label=point_label, source="auto", occluded="0",
                          points=f"{x:.2f},{y:.2f}", z_order="0")
    ET.indent(root, space=" ")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(str(path), encoding="utf-8", xml_declaration=True)


def parse_cvat_xml(source: str | Path | bytes, label: str | None = None) -> list[PointAnnotation]:
    """Points shapes from a CVAT "CVAT for images 1.1" export (optionally one label only)."""
    result = []
    for image in _cvat_images(source):
        points = []
        for shape in image.findall("points"):
            if label is not None and shape.get("label") != label:
                continue
            for pair in shape.get("points", "").split(";"):
                if pair.strip():
                    x, y = pair.split(",")
                    points.append((float(x), float(y)))
        name = PurePosixPath(image.get("name", "").replace("\\", "/"))
        result.append(
            PointAnnotation(
                image=name.name,
                video=name.parent.name,
                width=int(image.get("width")),
                height=int(image.get("height")),
                points=np.asarray(points, np.float64).reshape(-1, 2),
            )
        )
    return result


# Frame status in a frames folder's frames.csv: an "unreadable" frame (heads cannot be made out)
# is left out of the evaluation instead of being guessed. tools/annotate_points.py sets it.
UNREADABLE = "unreadable"


def read_frame_status(frames_csv: str | Path) -> dict[str, tuple[str, str]]:
    """Frame file name -> (status, reason) from a frames.csv; frames without a status are left out."""
    path = Path(frames_csv)
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["frame"]: (r["status"], r.get("reason") or "") for r in csv.DictReader(f) if r.get("status")}


def set_frame_status(frames_csv: str | Path, frame: str, status: str, reason: str = "") -> None:
    """Set a frame's status and reason in frames.csv (status "" clears them), keeping every other
    column. A frame not listed yet is added, a missing file is created; written atomically."""
    path = Path(frames_csv)
    fields, rows = ["frame"], []
    if path.exists():
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields, rows = list(reader.fieldnames or fields), list(reader)
    fields += [c for c in ("status", "reason") if c not in fields]
    row = next((r for r in rows if r.get("frame") == frame), None)
    if row is None:
        row = {"frame": frame}
        rows.append(row)
    row["status"], row["reason"] = status, reason if status else ""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r.get(k) or "" for k in fields} for r in rows)
    os.replace(tmp, path)


def evaluation_frames(frames_dir: str | Path, annotations_dir: str | Path) -> list[tuple[Path, PointAnnotation]]:
    """(frame path, annotation) for every annotated frame not marked unreadable in frames.csv."""
    frames_dir = Path(frames_dir)
    status = read_frame_status(frames_dir / "frames.csv")
    result = []
    for path in sorted(Path(annotations_dir).glob("*.json")):
        ann = load_points_json(path)
        if status.get(ann.image, ("", ""))[0] != UNREADABLE:
            result.append((frames_dir / ann.image, ann))
    return result
