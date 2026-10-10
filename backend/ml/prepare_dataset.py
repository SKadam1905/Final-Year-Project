"""Convert raw datasets into the processed training layout (see ml.dataset_index).

    python -m ml.prepare_dataset shanghaitech           # data/shanghaitech (Part A) -> data/processed/shanghaitech_a
    python -m ml.prepare_dataset shanghaitech_b         # data/shanghaitech (Part B) -> data/processed/shanghaitech_b
    python -m ml.prepare_dataset qnrf                   # data/ucf_qnrf -> data/processed/ucf_qnrf (long side <= 2048)
    python -m ml.prepare_dataset india --test-videos shirdi_01 --val-videos pandharpur_02

Each image is resized so both sides are multiples of 8 (and the long side <= --max-side);
head points are scaled with it, a density map is built (geometry-adaptive kernels; a fixed
sigma of 15 px for ShanghaiTech Part B, as in the CSRNet paper), then sum-pooled to 1/8
resolution (CSRNet's output stride), which keeps the count exact.

Splits: ShanghaiTech/QNRF keep their official train/test and carve `val` out of train
(--val-fraction). The India dataset is split by whole video (--test-videos/--val-videos,
or --test-fraction/--val-fraction of the videos).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from app.config import DATA_DIR
from app.csrnet_model import OUTPUT_STRIDE, stride_aligned_size
from app.image_io import read_image, write_image
from ml.annotations import load_points_json, load_qnrf_mat, load_shanghaitech_mat
from ml.dataset_index import Sample, choose_val_ids, split_videos, write_index
from ml.density_maps import density_map, sum_pool

LOADERS = {"shanghaitech": load_shanghaitech_mat, "qnrf": load_qnrf_mat, "json": lambda p: load_points_json(p).points}
DEFAULTS = {  # dataset -> (raw root, processed folder name, default max side)
    "shanghaitech": (DATA_DIR / "shanghaitech", "shanghaitech_a", None),
    "shanghaitech_b": (DATA_DIR / "shanghaitech", "shanghaitech_b", None),
    "qnrf": (DATA_DIR / "ucf_qnrf", "ucf_qnrf", 2048),
    "india": (DATA_DIR / "india_dataset" / "frames", "india", None),
}


@dataclass(frozen=True)
class RawItem:
    id: str
    split: str
    image: Path
    annotation: Path
    format: str  # key of LOADERS
    video: str = ""


def _find_base(root: Path, marker: str, prefer: str) -> Path:
    """`root` itself if it contains `marker`, else the unique (or `prefer`-named) nested one."""
    if (root / marker).is_dir():
        return root
    candidates = sorted(p.parent for p in root.rglob(marker) if p.is_dir())
    preferred = [c for c in candidates if prefer in c.as_posix().lower()]
    if len(preferred) == 1:
        return preferred[0]
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(f"expected one folder containing '{marker}' under {root}, found {candidates}")


FIXED_SIGMA = {"shanghaitech_b": 15.0}  # Part B: fixed kernels (sparser crowds); the others adaptive


def find_shanghaitech(root: Path, prefer: str = "part_a") -> list[RawItem]:
    base = _find_base(root, "train_data", prefer=prefer)
    items = []
    for split in ("train", "test"):
        folder = base / f"{split}_data"
        gt_dirs = [d for d in folder.iterdir() if d.is_dir() and d.name.lower().startswith("ground")]
        if not (folder / "images").is_dir() or len(gt_dirs) != 1:
            raise FileNotFoundError(f"{folder} should contain images/ and ground-truth/ (or ground_truth/)")
        for image in sorted((folder / "images").glob("*.jpg")):
            mat = gt_dirs[0] / f"GT_{image.stem}.mat"
            if not mat.exists():
                raise FileNotFoundError(f"missing annotation {mat}")
            items.append(RawItem(f"{split}_{image.stem}", split, image, mat, "shanghaitech"))
    return items


def find_qnrf(root: Path) -> list[RawItem]:
    base = _find_base(root, "Train", prefer="qnrf")
    items = []
    for split, folder in (("train", "Train"), ("test", "Test")):
        for image in sorted((base / folder).glob("*.jpg")):
            ann = image.with_name(f"{image.stem}_ann.mat")
            if not ann.exists():
                raise FileNotFoundError(f"missing annotation {ann}")
            items.append(RawItem(f"{split}_{image.stem}", split, image, ann, "qnrf"))
    return items


def find_point_json(frames_root: Path, annotations_root: Path) -> list[RawItem]:
    items = []
    for ann_path in sorted(annotations_root.glob("*/*.json")):
        video = ann_path.parent.name
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        if data.get("roi"):  # only part of the frame is annotated: unusable as a whole-frame density target
            print(f"skipping {ann_path}: annotated inside an ROI only")
            continue
        image = frames_root / video / data["image"]
        if not image.exists():
            raise FileNotFoundError(f"{ann_path} refers to missing frame {image}")
        items.append(RawItem(f"{video}_{ann_path.stem}", "train", image, ann_path, "json", video))
    return items


def process_item(item: RawItem, out_root: Path, max_side: int | None, sigma: float | None = None) -> Sample:
    image = read_image(item.image)
    points = LOADERS[item.format](item.annotation)
    h, w = image.shape[:2]
    max_width = None
    if max_side and max(h, w) > max_side:
        max_width = round(w * max_side / max(h, w))
    new_w, new_h = stride_aligned_size(w, h, max_width)
    if (new_w, new_h) != (w, h):
        interp = cv2.INTER_AREA if new_w < w else cv2.INTER_LINEAR
        image = cv2.resize(image, (new_w, new_h), interpolation=interp)
        points = points * np.array([new_w / w, new_h / h])
    density = sum_pool(density_map(points, (new_h, new_w), sigma=sigma), OUTPUT_STRIDE)

    image_path = out_root / "images" / f"{item.id}.jpg"
    density_path = out_root / "density" / f"{item.id}.npy"
    write_image(image_path, image, quality=95)
    density_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(density_path, density)
    return Sample(item.id, item.split, image_path, density_path, float(len(points)), item.video, new_w, new_h)


def assign_splits(items: list[RawItem], args: argparse.Namespace) -> list[RawItem]:
    if args.dataset == "india":
        split_of = split_videos(
            [i.video for i in items],
            test_videos=_csv(args.test_videos), val_videos=_csv(args.val_videos),
            test_fraction=args.test_fraction, val_fraction=args.val_fraction, seed=args.seed,
        )
        return [replace(i, split=split_of[i.video]) for i in items]
    val_ids = choose_val_ids([i.id for i in items if i.split == "train"], args.val_fraction, args.seed)
    return [replace(i, split="val") if i.id in val_ids else i for i in items]


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", choices=sorted(DEFAULTS))
    p.add_argument("--root", type=Path, help="raw dataset folder (india: the frames/ folder)")
    p.add_argument("--annotations", type=Path, help="india: annotations folder (default: next to frames/)")
    p.add_argument("--out", type=Path, help="output folder (default: data/processed/<name>)")
    p.add_argument("--max-side", type=int, help="downscale so the long side is at most this")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--test-videos", default="", help="india: comma-separated video ids for test")
    p.add_argument("--val-videos", default="", help="india: comma-separated video ids for val")
    p.add_argument("--test-fraction", type=float, default=0.25, help="india: share of videos for test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = p.parse_args(argv)
    default_root, name, default_max_side = DEFAULTS[args.dataset]
    args.root = args.root or default_root
    args.out = args.out or DATA_DIR / "processed" / name
    args.max_side = args.max_side or default_max_side
    if args.dataset == "india":
        args.annotations = args.annotations or args.root.parent / "annotations"
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    sigma = FIXED_SIGMA.get(args.dataset)
    if args.dataset in ("shanghaitech", "shanghaitech_b"):
        items = find_shanghaitech(args.root, prefer="part_b" if args.dataset == "shanghaitech_b" else "part_a")
    elif args.dataset == "qnrf":
        items = find_qnrf(args.root)
    else:
        items = find_point_json(args.root, args.annotations)
    if not items:
        raise SystemExit(f"no annotated images found under {args.root}")
    items = assign_splits(items, args)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{args.dataset}: {len(items)} images -> {args.out}")

    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            futures = [pool.submit(process_item, item, args.out, args.max_side, sigma) for item in items]
            samples = []
            for n, future in enumerate(futures, 1):
                samples.append(future.result())
                if n % 50 == 0:
                    print(f"  {n}/{len(items)}")
    else:
        samples = [process_item(item, args.out, args.max_side, sigma) for item in items]

    write_index(args.out, samples)
    per_split = Counter(s.split for s in samples)
    heads = Counter()
    for s in samples:
        heads[s.split] += s.count
    for split in ("train", "val", "test"):
        if per_split[split]:
            print(f"  {split:5s} {per_split[split]:5d} images  {heads[split]:9.0f} heads")
    if args.dataset == "india":
        for split in ("train", "val", "test"):
            videos = sorted({s.video for s in samples if s.split == split})
            print(f"  {split} videos: {', '.join(videos) or '-'}")


if __name__ == "__main__":
    main()
