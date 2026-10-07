r"""Evaluate CSRNet: MAE and RMSE of predicted counts, a per-image CSV and heatmaps (plan §8).

    # A processed dataset from ml.prepare_dataset (its held-out test split)
    python -m ml.evaluate --weights ml\weights\parta_best.pth --data ..\data\processed\shanghaitech_a --out ml\results

    # A raw ShanghaiTech folder: true counts from the GT_IMG_*.mat files beside images\
    python -m ml.evaluate --weights ml\weights\csrnet_best.pth --images ..\data\shanghaitech_part_a\part_A_final\test_data\images --out ml\results

    # ...or from density maps instead (one <image name>.npy per image, summed)
    python -m ml.evaluate --weights ml\weights\csrnet_best.pth --images <folder> --dmaps <folder> --out ml\results

    # The CSRNet authors' released Part A checkpoint is an old pickle: add --allow-pickle
    # (only for files from that trusted source)

Writes <out>\results.csv (image_name, gt_count, pred_count, abs_error) and a predicted-density
heatmap PNG for every 10th image (--vis-every). Targets on ShanghaiTech Part A are the paper's
numbers (Li et al. 2018): MAE 68.2, RMSE 115.0. Crowd-counting papers label RMSE as "MSE";
this script reports RMSE.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.csrnet_model import DensityEstimator
from app.image_io import read_image, write_image
from app.visualize import overlay_density
from ml.annotations import load_shanghaitech_mat
from ml.dataset_index import read_index

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class EvalItem:
    name: str
    image: Path
    gt_count: float


def items_from_processed(roots: list[Path], split: str) -> list[EvalItem]:
    return [EvalItem(s.id, s.image, s.count) for root in roots for s in read_index(root, [split])]


def _shanghaitech_mat(image: Path) -> Path:
    for folder in sorted(p for p in image.parent.parent.iterdir() if p.is_dir()):
        mat = folder / f"GT_{image.stem}.mat"
        if folder.name.lower().startswith("ground") and mat.exists():
            return mat
    raise FileNotFoundError(f"no GT_{image.stem}.mat in a ground-truth folder next to {image.parent}; "
                            "pass --dmaps with density maps instead")


def items_from_folder(images: Path, dmaps: Path | None = None) -> list[EvalItem]:
    """True count = sum of <dmaps>/<stem>.npy, or the number of heads in ShanghaiTech's .mat."""
    items = []
    for image in sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        if dmaps is not None:
            gt = float(np.load(dmaps / f"{image.stem}.npy").sum())
        else:
            gt = float(len(load_shanghaitech_mat(_shanghaitech_mat(image))))
        items.append(EvalItem(image.name, image, gt))
    return items


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", type=Path, action="append", help="processed dataset folder (repeatable)")
    source.add_argument("--images", type=Path, help="folder of raw images (ShanghaiTech layout)")
    p.add_argument("--dmaps", type=Path, help="with --images: density maps (.npy) giving the true counts")
    p.add_argument("--split", default="test", help="with --data: which split to evaluate")
    p.add_argument("--out", type=Path, default=Path("ml") / "results")
    p.add_argument("--vis-every", type=int, default=10, help="save a heatmap for every Nth image (0 = none)")
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--allow-pickle", action="store_true", help="needed for old checkpoints; trusted files only")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    items = items_from_folder(args.images, args.dmaps) if args.images else items_from_processed(args.data, args.split)
    if not items:
        raise SystemExit("no images to evaluate")
    estimator = DensityEstimator(args.weights, device=args.device, max_width=None,
                                 amp=not args.no_amp, allow_pickle=args.allow_pickle)
    args.out.mkdir(parents=True, exist_ok=True)

    rows, errors = [], []
    for i, item in enumerate(items):
        image = read_image(item.image)
        result = estimator.predict(image)
        error = result.count - item.gt_count
        errors.append(error)
        rows.append([item.name, f"{item.gt_count:g}", f"{result.count:.2f}", f"{abs(error):.2f}"])
        if args.vis_every and i % args.vis_every == 0:
            name = f"{Path(item.name).stem}_gt{item.gt_count:.0f}_pred{result.count:.0f}.png"
            write_image(args.out / "heatmaps" / name, overlay_density(image, result.density))

    with open(args.out / "results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "gt_count", "pred_count", "abs_error"])
        writer.writerows(rows)

    errors_arr = np.asarray(errors)
    mae = float(np.abs(errors_arr).mean())
    rmse = math.sqrt(float((errors_arr**2).mean()))
    print(f"{len(items)} images on {estimator.device}: MAE {mae:.2f}  RMSE {rmse:.2f}")
    print("Paper (Li et al. 2018), ShanghaiTech Part A: MAE 68.2, RMSE 115.0")
    print(f"per-image results -> {args.out / 'results.csv'}")


if __name__ == "__main__":
    main()
