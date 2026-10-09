"""Part C: evaluate a trained CSRNet on the ShanghaiTech Part A test images, for the thesis.

    venv\\Scripts\\python.exe scripts\\report_csrnet.py --weights backend\\ml\\weights\\parta_best.pth

Runs once on the 182 held-out test images (the validation split chose the checkpoint; nothing
here is tuned on the test set) and writes to results/:
    csrnet_eval.csv / .md        per image: name, true count, predicted count, absolute error
    csrnet_summary.csv / .md     MAE and RMSE next to Li et al. (CVPR 2018): 68.2 and 115.0
    csrnet_scatter.png           predicted against true count, with y = x
    csrnet_samples.png           best, median and worst image with their predicted density maps
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.csrnet_model import DensityEstimator  # noqa: E402
from app.image_io import read_image  # noqa: E402
from ml.evaluate import items_from_folder, items_from_processed  # noqa: E402
from report_utils import save_figure, write_table  # noqa: E402

PAPER = {"mae": 68.2, "rmse": 115.0}  # Li, Zhang & Chen, CVPR 2018, Table (ShanghaiTech Part A)


def verdict(mae: float) -> str:
    gap = (mae - PAPER["mae"]) / PAPER["mae"] * 100
    if gap <= 0:
        return f"matches or beats the paper (MAE {abs(gap):.0f}% lower)"
    if gap <= 10:
        return f"close to the paper (MAE {gap:.0f}% higher, within 10%)"
    return f"does not match the paper (MAE {gap:.0f}% higher)"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "shanghaitech_a")
    source.add_argument("--images", type=Path, help="raw test images folder (true counts from GT_*.mat)")
    p.add_argument("--out", type=Path, default=ROOT / "results")
    p.add_argument("--allow-pickle", action="store_true", help="for the authors' old checkpoint; trusted files only")
    p.add_argument("--device", default="auto")
    p.add_argument("--record-provenance", action="store_true",
                   help="store the test MAE/RMSE in the checkpoint's provenance (refused if it already has one)")
    p.add_argument("--force", action="store_true", help=argparse.SUPPRESS)  # only if an earlier record was lost
    args = p.parse_args(argv)

    if args.record_provenance:
        import torch

        checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
        recorded = (checkpoint.get("provenance") or {}).get("test_mae")
        if recorded is not None and not args.force:
            raise SystemExit(f"{args.weights.name} was already evaluated on the test set (MAE {recorded}); "
                             "each model is evaluated on it once")

    items = items_from_folder(args.images) if args.images else items_from_processed([args.data], "test")
    if not items:
        raise SystemExit("no test images found")
    estimator = DensityEstimator(args.weights, device=args.device, max_width=None, allow_pickle=args.allow_pickle)
    preds = []
    for i, item in enumerate(items, 1):
        preds.append(estimator.predict(read_image(item.image)).count)
        if i % 50 == 0:
            print(f"  {i}/{len(items)}", flush=True)
    true = np.array([item.gt_count for item in items])
    pred = np.array(preds)
    err = np.abs(pred - true)
    mae, rmse = float(err.mean()), math.sqrt(float(((pred - true) ** 2).mean()))

    rows = [{"name": it.name, "true_count": f"{t:g}", "predicted_count": f"{p:.1f}", "absolute_error": f"{e:.1f}"}
            for it, t, p, e in zip(items, true, pred, err)]
    write_table(args.out / "csrnet_eval", rows, "CSRNet on ShanghaiTech Part A: per test image")
    summary = [
        {"metric": "MAE", "this_model": round(mae, 1), "paper": PAPER["mae"]},
        {"metric": "RMSE", "this_model": round(rmse, 1), "paper": PAPER["rmse"]},
    ]
    write_table(args.out / "csrnet_summary", summary, "CSRNet counting error on ShanghaiTech Part A (test set)", [
        f"{len(items)} test images, evaluated once at full resolution; checkpoint {args.weights.name}, "
        + ("released by the CSRNet authors (their checkpoint choice; nothing tuned here)."
           if estimator.info["kind"] == "public_pretrained" else
           "chosen on the validation split carved from the training images (no tuning on the test set)."),
        "Paper: Y. Li, X. Zhang, D. Chen, 'CSRNet: Dilated Convolutional Neural Networks for Understanding the "
        "Highly Congested Scenes', CVPR 2018. Papers in this field label RMSE as 'MSE'.",
        f"Result: this model {verdict(mae)}.",
        f"Mean true count {true.mean():.0f}; mean predicted {pred.mean():.0f} "
        f"({'over' if pred.mean() > true.mean() else 'under'}-counts on average by {abs(pred.mean() - true.mean()):.0f}).",
    ])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    top = float(max(true.max(), pred.max())) * 1.05
    ax.scatter(true, pred, s=14, alpha=0.7, color="#0f766e")
    ax.plot([0, top], [0, top], color="black", linewidth=1, label="y = x (perfect count)")
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("True count (people per image)")
    ax.set_ylabel("Predicted count (people per image)")
    ax.set_title(f"CSRNet on Part A test set: MAE {mae:.1f}, RMSE {rmse:.1f}")
    ax.legend(loc="upper left")
    save_figure(fig, args.out / "csrnet_scatter.png")

    order = np.argsort(err)
    picks = [("Best", order[0]), ("Median", order[len(order) // 2]), ("Worst", order[-1])]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for col, (label, idx) in enumerate(picks):
        image = read_image(items[idx].image)
        density = estimator.predict(image).density
        axes[0, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, col].set_title(f"{label}: {items[idx].name}\ntrue {true[idx]:.0f}, predicted {pred[idx]:.0f}")
        shown = axes[1, col].imshow(np.clip(density, 0, None), cmap="turbo")
        axes[1, col].set_title("Predicted density (1/8 resolution)")
        fig.colorbar(shown, ax=axes[1, col], fraction=0.046, label="people per 8x8 px cell")
        for row in (0, 1):
            axes[row, col].set_xlabel("x (px)" if row == 0 else "x (density cells)")
            axes[row, col].set_ylabel("y (px)" if row == 0 else "y (density cells)")
    save_figure(fig, args.out / "csrnet_samples.png")
    if args.record_provenance:
        from datetime import date

        checkpoint.setdefault("provenance", {}).update(
            test_mae=round(mae, 3), test_rmse=round(rmse, 3), test_images=len(items),
            test_set=str(args.images or args.data), evaluated=date.today().isoformat())
        torch.save(checkpoint, args.weights)
        print(f"stored test MAE {mae:.2f} / RMSE {rmse:.2f} in {args.weights.name}'s provenance")
    print(f"{len(items)} test images: MAE {mae:.2f}, RMSE {rmse:.2f} (paper 68.2 / 115.0) -> {verdict(mae)}")


if __name__ == "__main__":
    main()
