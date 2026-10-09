"""Phase 5A step 3: check the ground-truth density maps.

    venv\\Scripts\\python.exe scripts\\check_density_maps.py

For every image of the processed ShanghaiTech Part A and Part B datasets, compares the sum of its
density map with its number of annotated heads (each map must be within 1%), and writes:
    results\\density_map_check.csv / .md   per dataset and split: images, heads, worst error
    results\\density_map_examples.png       4 images with their density maps overlaid
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.image_io import read_image  # noqa: E402
from ml.dataset_index import read_index  # noqa: E402
from report_utils import save_figure, write_table  # noqa: E402

DATASETS = {"ShanghaiTech Part A (adaptive kernels)": ROOT / "data" / "processed" / "shanghaitech_a",
            "ShanghaiTech Part B (sigma 15)": ROOT / "data" / "processed" / "shanghaitech_b"}


def main() -> None:
    rows, examples = [], []
    worst_overall = (0.0, "")
    for name, root in DATASETS.items():
        if not (root / "index.csv").exists():
            print(f"{name}: not prepared, skipped")
            continue
        for split in ("train", "val", "test"):
            samples = read_index(root, [split])
            errors = []
            for s in samples:
                total = float(np.load(s.density).astype(np.float64).sum())
                error = abs(total - s.count) / max(s.count, 1.0)
                errors.append((error, s.id, s.count, total))
            worst = max(errors)
            worst_overall = max(worst_overall, (worst[0], f"{name} {split} {worst[1]}"))
            rows.append({"dataset": name, "split": split, "images": len(samples),
                         "heads": int(round(sum(s.count for s in samples))),
                         "density_sum": round(sum(e[3] for e in errors), 1),
                         "worst_error_pct": f"{100 * worst[0]:.1e}", "worst_image": worst[1],
                         "maps_over_1pct": sum(e[0] > 0.01 for e in errors)})
            print(rows[-1])
        train = sorted(read_index(root, ["train"]), key=lambda s: s.count)
        picks = [train[len(train) // 10], train[len(train) // 2], train[-1]] if "Part A" in name else [train[len(train) // 2]]
        examples += [(name, s) for s in picks]
    write_table(ROOT / "results" / "density_map_check", rows, "Ground-truth density maps: sum against head count", [
        "Each head is a Gaussian normalised to 1 after clipping at the image border, then the map is sum-pooled to "
        "1/8 resolution (equivalent to resizing by 1/8 and multiplying by 64), so a map's sum equals its head count.",
        "Part A: geometry-adaptive kernels, sigma = 0.3 x mean distance to the 3 nearest heads. Part B: fixed sigma 15.",
        f"Worst relative error over all maps: {100 * worst_overall[0]:.1e}% ({worst_overall[1]}). Requirement: within 1%.",
        "Validation = 30 of Part A's 300 training images (40 of Part B's 400), chosen with a fixed seed; the test "
        "images are untouched.",
    ])

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, (name, s) in zip(axes.ravel(), examples):
        image = cv2.cvtColor(read_image(s.image), cv2.COLOR_BGR2RGB)
        density = np.load(s.density).astype(np.float32)
        up = cv2.resize(density, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        ax.imshow(image)
        heat = ax.imshow(up / 64.0, cmap="jet", alpha=0.5)  # /64: back to heads per original pixel
        ax.set_title(f"{name.split(' (')[0]} {s.id}: {s.count:.0f} heads, map sums to {density.sum():.1f}", fontsize=9)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        fig.colorbar(heat, ax=ax, fraction=0.03, label="density (heads per pixel)")
    fig.suptitle("Ground-truth density maps over their images")
    save_figure(fig, ROOT / "results" / "density_map_examples.png")


if __name__ == "__main__":
    main()
