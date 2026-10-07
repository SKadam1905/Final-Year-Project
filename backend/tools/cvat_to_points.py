"""Convert a CVAT export into this project's per-frame JSON annotations.

    python -m tools.cvat_to_points annotations.xml --video pandharpur_01

In CVAT (app.cvat.ai): one task per video's frame folder, a label named "head", the
Points tool on each head, then Export task dataset -> "CVAT for images 1.1".
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import DATA_DIR
from ml.annotations import parse_cvat_xml, save_points_json


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("xml", type=Path, help="annotations.xml from the CVAT export")
    p.add_argument("--video", help="video id (default: the folder name in CVAT's image names)")
    p.add_argument("--label", default=None, help="only use Points with this label (e.g. head)")
    p.add_argument("--out", type=Path, default=DATA_DIR / "india_dataset" / "annotations")
    p.add_argument("--frames", type=Path, default=DATA_DIR / "india_dataset" / "frames")
    args = p.parse_args(argv)

    annotations = parse_cvat_xml(args.xml, args.label)
    written, heads, missing = 0, 0, []
    for ann in annotations:
        ann.video = args.video or ann.video
        if not ann.video:
            raise SystemExit("image names in the export have no folder; pass --video <id>")
        if not (args.frames / ann.video / ann.image).exists():
            missing.append(f"{ann.video}/{ann.image}")
        save_points_json(args.out / ann.video / f"{Path(ann.image).stem}.json", ann)
        written += 1
        heads += len(ann.points)
    print(f"{written} frames, {heads} heads -> {args.out}")
    if missing:
        print(f"warning: {len(missing)} frames not found under {args.frames} (e.g. {missing[0]}); "
              "ml.prepare_dataset needs them there")


if __name__ == "__main__":
    main()
