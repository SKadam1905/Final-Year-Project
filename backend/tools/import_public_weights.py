"""Convert the CSRNet authors' public ShanghaiTech checkpoint into our format, with provenance.

    python -m tools.import_public_weights ml\\weights\\public\\PartAmodel_best.pth.tar ml\\weights\\public\\csrnet_partA_public.pth

The authors' file (github.com/leeyeehoo/CSRNet-pytorch, README "ShanghaiA MAE: 66.4", Google
Drive id 1Z-atzS5Y2pOd-nEWqZRVBDMYJDreGWHH) is an old PyTorch pickle that the safe loader
cannot read. This tool first lists every Python object the pickle would construct (pickletools,
nothing is executed) and refuses unless they are only tensors and ordered dicts; then it loads
it, checks every layer name and shape against our CSRNet, and saves the weights in the current
format with a "provenance" record, so they load with weights_only=True from then on.
"""

from __future__ import annotations

import argparse
import hashlib
import pickletools
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

from app.csrnet_model import CSRNet, extract_state_dict

SAFE_GLOBALS = {"collections OrderedDict", "torch FloatStorage", "torch._utils _rebuild_tensor_v2",
                "torch._utils _rebuild_tensor", "torch._utils _rebuild_parameter"}


def pickle_globals(path: Path) -> set[str]:
    """Every global a legacy (non-zip) torch file's pickles reference, without executing them."""
    found = set()
    with open(path, "rb") as f:
        for _ in range(5):  # magic number, protocol, sys info, the object, storage keys
            try:
                for op, arg, _ in pickletools.genops(f):
                    if op.name in ("GLOBAL", "INST"):
                        found.add(str(arg))
                    elif op.name == "STACK_GLOBAL":
                        found.add("STACK_GLOBAL")  # name not visible statically: treat as unsafe
            except Exception:
                break
    return found


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--dataset", default="ShanghaiTech Part A")
    p.add_argument("--origin", default="github.com/leeyeehoo/CSRNet-pytorch, Google Drive id 1Z-atzS5Y2pOd-nEWqZRVBDMYJDreGWHH")
    p.add_argument("--reported-mae", type=float, default=66.4, help="MAE the authors report (not measured here)")
    args = p.parse_args(argv)

    if zipfile.is_zipfile(args.source):
        checkpoint = torch.load(args.source, map_location="cpu", weights_only=True)
    else:
        unsafe = pickle_globals(args.source) - SAFE_GLOBALS
        if unsafe:
            raise SystemExit(f"refusing to load: the pickle references {sorted(unsafe)}")
        checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)  # only tensors, checked above
    state = extract_state_dict(checkpoint)
    model = CSRNet(pretrained_frontend=False)
    model.load_state_dict(state)  # strict: every name and shape must match
    provenance = {
        "source": "public_pretrained",
        "dataset": args.dataset,
        "origin": args.origin,
        "original_file": args.source.name,
        "original_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "authors_epoch": int(checkpoint.get("epoch")) if checkpoint.get("epoch") is not None else None,
        "authors_reported_mae": args.reported_mae,
        "val_mae": None,
        "test_mae": None,  # filled in by an evaluation on the test set run here
        "test_rmse": None,
        "imported": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "provenance": provenance}, args.out)
    print(f"saved {args.out} ({len(state)} tensors, all names and shapes match our CSRNet)")
    for key, value in provenance.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
