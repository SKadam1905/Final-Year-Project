"""Show what a CSRNet weights file is: its provenance and how the backend will describe it.

    python -m tools.inspect_weights ml\\weights\\public\\csrnet_partA_public.pth

Loads with the safe loader (weights_only=True) only; an old pickled checkpoint must first go
through tools.import_public_weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from app.csrnet_model import describe_weights


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", type=Path, nargs="+")
    args = p.parse_args(argv)
    for path in args.weights:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        info = describe_weights(checkpoint, path)
        print(f"{path}\n  kind: {info['kind']}\n  {info['label']}")
        extra = {k: v for k, v in checkpoint.items() if k not in ("model", "state_dict", "optimizer", "scheduler", "scaler")}
        print("  metadata:", json.dumps(extra, default=str, indent=2).replace("\n", "\n  "))


if __name__ == "__main__":
    main()
