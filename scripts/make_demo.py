"""Demo setup for trying the whole system by hand before CSRNet is trained.

    venv\\Scripts\\python.exe scripts\\make_demo.py        # -> data\\demo\\ (git-ignored)

Writes:
    crowd_clip.avi     10 s of dots drifting right (640x360, loops)
    demo_csrnet.pth    a CSRNet that predicts about 700 people on any 640x360 frame. NOT a trained
                       model: it ignores the image, it only drives the pipeline
    zones_demo.json    4 zones on that clip whose areas give about 7, 5, 3 and 1 person/m2,
                       i.e. RED, ORANGE, YELLOW and GREEN

Then run the backend with ZONES_CONFIG, CSRNET_WEIGHTS and FRAME_SOURCE pointing at these and
DEMO_MODE=1 (the backend refuses demo weights otherwise; the script prints the commands). With ALERT_CHANNEL=telegram the RED and ORANGE zones each
send one Telegram alert, then stay quiet for the 10 min cooldown.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.csrnet_model import CSRNet, DensityEstimator  # noqa: E402

OUT = ROOT / "data" / "demo"
W, H, PEOPLE = 640, 360, 700
ZONES = [  # area -> 700 people / area = persons per m2
    ("RAMKUND_NORTH", "Ramkund north steps", 100),
    ("RAMKUND_SOUTH", "Ramkund south steps", 140),
    ("GHAT_1", "Ghat 1 approach", 233),
    ("GHAT_2", "Ghat 2 riverside", 700),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clip = OUT / "crowd_clip.avi"
    rng = np.random.default_rng(0)
    dots = rng.uniform((0, 0), (W, H), (400, 2))
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 25, (W, H))
    for _ in range(250):
        dots[:, 0] = (dots[:, 0] + 1.5) % W
        frame = np.full((H, W, 3), (90, 110, 130), np.uint8)
        for x, y in dots.astype(int):
            cv2.circle(frame, (int(x), int(y)), 6, (30, 40, 60), -1)
        writer.write(frame)
    writer.release()

    torch.manual_seed(0)
    model = CSRNet(pretrained_frontend=False)
    with torch.no_grad():
        model.output_layer.weight.normal_(0, 1e-3)
        model.output_layer.bias.fill_(PEOPLE / ((W // 8) * (H // 8)))  # spread evenly over the density cells
    weights = OUT / "demo_csrnet.pth"
    # Marked as a demo: the backend refuses it unless DEMO_MODE=1 is set (only the tests and demos use it).
    torch.save({"model": model.state_dict(), "demo": True,
                "provenance": {"source": "demo", "dataset": None, "note": "fixed count, ignores the image"}}, weights)

    zones = [{"id": zid, "name": name, "source": "data/demo/crowd_clip.avi", "area_m2": area}
             for zid, name, area in ZONES]
    (OUT / "zones_demo.json").write_text(json.dumps({"zones": zones}, indent=2), encoding="utf-8")

    ok, frame = cv2.VideoCapture(str(clip)).read()
    count = DensityEstimator(weights, device="cpu").predict(frame).count
    print(f"wrote {OUT}; demo model count on a frame: {count:.0f} people")
    print("\nRun the backend (from backend\\, in one PowerShell window):")
    print('  $env:ZONES_CONFIG = "data/demo/zones_demo.json"')
    print('  $env:CSRNET_WEIGHTS = "data/demo/demo_csrnet.pth"')
    print('  $env:FRAME_SOURCE = "local"')
    print('  $env:DEMO_MODE = "1"')
    print("  ..\\venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000")


if __name__ == "__main__":
    main()
