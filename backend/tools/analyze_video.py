"""Run the full per-zone pipeline on a video file, offline (no Redis or database needed).

    python -m tools.analyze_video ../data/demo_videos/zone1.mp4 --weights ml/weights/parta_best.pth --area 150

Every --interval seconds of video it takes --burst consecutive frames, runs CSRNet on the
middle one and Farneback flow across the burst, then updates the zone's smoothed risk
level. Writes an annotated video (density heatmap + status banner) and a CSV of readings,
which are useful for thesis figures and for tuning k and the thresholds.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from app.config import REPO_ROOT, ZoneConfig, get_settings, load_zones
from app.csrnet_model import DensityEstimator
from app.pipeline import ZoneAnalyzer
from app.visualize import density_scale, draw_banner, draw_density_legend, draw_polygon, overlay_density_m2


def parse_roi(text: str | None) -> tuple[tuple[float, float], ...] | None:
    if not text:
        return None
    return tuple(tuple(float(v) for v in pair.split(",")) for pair in text.split(";"))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--allow-pickle", action="store_true", help="for old checkpoints; trusted files only")
    p.add_argument("--area", type=float, help="ground area of the ROI in m² (if no calibration)")
    p.add_argument("--roi", help="ROI polygon in pixels: 'x1,y1;x2,y2;...' (default: whole frame)")
    p.add_argument("--zone", help="use this zone's ROI/calibration from config/zones.json instead")
    p.add_argument("--interval", type=float, default=2.0, help="seconds of video between bursts")
    p.add_argument("--burst", type=int, default=5)
    p.add_argument("--out", type=Path, help="annotated video (default: outputs/<video>_analysis.mp4)")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    settings = get_settings()
    if args.zone:
        zone = next((z for z in load_zones(settings.zones_config) if z.id == args.zone), None)
        if zone is None:
            raise SystemExit(f"zone {args.zone!r} not in {settings.zones_config}")
    elif args.area:
        zone = ZoneConfig(id=args.video.stem, name=args.video.stem, source=str(args.video),
                          roi=parse_roi(args.roi), area_m2=args.area)
    else:
        raise SystemExit("give --area (m² of the ROI) or --zone with calibration")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = args.out or REPO_ROOT / "outputs" / f"{args.video.stem}_analysis.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    estimator = DensityEstimator(args.weights, device=args.device, max_width=settings.infer_max_width,
                                 allow_pickle=args.allow_pickle)
    analyzer = ZoneAnalyzer(zone, estimator, settings)
    step = max(args.burst, round(args.interval * fps))
    scale = density_scale(settings.risk_yellow, settings.risk_orange, settings.risk_red)  # fixed persons/m² colours
    print(f"{args.video.name}: {width}x{height} @ {fps:.1f} fps, burst every {step} frames on {estimator.device}")

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["time_s", "count", "density_m2", "turbulence", "pressure", "smoothed", "level",
                      "camera_motion_px", "flag", "inference_ms", "flow_ms"])
        burst, last, index = [], None, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step < args.burst:
                burst.append(frame)
            if len(burst) == args.burst:
                last = analyzer.analyze(burst, timestamp=index / fps)
                if last.scene_cut:  # skipped: no reading, the risk average is untouched
                    log.writerow([f"{index / fps:.1f}", "", "", "", "", "", "", "", "scene cut", "", ""])
                    print(f"t={index / fps:6.1f}s  scene cut: skipped")
                    last = None
                else:
                    s, t = last.status, last.turbulence
                    log.writerow([f"{s.timestamp:.1f}", f"{s.count:.1f}", f"{s.density:.3f}", f"{s.turbulence:.3f}",
                                  f"{s.pressure:.3f}", f"{s.smoothed_pressure:.3f}", s.level.name,
                                  f"{t.camera_motion:.2f}", "moving camera" if t.moving_camera else "",
                                  f"{last.inference_ms:.0f}", f"{last.flow_ms:.0f}"])
                    print(f"t={s.timestamp:6.1f}s  count {s.count:7.1f}  {s.density:5.2f}/m²  turb {s.turbulence:.2f}  "
                          f"pressure {s.smoothed_pressure:5.2f}  {s.level.name:6s}  ({last.inference_ms:.0f} ms CSRNet)")
                burst = []
            if last is not None:
                per_m2 = analyzer.geometry.cell_density(last.density.density, last.density.frame_size)
                frame = draw_density_legend(overlay_density_m2(frame, per_m2, scale), scale)
                s = last.status
                draw_banner(frame, [f"{s.level.name}  pressure {s.smoothed_pressure:.2f}",
                                    f"~{s.count:.0f} people  {s.density:.2f}/m2  turbulence {s.turbulence:.2f}"], s.level)
            if zone.roi:
                draw_polygon(frame, zone.roi)
            writer.write(frame)
            index += 1
    cap.release()
    writer.release()
    print(f"annotated video -> {out_path}\nreadings -> {csv_path}")


if __name__ == "__main__":
    main()
