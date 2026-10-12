r"""India crowd dataset pipeline: videos -> distinct frames -> dense crops -> pre-annotated CVAT tasks
-> imported head points -> density maps split by location. See results\ANNOTATION_GUIDE.md.

    cd backend
    ..\venv\Scripts\python.exe -m tools.india_dataset check          # 1. videos vs sources.csv, duration, resolution
    ..\venv\Scripts\python.exe -m tools.india_dataset frames         # 2. ~10 distinct, usable frames per video
    ..\venv\Scripts\python.exe -m tools.india_dataset crops          # 3. up to 2 dense 512x512 crops per frame + contact sheets
    ..\venv\Scripts\python.exe -m tools.india_dataset preannotate    # 4. suggested head points for every crop
    ..\venv\Scripts\python.exe -m tools.india_dataset batches --annotators "A,B,C,D"   # 5. one CVAT task package each
    ..\venv\Scripts\python.exe -m tools.india_dataset import         # 6. CVAT exports -> checked points -> dataset
    ..\venv\Scripts\python.exe -m tools.india_dataset build          #    (split + density maps again, as the fine-tune does)

Everything is written under data\india_dataset\ (git-ignored: the frames show faces), except the
reports in results\ (counts and names only). Steps 2-4 skip videos already done; --force redoes
them. Reject a crop by listing its file name in crops\<video>\rejected.txt (or deleting it) before
step 5. Step 5 refuses to deal the batches again once dealt, because the team works from them.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from app.config import DATA_DIR, REPO_ROOT, get_settings
from app.image_io import read_image, write_image
from ml.annotation_batches import CropCheck, balanced_batches, point_flags, ratio_flags
from ml.annotations import (PointAnnotation, load_points_json, parse_cvat_tags, parse_cvat_xml, read_cvat_export,
                            save_points_json, write_cvat_xml)
from ml.crops import contact_sheet, crop_size, densest_crops
from ml.dataset_index import split_by_location
from ml.frame_select import WORK_WIDTH, read_frame, reject_unusable, scan_video, select_frames
from ml.india_sources import check_intake, dataset_videos
from ml.preannotate import suggest_points

LABELS = {"head": "points", "done": "tag", "bad_crop": "tag"}  # the CVAT task's labels
SCREEN_WIDTH = 640  # density-model width for screening frames (fast); crops use INFER_MAX_WIDTH


class Paths:
    def __init__(self, root: Path, results: Path) -> None:
        self.root, self.results = root, results
        self.frames, self.crops = root / "frames", root / "crops"
        self.suggestions, self.cvat = root / "suggestions", root / "cvat"
        self.exports, self.annotations = root / "cvat_exports", root / "crop_annotations"
        self.split = root / "split.csv"


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or (list(rows[0]) if rows else []))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_report(stem: Path, rows: list[dict], title: str, notes: list[str] = ()) -> None:
    """`<stem>.csv` and a Markdown table `<stem>.md` (same layout as the other results files)."""
    write_csv(stem.with_suffix(".csv"), rows)
    lines = [f"# {title}", ""]
    if rows:
        columns = list(rows[0])
        lines += ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        lines += ["| " + " | ".join(str(r[c]) for c in columns) + " |" for r in rows]
    lines += [""] + [f"- {n}" for n in notes]
    stem.with_suffix(".md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def selected(videos: list, wanted: list[str] | None) -> list:
    if not wanted:
        return videos
    unknown = set(wanted) - {v.video_id for v, _ in videos}
    if unknown:
        raise SystemExit(f"unknown videos: {sorted(unknown)}")
    return [(v, p) for v, p in videos if v.video_id in wanted]


def load_estimator(args):
    from app.csrnet_model import DensityEstimator, TRUSTED_KINDS

    s = get_settings()
    weights = Path(args.weights) if args.weights else s.csrnet_weights
    estimator = DensityEstimator(weights, device=args.device or s.device, max_width=s.infer_max_width,
                                 allow_pickle=s.csrnet_allow_pickle)
    if estimator.info["kind"] not in TRUSTED_KINDS:
        raise SystemExit(f"refusing {estimator.info['kind']} weights ({weights}): use trained or imported public weights")
    print(f"density model: {estimator.info['label']}")
    return estimator


def rejected_crops(folder: Path) -> set[str]:
    path = folder / "rejected.txt"
    if not path.exists():
        return set()
    names = {line.split("#")[0].strip() for line in path.read_text(encoding="utf-8").splitlines()}
    return {n if n.endswith(".jpg") else f"{n}.jpg" for n in names if n}


def usable_crops(p: Paths, video: str) -> list[dict]:
    """crops.csv rows not rejected (rejected.txt) and whose image still exists."""
    folder = p.crops / video
    rejected = rejected_crops(folder)
    return [r for r in read_csv(folder / "crops.csv") if r["crop"] not in rejected and (folder / r["crop"]).exists()]


# 1. intake -----------------------------------------------------------------------------------

def cmd_check(args, p: Paths) -> None:
    report = check_intake(p.root)
    rows = []
    for v in report.videos:
        src = report.source_of(v)
        rows.append({"video": v.video_id, "location": v.location, "file": v.rel,
                     "duration_s": round(v.duration_s, 1), "resolution": f"{v.width}x{v.height}",
                     "fps": round(v.fps, 2), "camera_type": src.get("camera_type", ""),
                     "event": src.get("event", ""), "year": src.get("year", ""),
                     "in_sources_csv": "yes" if src else "NO"})
    for r in rows:
        print(f"{r['file']:<45} {r['duration_s']:>7} s  {r['resolution']:>10}  {r['fps']:>6} fps  "
              f"{r['camera_type']:<8}  in sources.csv: {r['in_sources_csv']}")
    if report.demo_clips:
        print(f"\n{len(report.demo_clips)} demo clips directly in raw_videos/ (live demo and counting-accuracy "
              f"evaluation; not part of the dataset): {', '.join(v.rel for v in report.demo_clips)}")
    for title, items in (("Videos missing from sources.csv", report.not_in_csv),
                         ("sources.csv rows whose video file is missing", report.missing_files),
                         ("Problems", report.problems), ("Warnings", report.warnings)):
        if items:
            print(f"\n{title}:")
            for item in items:
                print(f"  - {item}")
    locations = Counter(v.location for v in report.videos)
    notes = [f"{len(report.videos)} videos from {len(locations)} locations, "
             f"{sum(v.duration_s for v in report.videos) / 60:.1f} min in total."]
    notes += [f"Missing from sources.csv: {v}" for v in report.not_in_csv]
    notes += [f"Listed in sources.csv but no file: {v}" for v in report.missing_files]
    notes += [f"Problem: {v}" for v in report.problems] + [f"Warning: {v}" for v in report.warnings]
    write_report(p.results / "india_intake", rows, "India dataset intake (raw_videos vs sources.csv)", notes)
    print(f"\n{'OK' if report.ok else 'NOT OK'}: {len(report.videos)} videos, {len(locations)} locations "
          f"-> {p.results / 'india_intake.md'}")
    if not report.videos:
        print("No dataset videos yet: put them in raw_videos/<location>/<video>.mp4 and list each in sources.csv.")
    if not report.ok:
        raise SystemExit(1)


# 2. frames -----------------------------------------------------------------------------------

def cmd_frames(args, p: Paths) -> None:
    videos = selected(dataset_videos(p.root), args.video)
    estimator = load_estimator(args)
    estimator.max_width = SCREEN_WIDTH
    summary = []
    for info, path in videos:
        out = p.frames / info.video_id
        if (out / "frames.csv").exists() and not args.force:
            print(f"{info.video_id}: frames already chosen (--force to redo)")
            continue
        scan = scan_video(path, count_fn=lambda frame: estimator.predict(frame).count)
        reject_unusable(scan.candidates)
        picks = select_frames(scan.candidates, n=args.per_video,
                              frame_area=WORK_WIDTH * WORK_WIDTH * scan.height / max(scan.width, 1))
        if out.exists():
            for old in out.glob("*.jpg"):
                old.unlink()
        rows = []
        for c in picks:
            name = f"{info.video_id}_t{c.time_s:06.1f}s.jpg"
            write_image(out / name, read_frame(path, c), quality=95)
            rows.append({"frame": name, "time_s": round(c.time_s, 2), "frame_index": c.frame_index,
                         "sharpness": round(c.sharpness, 1), "screen_count": round(c.count or 0, 1)})
        write_csv(out / "frames.csv", rows, ["frame", "time_s", "frame_index", "sharpness", "screen_count"])
        write_csv(out / "candidates.csv", [
            {"time_s": round(c.time_s, 2), "frame_index": c.frame_index, "sharpness": round(c.sharpness, 1),
             "flat_share": round(c.flat_share, 3), "screen_count": round(c.count or 0, 1),
             "result": "picked" if any(c is k for k in picks) else c.reason} for c in scan.candidates])
        (out / "overlays.json").write_text(json.dumps({
            "width": scan.width, "height": scan.height, "overlays": scan.overlays,
            "cuts_s": [round(t, 2) for t in scan.cuts_s]}, indent=1), encoding="utf-8")
        reasons = Counter(c.reason.split(" of ")[0] for c in scan.candidates if c.reason)
        summary.append({"video": info.video_id, "location": info.location, "candidates": len(scan.candidates),
                        "frames": len(picks), "scene_cuts": len(scan.cuts_s), "overlays": len(scan.overlays),
                        **{k.replace(" ", "_").replace("/", "or"): reasons.get(k, 0) for k in
                           ("scene cut", "title card / blank", "blurry", "little or no crowd", "near-duplicate",
                            "not needed (enough frames)")}})
        print(f"{info.video_id}: {len(picks)} frames from {len(scan.candidates)} candidates; "
              f"skipped {dict(reasons)}; {len(scan.overlays)} static overlays")
    if summary:
        write_report(p.results / "india_frames", summary, "India dataset: frames chosen per video", [
            "Candidates are sampled evenly (about 120 per video, at least 1 s apart); 'skipped' columns say why "
            "the others were not used. Overlays = static logos/tickers found near the frame border (crops avoid them).",
            f"Screening counts come from the density model at {SCREEN_WIDTH} px width.",
        ])


# 3. crops ------------------------------------------------------------------------------------

def cmd_crops(args, p: Paths) -> None:
    estimator = load_estimator(args)
    estimator.max_width = get_settings().infer_max_width
    videos = selected(dataset_videos(p.root), args.video)
    summary = []
    for info, _ in videos:
        frames_dir, out = p.frames / info.video_id, p.crops / info.video_id
        frames = [r for r in read_csv(frames_dir / "frames.csv") if (frames_dir / r["frame"]).exists()]
        if not frames:
            print(f"{info.video_id}: no frames (run the frames step first)")
            continue
        if (out / "crops.csv").exists() and not args.force:
            print(f"{info.video_id}: crops already cut (--force to redo)")
            continue
        overlays = json.loads((frames_dir / "overlays.json").read_text(encoding="utf-8")).get("overlays", [])
        for old in out.glob("*.jpg"):
            old.unlink()
        rows, tiles = [], []
        for r in frames:
            frame = read_image(frames_dir / r["frame"])
            h, w = frame.shape[:2]
            result = estimator.predict(frame)
            work_w, work_h = result.input_size
            work = frame if (work_w, work_h) == (w, h) else cv2.resize(
                frame, (work_w, work_h), interpolation=cv2.INTER_AREA if work_w < w else cv2.INTER_LINEAR)
            sx, sy = work_w / w, work_h / h
            exclude = [(x1 * sx, y1 * sy, x2 * sx, y2 * sy) for x1, y1, x2, y2 in overlays]
            size = crop_size(work_w, work_h, args.size)
            boxes = densest_crops(result.density, size, args.per_frame, exclude) or \
                densest_crops(result.density, size, 1)  # overlays everywhere: crop anyway, review on the sheet
            for k, box in enumerate(boxes, start=1):
                name = f"{Path(r['frame']).stem}_c{k}.jpg"
                crop = work[box.y:box.y + size, box.x:box.x + size]
                write_image(out / name, crop, quality=95)
                rows.append({"crop": name, "frame": r["frame"], "time_s": r["time_s"], "x": box.x, "y": box.y,
                             "size": size, "scale": round(sx, 4), "predicted": round(box.predicted, 1)})
                tiles.append((crop, f"{Path(name).stem}  pred {box.predicted:.0f}"))
        write_csv(out / "crops.csv", rows, ["crop", "frame", "time_s", "x", "y", "size", "scale", "predicted"])
        cv2.imwrite(str(out / "contact_sheet.jpg"), contact_sheet(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not (out / "rejected.txt").exists():
            (out / "rejected.txt").write_text(
                "# One crop file name per line (e.g. myvideo_t0012.0s_c2.jpg) to leave it out of annotation.\n"
                "# Reject crops that show mostly a logo, caption or screen, are blurred, or hold no crowd.\n",
                encoding="utf-8")
        summary.append({"video": info.video_id, "location": info.location, "frames": len(frames),
                        "crops": len(rows), "crop_size": rows[0]["size"] if rows else "",
                        "predicted_heads": round(sum(float(x["predicted"]) for x in rows))})
        print(f"{info.video_id}: {len(rows)} crops from {len(frames)} frames -> {out / 'contact_sheet.jpg'}")
    if summary:
        write_report(p.results / "india_crops", summary, "India dataset: crops per video", [
            f"Up to {args.per_frame} non-overlapping crops per frame, centred on the densest areas the density model "
            f"sees at the working width (INFER_MAX_WIDTH); a second crop only if it holds at least 30% of the first's "
            "predicted count. predicted_heads uses the current model, which undercounts dense, distant crowds.",
        ])


# 4. pre-annotation -----------------------------------------------------------------------------

def cmd_preannotate(args, p: Paths) -> None:
    from app.detector import load_detector

    s = get_settings()
    estimator = load_estimator(args)
    estimator.max_width = None  # crops are already at the working resolution
    detector, why = load_detector(s.yolo_weights, args.device or s.device, s.yolo_imgsz, s.yolo_conf)
    print(f"detector: {detector.info['model'] if detector else why}")
    for info, _ in selected(dataset_videos(p.root), args.video):
        crops = usable_crops(p, info.video_id)
        out = p.suggestions / info.video_id
        if not crops:
            print(f"{info.video_id}: no crops (run the crops step first)")
            continue
        if (out / "suggestions.csv").exists() and not args.force:
            print(f"{info.video_id}: already pre-annotated (--force to redo)")
            continue
        rows = []
        for r in crops:
            image = read_image(p.crops / info.video_id / r["crop"])
            sug = suggest_points(image, estimator, detector)
            h, w = image.shape[:2]
            save_points_json(out / f"{Path(r['crop']).stem}.json", PointAnnotation(r["crop"], info.video_id, w, h, sug.points))
            rows.append({"crop": r["crop"], "method": sug.method, "suggested": len(sug.points),
                         "density_count": round(sug.density_count, 1),
                         "detector_count": "" if sug.detector_count is None else sug.detector_count})
        write_csv(out / "suggestions.csv", rows)
        methods = Counter(r["method"] for r in rows)
        print(f"{info.video_id}: {len(rows)} crops, {sum(r['suggested'] for r in rows)} suggested heads "
              f"({dict(methods)})")


# 5. batches ------------------------------------------------------------------------------------

def cmd_batches(args, p: Paths) -> None:
    annotators = [a.strip() for a in args.annotators.split(",") if a.strip()]
    if not annotators:
        raise SystemExit("--annotators needs at least one name")
    if (p.cvat / "batch_crops.csv").exists() and not args.force:
        raise SystemExit(f"batches were already dealt ({p.cvat / 'batches.csv'}); the team works from them. "
                         "Use --force only if no one has started annotating.")
    report = check_intake(p.root)
    location = {v.video_id: v.location for v in report.videos}
    work, crops_of = {}, {}
    for video in sorted(location):
        sug = {r["crop"]: int(r["suggested"]) for r in read_csv(p.suggestions / video / "suggestions.csv")}
        crops = [r["crop"] for r in usable_crops(p, video) if r["crop"] in sug]
        if crops:
            crops_of[video] = crops
            work[video] = (len(crops), float(sum(sug[c] for c in crops)))
    if not work:
        raise SystemExit("no pre-annotated crops: run frames, crops and preannotate first")
    if len(work) < len(annotators):
        print(f"warning: {len(work)} videos for {len(annotators)} annotators; some batches will be empty")
    if p.cvat.exists():
        shutil.rmtree(p.cvat)
    batch_rows, crop_rows = [], []
    for k, (annotator, videos) in enumerate(zip(annotators, balanced_batches(work, len(annotators))), start=1):
        batch = f"batch_{k}"
        folder = p.cvat / batch
        images = []
        for video in videos:
            (folder / "images").mkdir(parents=True, exist_ok=True)
            for crop in crops_of[video]:
                shutil.copy2(p.crops / video / crop, folder / "images" / crop)
                ann = load_points_json(p.suggestions / video / f"{Path(crop).stem}.json")
                images.append((crop, ann.width, ann.height, ann.points))
                crop_rows.append({"batch": batch, "annotator": annotator, "video": video, "crop": crop,
                                  "suggested": len(ann.points)})
            batch_rows.append({"batch": batch, "annotator": annotator, "video": video, "location": location[video],
                               "crops": work[video][0], "suggested_heads": round(work[video][1])})
        if not images:
            continue
        write_cvat_xml(folder / "annotations.xml", images, LABELS, "head", task_name=f"india_{batch}")
        with zipfile.ZipFile(folder / "images.zip", "w", zipfile.ZIP_STORED) as z:  # JPEGs: no gain from deflate
            for name, *_ in images:
                z.write(folder / "images" / name, name)
        with zipfile.ZipFile(folder / "preannotations.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write(folder / "annotations.xml", "annotations.xml")
    (p.cvat / "labels.json").write_text(json.dumps(
        [{"name": n, "type": t, "attributes": []} for n, t in LABELS.items()], indent=1), encoding="utf-8")
    write_csv(p.cvat / "batches.csv", batch_rows)
    write_csv(p.cvat / "batch_crops.csv", crop_rows)
    p.exports.mkdir(parents=True, exist_ok=True)
    totals = defaultdict(lambda: [0, 0, 0])
    for r in batch_rows:
        t = totals[(r["batch"], r["annotator"])]
        t[0] += 1
        t[1] += r["crops"]
        t[2] += r["suggested_heads"]
    rows = [{"batch": b, "annotator": a, "videos": t[0], "crops": t[1], "suggested_heads": t[2],
             "video_ids": " ".join(r["video"] for r in batch_rows if r["batch"] == b)}
            for (b, a), t in sorted(totals.items())]
    write_report(p.results / "india_batches", rows, "India dataset: annotation batches", [
        "Whole videos per annotator (no video is split), balanced on crops and suggested heads. "
        "Suggested heads undercount dense crowds, so a batch with dense aerial videos is more work than it looks.",
        f"Task packages: {p.cvat}\\batch_<n>\\ (images.zip + preannotations.zip). Exports go to {p.exports}.",
    ])
    for r in rows:
        print(f"{r['batch']} -> {r['annotator']}: {r['videos']} videos, {r['crops']} crops, "
              f"{r['suggested_heads']} suggested heads ({r['video_ids']})")


# 6. import, check and build --------------------------------------------------------------------

def cmd_import(args, p: Paths) -> None:
    expected = {r["crop"]: r for r in read_csv(p.cvat / "batch_crops.csv")}
    if not expected:
        raise SystemExit("no batches dealt yet (step 5)")
    exports = [Path(e) for e in args.exports] or sorted(
        e for e in p.exports.glob("*") if e.suffix.lower() in (".zip", ".xml"))
    if not exports:
        raise SystemExit(f"no CVAT exports in {p.exports} (put each batch's export zip there)")
    found: dict[str, tuple[PointAnnotation, set[str], str]] = {}
    unknown, doubles = [], []
    for export in exports:
        xml = read_cvat_export(export)
        tags = parse_cvat_tags(xml)
        for ann in parse_cvat_xml(xml, label="head"):
            if ann.image not in expected:
                unknown.append(f"{export.name}: {ann.image}")
                continue
            if ann.image in found:
                doubles.append(f"{ann.image} in {found[ann.image][2]} and {export.name} (the later one is used)")
            found[ann.image] = (ann, tags.get(ann.image, set()), export.name)
        print(f"{export.name}: {len(tags)} images")

    checks = []
    for crop, row in expected.items():
        sug = load_points_json(p.suggestions / row["video"] / f"{Path(crop).stem}.json")
        check = CropCheck(crop, row["video"], row["batch"], "missing from exports", len(sug.points))
        if crop in found:
            ann, tags, _ = found[crop]
            if "bad_crop" in tags:
                check.status = "bad_crop"
            elif "done" not in tags:
                check.status = "not marked done"
            else:
                check.status = "done"
                check.corrected = len(ann.points)
                check.flags = point_flags(ann.points, ann.width, ann.height, sug.points)
        checks.append(check)
    ratio_flags(checks)

    status = Counter(c.status for c in checks)
    rows = [{"batch": c.batch, "video": c.video, "crop": c.crop, "status": c.status, "suggested": c.suggested,
             "corrected": "" if c.corrected is None else c.corrected, "flags": "; ".join(c.flags)}
            for c in checks if c.status != "done" or c.flags]
    notes = [f"{len(checks)} crops dealt: " + ", ".join(f"{n} {s}" for s, n in status.most_common()),
             f"{sum(1 for c in checks if c.flags)} done crops flagged for a second look (listed above with the "
             "unfinished ones); fix them in CVAT and export again, or accept them.",
             "Suggested = points the model proposed; corrected = heads after the annotator's corrections."]
    notes += [f"Image not in any batch: {u}" for u in unknown] + [f"Exported twice: {d}" for d in doubles]
    write_report(p.results / "india_import_check", rows, "India dataset: import check of the CVAT exports", notes)
    for c in checks:
        if c.status != "done" or c.flags:
            print(f"  {c.crop}: {c.status}{' - ' + '; '.join(c.flags) if c.flags else ''}")
    print(f"{dict(status)}; {sum(1 for c in checks if c.flags)} flagged -> {p.results / 'india_import_check.md'}")
    unfinished = status["missing from exports"] + status["not marked done"]
    if unfinished and not args.allow_incomplete:
        raise SystemExit(f"{unfinished} crops are not finished; finish them in CVAT and export again "
                         "(or --allow-incomplete to build the dataset without them)")

    if p.annotations.exists():
        shutil.rmtree(p.annotations)  # rebuilt from the exports every time, so no stale crop survives
    heads = 0
    for c in checks:
        if c.status == "done":
            ann = found[c.crop][0]
            ann.video = c.video
            save_points_json(p.annotations / c.video / f"{Path(c.crop).stem}.json", ann)
            heads += len(ann.points)
    print(f"imported {status['done']} crops, {heads} heads -> {p.annotations}")
    if not args.no_build:
        cmd_build(args, p)


def cmd_build(args, p: Paths) -> None:
    from ml.prepare_dataset import main as prepare_main

    report = check_intake(p.root)
    location = {v.video_id: v.location for v in report.videos}
    crops = Counter(f.parent.name for f in p.annotations.glob("*/*.json"))
    if not crops:
        raise SystemExit(f"no imported annotations in {p.annotations} (run the import step)")
    unknown = sorted(set(crops) - set(location))
    if unknown:
        raise SystemExit(f"annotations for videos not in raw_videos/<location>/: {unknown}")
    old = {r["video"]: r for r in read_csv(p.split)}
    test_v, val_v = [s.strip() for s in args.test_videos.split(",") if s.strip()], \
        [s.strip() for s in args.val_videos.split(",") if s.strip()]
    if test_v or val_v:
        split = {v: "test" if v in test_v else "val" if v in val_v else "train" for v in crops}
        how = "named on the command line"
    elif old and not args.resplit:
        test_locations = {r["location"] for r in old.values() if r["split"] == "test"}
        split = {}
        for v in crops:
            split[v] = old[v]["split"] if v in old else ("test" if location[v] in test_locations else "train")
            if v not in old:
                print(f"new video {v} ({location[v]}) -> {split[v]}")
        how = f"kept from {p.split}"
    else:
        split = split_by_location({v: (location[v], n) for v, n in crops.items()}, args.test_share,
                                  args.val_share, args.min_test_locations, args.seed)
        how = f"chosen by location (seed {args.seed})"
    write_csv(p.split, [{"video": v, "location": location[v], "split": split[v], "crops": crops[v]}
                        for v in sorted(crops)])
    total = sum(crops.values())
    by_split = {s: [v for v in sorted(crops) if split[v] == s] for s in ("train", "val", "test")}
    train_locations = {location[v] for v in by_split["train"]}
    rows = [{"split": s, "videos": len(vs), "crops": sum(crops[v] for v in vs),
             "share_pct": round(100 * sum(crops[v] for v in vs) / total, 1),
             "locations": " ".join(sorted({location[v] for v in vs})), "video_ids": " ".join(vs)}
            for s, vs in by_split.items()]
    unseen = sorted({location[v] for v in by_split["test"]} - train_locations)
    write_report(p.results / "india_split", rows, "India dataset: train / validation / test split by video", [
        f"Split {how}. Test locations never used in training: {', '.join(unseen) or 'none'}.",
        "Keep this split for every model compared; do not move videos after seeing test results.",
    ])
    for r in rows:
        print(f"{r['split']:5}: {r['videos']} videos, {r['crops']} crops ({r['share_pct']}%), {r['locations']}")
    if not by_split["test"] or not by_split["val"] or not by_split["train"]:
        raise SystemExit("train, val and test each need at least one video")
    if len(unseen) < 2 and not (test_v or val_v):
        print("warning: fewer than 2 test locations are unseen in training")
    prepare_main(["india", "--root", str(p.crops), "--annotations", str(p.annotations), "--out", str(args.out),
                  "--test-videos", ",".join(by_split["test"]), "--val-videos", ",".join(by_split["val"]),
                  "--workers", "1"])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DATA_DIR / "india_dataset")
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results")
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("check", help="1. raw_videos vs sources.csv; duration and resolution")
    for name, text in (("frames", "2. distinct, usable frames per video"), ("crops", "3. dense crops + contact sheets"),
                       ("preannotate", "4. suggested head points")):
        s = sub.add_parser(name, help=text)
        s.add_argument("--video", action="append", help="only this video id (repeatable)")
        s.add_argument("--force", action="store_true", help="redo videos already done")
        s.add_argument("--weights", help="density model weights (default: CSRNET_WEIGHTS in .env)")
        s.add_argument("--device", help="default: DEVICE in .env")
        if name == "frames":
            s.add_argument("--per-video", type=int, default=10)
        if name == "crops":
            s.add_argument("--size", type=int, default=512)
            s.add_argument("--per-frame", type=int, default=2)
    s = sub.add_parser("batches", help="5. one CVAT task package per annotator")
    s.add_argument("--annotators", required=True, help='comma-separated names, e.g. "Asha,Ravi,Meera,Kiran"')
    s.add_argument("--force", action="store_true", help="deal again (only before anyone has started)")
    for name, text in (("import", "6. import the CVAT exports, check them, build the dataset"),
                       ("build", "split by video/location and build density maps from the imported points")):
        s = sub.add_parser(name, help=text)
        if name == "import":
            s.add_argument("exports", nargs="*", help="export .zip/.xml files (default: all in cvat_exports/)")
            s.add_argument("--allow-incomplete", action="store_true", help="build without unfinished crops")
            s.add_argument("--no-build", action="store_true", help="only import and check")
        s.add_argument("--out", type=Path, default=DATA_DIR / "processed" / "india")
        s.add_argument("--test-videos", default="", help="name the test videos yourself (comma-separated)")
        s.add_argument("--val-videos", default="", help="name the validation videos yourself")
        s.add_argument("--resplit", action="store_true", help="choose the split again (only before any test result)")
        s.add_argument("--test-share", type=float, default=0.15)
        s.add_argument("--val-share", type=float, default=0.15)
        s.add_argument("--min-test-locations", type=int, default=2)
        s.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    p = Paths(args.root, args.results)
    {"check": cmd_check, "frames": cmd_frames, "crops": cmd_crops, "preannotate": cmd_preannotate,
     "batches": cmd_batches, "import": cmd_import, "build": cmd_build}[args.step](args, p)


if __name__ == "__main__":
    main()
