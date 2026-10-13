"""India dataset pipeline: intake check, frame selection, crops, suggestions, batches, split, CVAT round trip."""

import csv
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from ml.annotation_batches import CropCheck, balanced_batches, point_flags, ratio_flags
from ml.annotations import load_points_json, parse_cvat_tags, parse_cvat_xml, read_cvat_export, write_cvat_xml
from ml.crops import contact_sheet, crop_size, densest_crops
from ml.dataset_index import read_index, split_by_location
from ml.frame_select import find_overlays, reject_unusable, scan_video, select_frames
from ml.india_sources import check_intake
from ml.preannotate import heads_from_boxes, points_from_density, suggest_points
from tests.conftest import write_video
from tests.fakes import FakeDetector, FakeEstimator
from tools.india_dataset import main as india_main


def scene(seed: int, h: int = 180, w: int = 320) -> np.ndarray:
    """A sharp, textured 'crowd' picture with its own dominant hue (seeds 0-5 differ in colour, so a
    switch between them is a scene cut); different seeds share no features."""
    rng = np.random.default_rng(seed)
    hsv = np.empty((h // 4, w // 4, 3), np.uint8)
    hsv[..., 0] = (seed * 30 + rng.integers(0, 12, hsv.shape[:2])) % 180
    hsv[..., 1] = rng.integers(120, 256, hsv.shape[:2])
    hsv[..., 2] = rng.integers(60, 256, hsv.shape[:2])
    return cv2.resize(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), (w, h), interpolation=cv2.INTER_NEAREST)


def write_sources(root, rows):
    with open(root / "sources.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["video_file", "location", "event", "year", "source_url", "camera_type"])
        writer.writeheader()
        writer.writerows(rows)


def row(rel, camera="fixed"):
    return {"video_file": rel, "location": rel.split("/")[0], "event": "Kumbh", "year": "2015",
            "source_url": "https://example.org/v", "camera_type": camera}


# 1. intake ----------------------------------------------------------------------------------

def test_check_intake_reports_both_directions_and_row_problems(tmp_path):
    raw = tmp_path / "raw_videos"
    for rel in ("nashik/ramkund_01.avi", "nashik/ramkund_02.avi", "pandharpur/wari_01.avi"):
        (raw / rel).parent.mkdir(parents=True, exist_ok=True)
        write_video(raw / rel, [scene(i) for i in range(5)], fps=10)
    write_video(raw / "demo_clip.avi", [scene(0)] * 3)  # a loose demo clip: never part of the dataset
    write_sources(tmp_path, [row("nashik/ramkund_01.avi"), {**row("nashik/ramkund_02.avi"), "camera_type": "phone"},
                             {**row("pandharpur/wari_01.avi"), "location": "pune"}, row("shirdi/missing_01.mp4")])
    report = check_intake(tmp_path)
    assert [v.video_id for v in report.videos] == ["ramkund_01", "ramkund_02", "wari_01"]
    assert [v.rel for v in report.demo_clips] == ["demo_clip.avi"]
    assert report.missing_files == ["shirdi/missing_01.mp4"] and report.not_in_csv == []
    assert any("camera_type 'phone'" in p for p in report.problems)
    assert any("does not match the folder" in p for p in report.problems)
    v = report.videos[0]
    assert (v.width, v.height, v.fps) == (320, 180, 10) and v.duration_s == pytest.approx(0.5)
    assert any("below 512 px" in w for w in report.warnings) and not report.ok


def test_check_intake_flags_unlisted_videos_and_duplicate_ids(tmp_path):
    raw = tmp_path / "raw_videos"
    for rel in ("a/clip.avi", "b/clip.avi"):
        (raw / rel).parent.mkdir(parents=True, exist_ok=True)
        write_video(raw / rel, [scene(0)] * 3)
    write_sources(tmp_path, [row("a/clip.avi")])
    report = check_intake(tmp_path)
    assert report.not_in_csv == ["b/clip.avi"]
    assert any("also used by a/clip.avi" in p for p in report.problems)


# 2. frame selection -------------------------------------------------------------------------

def make_clip(path, fps=10):
    """Four 4-s shots of different scenes, then a blurred shot, a title card and a crowd-less shot;
    a static white logo sits in the top-right corner throughout."""
    frames = []
    for shot in range(4):
        base = scene(shot)
        for k in range(40):  # slight camera drift inside a shot
            frames.append(np.roll(base, k // 10, axis=1))
    frames += [cv2.GaussianBlur(scene(4), (0, 0), 3)] * 40
    title = np.full((180, 320, 3), 30, np.uint8)
    cv2.putText(title, "TITLE", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    frames += [title] * 40
    frames += [scene(5)] * 40  # sharp, but the fake counter says no crowd
    for f in frames:
        f[6:22, 270:310] = 255
        cv2.putText(f, "TV", (276, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return write_video(path, frames, fps)


def test_scan_rejects_cuts_blur_titles_empty_and_finds_the_logo(tmp_path):
    clip = make_clip(tmp_path / "clip.avi")
    crowdless = scene(5)
    scan = scan_video(clip, count_fn=lambda f: 1.0 if np.abs(f[40:, :260].astype(int) - crowdless[40:, :260]).mean() < 10
                      else 100.0, every_s=1.0)
    assert scan.frames == 280 and len(scan.cuts_s) == 6  # cuts at 4, 8, ... 24 s
    reject_unusable(scan.candidates)
    reasons = {int(c.time_s): c.reason for c in scan.candidates}  # candidates at 0.5, 1.5, ... s
    assert reasons[3] == reasons[4] == "scene cut" and reasons[1] == reasons[2] == ""
    assert all(reasons[t] in ("blurry", "scene cut") for t in range(16, 20))
    assert all(reasons[t] in ("title card / blank", "scene cut") for t in range(20, 24))
    assert all(reasons[t] in ("little or no crowd", "scene cut") for t in range(24, 28))
    assert len(scan.overlays) == 1
    x1, y1, x2, y2 = scan.overlays[0]
    assert x1 <= 270 and y1 <= 6 and x2 >= 310 and y2 >= 22 and x2 - x1 < 80


def test_select_frames_one_per_scene_without_near_duplicates(tmp_path):
    scan = scan_video(make_clip(tmp_path / "clip.avi"), count_fn=lambda f: 100.0, every_s=1.0)
    reject_unusable(scan.candidates)
    picks = select_frames(scan.candidates, n=10, frame_area=640 * 360)
    shots = [int(c.time_s // 4) for c in picks]
    assert shots == [0, 1, 2, 3, 6]  # one frame per usable scene (the crowd-less scene counts 100 here)
    assert all(not c.reason for c in picks)
    others = [c for c in scan.candidates if c.reason.startswith("near-duplicate")]
    assert others, "frames of the same shot must be recognised as the same people"


def test_find_overlays_ignores_static_scenery_when_nothing_moves():
    still = [scene(1)] * 20
    greys = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in still]
    assert find_overlays(greys, 320, 180) == []


# 3. crops ------------------------------------------------------------------------------------

def test_densest_crops_picks_two_dense_areas_without_overlap_and_avoids_logos():
    density = np.zeros((90, 160), np.float32)  # a 1280x720 frame at 1/8
    density[10:20, 10:20] = 1.0  # 100 people top-left
    density[60:70, 120:130] = 0.6  # 60 people bottom-right
    crops = densest_crops(density, 512, max_crops=2)
    assert len(crops) == 2
    a, b = crops
    assert a.predicted == pytest.approx(100) and b.predicted == pytest.approx(60)
    assert a.x <= 80 and a.x + 512 >= 160 and b.x <= 960 and b.x + 512 >= 1040
    assert abs(a.x - b.x) >= 512 or abs(a.y - b.y) >= 512
    for c in crops:
        assert 0 <= c.x <= 1280 - 512 and 0 <= c.y <= 720 - 512 and c.x % 8 == 0 and c.y % 8 == 0
    logo = [(0.0, 0.0, 300.0, 300.0)]  # covers the densest area: the crop must move off it
    c = densest_crops(density, 512, max_crops=1, exclude=logo)[0]
    assert c.x >= 300 or c.y >= 300


def test_second_crop_only_when_crowded_enough():
    density = np.zeros((90, 160), np.float32)
    density[10:20, 10:20] = 1.0
    density[80, 150] = 5.0  # a few people far away: below 30% of the first crop
    assert len(densest_crops(density, 512, max_crops=2)) == 1
    assert crop_size(640, 360) == 360 and crop_size(1920, 1080) == 512 and crop_size(500, 300) == 296


def test_contact_sheet_grid():
    sheet = contact_sheet([(np.zeros((360, 360, 3), np.uint8), f"c{i}") for i in range(7)], columns=5, tile=100)
    assert sheet.shape == (2 * 104, 5 * 104, 3)


# 4. suggestions ------------------------------------------------------------------------------

def test_points_from_density_matches_count_and_location():
    density = np.zeros((64, 64), np.float32)
    density[10:14, 10:14] = 30 / 16  # 30 people top-left
    density[50:52, 40:42] = 5 / 4  # 5 people lower right
    points = points_from_density(density, (512, 512))
    assert len(points) == 35
    top_left = ((points[:, 0] < 160) & (points[:, 1] < 160)).sum()
    assert top_left == pytest.approx(30, abs=2)
    assert len(points_from_density(np.zeros((8, 8)), (64, 64))) == 0


def test_heads_from_boxes():
    heads = heads_from_boxes(np.array([[100, 50, 140, 170], [0, 0, 100, 40]]))
    np.testing.assert_allclose(heads, [[120, 58], [50, 20]])  # 0.2 x width below the top; capped at mid-height


def test_suggest_points_combines_detector_and_density():
    crop = np.zeros((512, 512, 3), np.uint8)
    boxes = [[0, 0, 512, 256]]  # one box over the top half: its density is replaced by the detector's head
    sug = suggest_points(crop, FakeEstimator(count=40), FakeDetector(boxes=boxes))
    assert sug.method == "detector+density" and sug.detector_count == 1
    assert len(sug.points) == 1 + 20  # the uniform density outside the box holds half of the 40
    assert (sug.points[1:, 1] >= 256).all()
    only_density = suggest_points(crop, FakeEstimator(count=12), None)
    assert only_density.method == "density" and len(only_density.points) == 12


# 5-6. batches, checks, split, CVAT ------------------------------------------------------------

def test_balanced_batches_keep_videos_whole():
    videos = {f"v{i}": (10 + i, 100.0 * (i % 3 + 1)) for i in range(9)}
    batches = balanced_batches(videos, 4)
    assert sorted(v for b in batches for v in b) == sorted(videos)
    work = [sum(videos[v][0] for v in b) for b in batches]
    assert max(work) - min(work) <= 15


def test_split_by_location_holds_out_whole_locations():
    videos = {f"{loc}_{k}": (loc, 20) for loc in ("nashik", "ujjain", "prayagraj", "haridwar", "pandharpur", "shirdi",
                                                  "puri") for k in range(2)}
    split = split_by_location(videos, test_share=0.15, val_share=0.15, min_test_locations=2, seed=1)
    test_locations = {videos[v][0] for v, s in split.items() if s == "test"}
    train_locations = {videos[v][0] for v, s in split.items() if s == "train"}
    assert len(test_locations) >= 2 and not test_locations & train_locations
    assert all(videos[v][0] in test_locations for v, s in split.items() if s == "test")
    shares = {s: sum(1 for x in split.values() if x == s) / len(split) for s in ("train", "val", "test")}
    assert 0.1 <= shares["test"] <= 0.35 and 0.05 <= shares["val"] <= 0.25 and shares["train"] >= 0.5
    assert split == split_by_location(videos, seed=1)  # reproducible
    with pytest.raises(ValueError, match="at least 3"):
        split_by_location({"a": ("x", 5), "b": ("y", 5)})


def test_point_and_ratio_flags():
    suggested = np.array([[10.0 * i, 10.0] for i in range(12)])
    assert "kept exactly" in point_flags(suggested, 200, 200, suggested)[0]
    flags = point_flags(np.array([[5.0, 5.0], [5.5, 5.0], [250.0, 5.0]]), 200, 200, suggested)
    assert any("outside" in f for f in flags) and any("double clicks" in f for f in flags)
    assert any("deleted" in f for f in point_flags(np.zeros((0, 2)), 200, 200, suggested))
    checks = [CropCheck(f"c{i}", "v", "b1", "done", 100, 200) for i in range(5)]  # the model undercounts 2x
    checks.append(CropCheck("odd", "v", "b1", "done", 100, 20))
    ratio_flags(checks)
    assert [c.crop for c in checks if c.flags] == ["odd"]


def test_cvat_xml_roundtrip_with_tags(tmp_path):
    xml = tmp_path / "annotations.xml"
    write_cvat_xml(xml, [("b.jpg", 64, 64, np.array([[1.0, 2.0], [3.5, 4.25]])), ("a.jpg", 64, 64, np.zeros((0, 2)))],
                   {"head": "points", "done": "tag"}, "head")
    anns = {a.image: a for a in parse_cvat_xml(xml, label="head")}
    np.testing.assert_allclose(anns["b.jpg"].points, [[1, 2], [3.5, 4.25]])
    assert len(anns["a.jpg"].points) == 0
    text = xml.read_text(encoding="utf-8").replace('<image id="0" name="a.jpg" width="64" height="64" />',
                                                   '<image id="0" name="a.jpg" width="64" height="64">'
                                                   '<tag label="done" source="manual" /></image>')
    exported = tmp_path / "task_export.zip"
    with zipfile.ZipFile(exported, "w") as z:  # CVAT's download: annotations.xml (+ images/)
        z.writestr("annotations.xml", text)
    assert parse_cvat_tags(read_cvat_export(exported)) == {"a.jpg": {"done"}, "b.jpg": set()}


def make_dataset(root):
    """6 locations x 2 videos, each 4 scenes of 3 s (640x360 at 10 fps), listed in sources.csv."""
    raw = root / "raw_videos"
    rows = []
    for n, loc in enumerate(("nashik", "ujjain", "prayagraj", "haridwar", "pandharpur", "shirdi")):
        for k in (1, 2):
            rel = f"{loc}/{loc}_{k}.avi"
            (raw / rel).parent.mkdir(parents=True, exist_ok=True)
            scenes = [scene(6 * (2 * n + k) + j, 360, 640) for j in range(4)]
            write_video(raw / rel, [scenes[i // 30] for i in range(120)], fps=10)
            rows.append(row(rel))
    write_sources(root, rows)


def test_pipeline_end_to_end_with_fake_models(tmp_path, monkeypatch, capsys):
    root, results, processed = tmp_path / "india", tmp_path / "results", tmp_path / "processed"
    make_dataset(root)
    monkeypatch.setattr("tools.india_dataset.load_estimator", lambda args: FakeEstimator(count=60))
    monkeypatch.setattr("app.detector.load_detector", lambda *a, **k: (FakeDetector(count=3), None))
    run = lambda *argv: india_main(["--root", str(root), "--results", str(results), *argv])
    run("check")
    run("frames", "--per-video", "4")
    run("crops")
    run("preannotate")
    assert len(list((root / "frames" / "nashik_1").glob("*.jpg"))) == 4  # one per scene
    rejected = next(csv.DictReader(open(root / "crops" / "nashik_1" / "crops.csv", encoding="utf-8")))["crop"]
    with open(root / "crops" / "nashik_1" / "rejected.txt", "a", encoding="utf-8") as f:
        f.write(rejected + "\n")
    run("batches", "--annotators", "A,B,C,D")
    with pytest.raises(SystemExit, match="already dealt"):
        run("batches", "--annotators", "A,B,C,D")

    dealt = list(csv.DictReader(open(root / "cvat" / "batch_crops.csv", encoding="utf-8")))
    assert {r["annotator"] for r in dealt} == {"A", "B", "C", "D"}
    by_video = {}
    for r in dealt:
        by_video.setdefault(r["video"], set()).add(r["annotator"])
    assert all(len(a) == 1 for a in by_video.values())  # no video split between annotators
    assert len(by_video) == 12 and len(dealt) == 12 * 4 - 1 and rejected not in {r["crop"] for r in dealt}
    package = root / "cvat" / "batch_1"
    assert (package / "images.zip").exists() and (package / "preannotations.zip").exists()

    # The annotators' CVAT exports: suggestions + 2 heads each, tagged done; one crop unfinished at first.
    exports = root / "cvat_exports"
    unfinished = dealt[0]["crop"]
    for batch in sorted({r["batch"] for r in dealt}):
        xml = (root / "cvat" / batch / "annotations.xml").read_text(encoding="utf-8")
        for r in (r for r in dealt if r["batch"] == batch):
            tag = "" if r["crop"] == unfinished else '<tag label="done" source="manual" />'
            extra = '<points label="head" source="manual" occluded="0" points="3,3" z_order="0" />' * 2
            xml = xml.replace(f'name="{r["crop"]}" width="360" height="360">',
                              f'name="{r["crop"]}" width="360" height="360">{tag}{extra}')
        exports.mkdir(exist_ok=True)
        with zipfile.ZipFile(exports / f"{batch}.zip", "w") as z:
            z.writestr("annotations.xml", xml)
    with pytest.raises(SystemExit, match="not finished"):
        run("import", "--out", str(processed))
    assert "not marked done" in (results / "india_import_check.md").read_text(encoding="utf-8")

    for batch in sorted({r["batch"] for r in dealt}):
        with zipfile.ZipFile(exports / f"{batch}.zip") as z:
            xml = z.read("annotations.xml").decode()
        with zipfile.ZipFile(exports / f"{batch}.zip", "w") as z:
            z.writestr("annotations.xml", xml.replace(f'name="{unfinished}" width="360" height="360">',
                                                      f'name="{unfinished}" width="360" height="360">'
                                                      '<tag label="bad_crop" source="manual" />'))
    run("import", "--out", str(processed))
    assert "imported 46 crops" in capsys.readouterr().out
    samples = read_index(processed)
    assert len(samples) == 46 and {s.split for s in samples} == {"train", "val", "test"}
    split = {r["video"]: r for r in csv.DictReader(open(root / "split.csv", encoding="utf-8"))}
    test_locations = {r["location"] for r in split.values() if r["split"] == "test"}
    train_locations = {r["location"] for r in split.values() if r["split"] == "train"}
    assert len(test_locations) >= 2 and not test_locations & train_locations
    first = next(iter((root / "crop_annotations").glob("*/*.json")))
    ann = load_points_json(first)
    sug = load_points_json(root / "suggestions" / first.parent.name / first.name)
    assert len(ann.points) == len(sug.points) + 2
    assert not (root / "crop_annotations" / dealt[0]["video"] / f"{unfinished[:-4]}.json").exists()
    for video, r in split.items():  # the processed dataset follows split.csv
        assert {s.split for s in samples if s.video == video} == {r["split"]}


def test_india_compare_reports_per_location(tmp_path, monkeypatch):
    import importlib.util

    scripts = Path(__file__).parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))  # as when run as a script: report_utils sits beside it
    spec = importlib.util.spec_from_file_location("india_compare", scripts / "india_compare.py")
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    rows = {"a_1_c1": (100, 60, 90), "a_2_c1": (50, 30, 45), "b_1_c1": (200, 150, 210)}  # gt, before, after
    for name, col in (("before", 1), ("after", 2)):
        with open(tmp_path / f"{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_name", "gt_count", "pred_count", "abs_error"])
            for image, r in rows.items():
                w.writerow([image, r[0], r[col], abs(r[col] - r[0])])
    with open(tmp_path / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "image", "density", "count", "video", "width", "height"])
        for image, video in (("a_1_c1", "a_1"), ("a_2_c1", "a_2"), ("b_1_c1", "b_1")):
            w.writerow([image, "test", "x", "y", 1, video, 8, 8])
    with open(tmp_path / "split.csv", "w", newline="") as f:
        f.write("video,location,split,crops\na_1,ghat,test,1\na_2,ghat,test,1\nb_1,temple,test,1\n")
    compare.main([str(tmp_path / "before.csv"), str(tmp_path / "after.csv"), "--index", str(tmp_path / "index.csv"),
                  "--split", str(tmp_path / "split.csv"), "--out", str(tmp_path / "cmp")])
    table = {r["test_set"]: r for r in csv.DictReader(open(tmp_path / "cmp.csv"))}
    assert list(table)[:3] == ["all test videos", "location ghat", "location temple"]
    assert float(table["location ghat"]["mae_before"]) == 30 and float(table["location ghat"]["mae_after"]) == 7.5
    assert float(table["location temple"]["mae_after"]) == 10
