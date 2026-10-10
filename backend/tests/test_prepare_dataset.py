import numpy as np
import pytest

from app.image_io import write_image
from ml.annotations import PointAnnotation, save_points_json
from ml.dataset_index import choose_val_ids, read_index, split_videos
from ml.prepare_dataset import main as prepare_main
from tests.test_annotations import write_shanghaitech_mat


def test_choose_val_ids_is_reproducible():
    ids = [f"img{i}" for i in range(30)]
    assert choose_val_ids(ids, 0.1, seed=1) == choose_val_ids(ids, 0.1, seed=1)
    assert len(choose_val_ids(ids, 0.1)) == 3
    assert choose_val_ids(ids, 0.0) == set()


def test_split_videos_explicit_and_fractional():
    videos = ["a", "b", "c", "d"]
    assert split_videos(videos, test_videos=["d"], val_videos=["c"]) == {"a": "train", "b": "train", "c": "val", "d": "test"}
    assignment = split_videos(videos, test_fraction=0.25, val_fraction=0.25, seed=3)
    assert sorted(assignment.values()) == ["test", "train", "train", "val"]


@pytest.mark.parametrize("kwargs", [{"test_videos": ["zzz"]}, {"test_videos": ["a"], "val_videos": ["a"]},
                                    {"test_videos": ["a", "b"]}])
def test_split_videos_rejects_bad_requests(kwargs):
    with pytest.raises(ValueError):
        split_videos(["a", "b"], **kwargs)


def make_image(rng, w, h):
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_prepare_shanghaitech_layout(tmp_path, rng):
    base = tmp_path / "raw" / "part_A_final"
    sizes = {"train": [(203, 157), (180, 120), (96, 64)], "test": [(150, 100)]}
    for split, dims in sizes.items():
        for i, (w, h) in enumerate(dims, 1):
            write_image(base / f"{split}_data" / "images" / f"IMG_{i}.jpg", make_image(rng, w, h))
            (base / f"{split}_data" / "ground-truth").mkdir(parents=True, exist_ok=True)
            points = rng.uniform([0, 0], [w, h], (10 * i, 2))
            write_shanghaitech_mat(base / f"{split}_data" / "ground-truth" / f"GT_IMG_{i}.mat", points)

    out = tmp_path / "processed"
    prepare_main(["shanghaitech", "--root", str(tmp_path / "raw"), "--out", str(out),
                  "--val-fraction", "0.34", "--workers", "1"])

    samples = read_index(out)
    assert sorted(s.split for s in samples) == ["test", "train", "train", "val"]
    # The validation image is taken from the training images; the test images stay untouched.
    assert all(s.id.startswith("train_") for s in samples if s.split == "val")
    assert [s.id for s in samples if s.split == "test"] == ["test_IMG_1"]
    for s in samples:
        density = np.load(s.density)
        assert s.width % 8 == 0 and s.height % 8 == 0
        assert density.shape == (s.height // 8, s.width // 8)
        assert density.sum() == pytest.approx(s.count, abs=1e-2)
    assert {s.count for s in samples if s.split == "test"} == {10.0}


def make_india_dataset(root, rng):
    """3 videos x 2 annotated 160x90 frames (25 heads each), processed with a split by video."""
    frames, annotations = root / "frames", root / "annotations"
    for video in ("pandharpur_01", "pandharpur_02", "shirdi_01"):
        for i in (1, 2):
            name = f"frame_{i:04d}.jpg"
            write_image(frames / video / name, make_image(rng, 160, 90))
            points = rng.uniform([0, 0], [160, 90], (25, 2))
            save_points_json(annotations / video / f"frame_{i:04d}.json", PointAnnotation(name, video, 160, 90, points))
    write_image(frames / "shirdi_01" / "frame_0003.jpg", make_image(rng, 160, 90))  # not annotated: skipped

    out = root / "processed"
    prepare_main(["india", "--root", str(frames), "--out", str(out), "--test-videos", "shirdi_01",
                  "--val-videos", "pandharpur_02", "--workers", "1"])
    return out


def test_prepare_india_splits_by_video(tmp_path, rng):
    samples = read_index(make_india_dataset(tmp_path, rng))
    assert len(samples) == 6
    assert {s.video for s in samples if s.split == "test"} == {"shirdi_01"}
    assert {s.video for s in samples if s.split == "val"} == {"pandharpur_02"}
    assert {s.video for s in samples if s.split == "train"} == {"pandharpur_01"}
    assert all(s.count == 25 for s in samples)


def test_prepare_downscales_to_max_side(tmp_path, rng):
    frames, annotations = tmp_path / "frames", tmp_path / "annotations"
    for video in ("v1", "v2"):
        write_image(frames / video / "f.jpg", make_image(rng, 400, 300))
        save_points_json(annotations / video / "f.json",
                         PointAnnotation("f.jpg", video, 400, 300, np.array([[399.0, 299.0], [10.0, 10.0]])))
    out = tmp_path / "processed"
    prepare_main(["india", "--root", str(frames), "--out", str(out), "--test-videos", "v2",
                  "--val-fraction", "0", "--max-side", "200", "--workers", "1"])
    for s in read_index(out):
        assert (s.width, s.height) == (200, 152)
        assert np.load(s.density).sum() == pytest.approx(2.0, abs=1e-3)


def test_prepare_india_skips_roi_annotations(tmp_path, rng, capsys):
    frames, annotations = tmp_path / "frames", tmp_path / "annotations"
    for video in ("v1", "v2"):
        write_image(frames / video / "f.jpg", make_image(rng, 160, 90))
        save_points_json(annotations / video / "f.json", PointAnnotation("f.jpg", video, 160, 90, np.array([[5.0, 5.0]])))
    write_image(frames / "v1" / "g.jpg", make_image(rng, 160, 90))
    save_points_json(annotations / "v1" / "g.json",
                     PointAnnotation("g.jpg", "v1", 160, 90, np.array([[5.0, 5.0]]), roi=(0, 0, 80, 45)))
    out = tmp_path / "processed"
    prepare_main(["india", "--root", str(frames), "--out", str(out), "--test-videos", "v2",
                  "--val-fraction", "0", "--workers", "1"])
    assert sorted(s.id for s in read_index(out)) == ["v1_f", "v2_f"]
    assert "annotated inside an ROI only" in capsys.readouterr().out
