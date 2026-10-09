"""Smoke tests: a couple of tiny training steps and an evaluation run on synthetic data."""

import numpy as np
import pytest
import torch

from app.csrnet_model import describe_weights
from ml.dataset_index import read_index
from ml.datasets import CrowdDataset, random_crop
from ml.evaluate import main as evaluate_main
from ml.train_csrnet import main as train_main
from tests.test_prepare_dataset import make_india_dataset


@pytest.fixture
def processed(tmp_path, rng):
    return make_india_dataset(tmp_path, rng)  # 3 videos x 2 frames of 160x90 (-> 160x88)


def test_random_crop_is_grid_aligned_and_padded():
    torch.manual_seed(0)
    image = np.zeros((88, 160, 3), np.uint8)
    density = np.ones((11, 20), np.float32)
    crop_img, crop_den = random_crop(image, density, 64)
    assert crop_img.shape == (64, 64, 3) and crop_den.shape == (8, 8) and crop_den.sum() == 64
    small_img, small_den = random_crop(image, density, 96)  # taller than the image: padded
    assert small_img.shape == (96, 96, 3) and small_den.shape == (12, 12) and small_den.sum() == 11 * 12


def test_crop_and_flip_keep_the_density_target_aligned():
    """The 1/8-resolution target must be exactly the density of the cropped (and flipped) pixels.

    Each pixel's colour encodes its own column and row, so the returned crop reveals where it
    was taken from and whether it was mirrored.
    """
    from ml.density_maps import density_map, sum_pool

    h, w, crop = 96, 160, 64
    rows, cols = np.mgrid[0:h, 0:w]
    image = np.dstack([cols, rows, np.zeros_like(rows)]).astype(np.uint8)
    full = density_map(np.random.default_rng(3).uniform([0, 0], [w, h], (40, 2)), (h, w), sigma=2.0)
    dataset = CrowdDataset([], train=True, crop_size=crop, color_jitter=False, grayscale_p=0)

    flipped_runs = 0
    for seed in range(20):
        torch.manual_seed(seed)
        crop_img, crop_den = dataset._augment(image, sum_pool(full, 8))
        flipped = crop_img[0, 1, 0] < crop_img[0, 0, 0]
        x0 = int(crop_img[0, 0, 0]) - (crop - 1 if flipped else 0)
        y0 = int(crop_img[0, 0, 1])
        assert x0 % 8 == 0 and y0 % 8 == 0  # grid-aligned, so pooling stays exact
        region = full[y0 : y0 + crop, x0 : x0 + crop]
        expected = sum_pool(region, 8)[:, ::-1] if flipped else sum_pool(region, 8)
        np.testing.assert_allclose(crop_den, expected, rtol=1e-5, atol=1e-7)
        assert crop_den.sum() == pytest.approx(region.sum(), rel=1e-5)  # count preserved
        flipped_runs += flipped
    assert 0 < flipped_runs < 20  # both the flipped and unflipped branches were checked


def test_dataset_items(processed):
    torch.manual_seed(0)
    train = CrowdDataset(read_index(processed, ["train"]), train=True, crop_size=64)
    image, density, count = train[0]
    assert image.shape == (3, 64, 64) and density.shape == (1, 8, 8)
    assert count.item() == pytest.approx(density.sum().item(), rel=1e-5)

    test = CrowdDataset(read_index(processed, ["test"]), train=False)
    image, density, count = test[0]
    assert image.shape == (3, 88, 160) and density.shape == (1, 11, 20) and count.item() == 25


def test_train_resume_and_evaluate(processed, tmp_path, capsys):
    out = tmp_path / "weights"
    common = ["--data", str(processed), "--out", str(out), "--name", "tiny", "--batch-size", "2",
              "--crop", "64", "--workers", "0", "--device", "cpu", "--no-pretrained", "--max-steps", "1"]
    train_main(common + ["--epochs", "1"])
    assert (out / "tiny_best.pth").exists() and (out / "tiny_last.pth").exists()
    best = torch.load(out / "tiny_best.pth", map_location="cpu", weights_only=True)
    prov = best["provenance"]  # where these weights came from travels inside the file
    assert (prov["source"], prov["dataset"], prov["epochs"], prov["test_mae"]) == ("trained_here", processed.name, 1, None)
    assert prov["val_mae"] == pytest.approx(best["val_mae"], abs=1e-3)
    assert describe_weights(best, out / "tiny_best.pth")["kind"] == "trained"
    train_main(common + ["--epochs", "2", "--resume"])
    log_lines = (out / "tiny_log.csv").read_text().strip().splitlines()
    assert [line.split(",")[0] for line in log_lines] == ["epoch", "1", "2"]
    assert log_lines[0] == "epoch,train_loss,val_loss,val_mae,val_rmse,lr,seconds"
    assert all(len(line.split(",")) == 7 for line in log_lines)

    custom_log = tmp_path / "logs" / "train_log.csv"  # --log puts the CSV anywhere (results\ for Phase 5)
    train_main(common + ["--epochs", "1", "--name", "other", "--log", str(custom_log)])
    assert custom_log.read_text().startswith("epoch,train_loss,val_loss")

    results = tmp_path / "results"
    evaluate_main(["--weights", str(out / "tiny_best.pth"), "--data", str(processed), "--device", "cpu",
                   "--out", str(results), "--vis-every", "2"])
    assert "MAE" in capsys.readouterr().out
    lines = (results / "results.csv").read_text().strip().splitlines()
    assert lines[0] == "image_name,gt_count,pred_count,abs_error" and len(lines) == 3  # 2 test frames
    assert len(list((results / "heatmaps").glob("*.png"))) == 1  # every 2nd image: just the first


def test_evaluate_raw_shanghaitech_folder(tmp_path, rng, capsys):
    """--images mode: counts from GT_IMG_*.mat beside images/, or from --dmaps .npy sums."""
    from app.csrnet_model import CSRNet
    from app.image_io import write_image
    from tests.test_annotations import write_shanghaitech_mat

    weights = tmp_path / "w.pth"
    torch.save({"model": CSRNet(pretrained_frontend=False).state_dict()}, weights)
    test_data = tmp_path / "part_A_final" / "test_data"
    (test_data / "ground-truth").mkdir(parents=True)
    dmaps = tmp_path / "dmaps"
    dmaps.mkdir()
    for i, heads in ((1, 12), (2, 30)):
        write_image(test_data / "images" / f"IMG_{i}.jpg", rng.integers(0, 255, (60, 90, 3), dtype=np.uint8))
        write_shanghaitech_mat(test_data / "ground-truth" / f"GT_IMG_{i}.mat", rng.uniform(0, 60, (heads, 2)))
        np.save(dmaps / f"IMG_{i}.npy", np.full((60, 90), 7.0 / (60 * 90)))

    common = ["--weights", str(weights), "--images", str(test_data / "images"), "--device", "cpu"]
    evaluate_main(common + ["--out", str(tmp_path / "mat")])
    evaluate_main(common + ["--dmaps", str(dmaps), "--out", str(tmp_path / "npy")])
    assert "RMSE" in capsys.readouterr().out
    mat_counts = [line.split(",")[1] for line in (tmp_path / "mat" / "results.csv").read_text().splitlines()[1:]]
    npy_counts = [line.split(",")[1] for line in (tmp_path / "npy" / "results.csv").read_text().splitlines()[1:]]
    assert mat_counts == ["12", "30"] and npy_counts == ["7", "7"]
