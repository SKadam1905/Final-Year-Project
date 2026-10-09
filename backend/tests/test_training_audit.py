"""Phase 5A step 4: one test per point of the training-code audit (CPU only).

Crop alignment and count preservation are covered in test_training.py
(test_random_crop_is_grid_aligned_and_padded, test_crop_and_flip_keep_the_density_target_aligned)
and the 1/8 sum-pooling in test_density_maps.py.
"""

import inspect
import re

import cv2
import numpy as np
import pytest
import torch

from ml import train_csrnet
from ml.dataset_index import Sample, read_index
from ml.datasets import CrowdDataset
from ml.train_csrnet import main as train_main
from tests.test_training import processed  # noqa: F401  (fixture: a tiny processed dataset)

TINY = ["--batch-size", "2", "--crop", "64", "--workers", "0", "--device", "cpu", "--no-pretrained"]


def test_validation_gives_whole_images_and_their_annotated_count(processed):  # noqa: F811
    s = read_index(processed, ["val"])[0]
    val = CrowdDataset([s], train=False)
    image, density, count = val[0]
    assert image.shape == (3, s.height, s.width) and density.shape == (1, s.height // 8, s.width // 8)
    assert float(count) == pytest.approx(s.count)  # the annotation, not a crop's share
    again = val[0]
    assert torch.equal(image, again[0]) and torch.equal(density, again[1])  # no random augmentation

    train = CrowdDataset(read_index(processed, ["train"]), train=True, crop_size=64)
    crop, crop_density, crop_count = train[0]
    assert crop.shape == (3, 64, 64) and float(crop_count) == pytest.approx(float(crop_density.sum()), rel=1e-5)


def test_greyscale_images_become_three_channels(tmp_path):
    grey = np.random.default_rng(0).integers(0, 255, (48, 64), np.uint8)
    cv2.imwrite(str(tmp_path / "grey.jpg"), grey)  # a one-channel JPEG, as some Part A images are
    np.save(tmp_path / "grey.npy", np.zeros((6, 8), np.float32))
    sample = Sample("g", "val", tmp_path / "grey.jpg", tmp_path / "grey.npy", 0.0, "", 64, 48)
    image, _, _ = CrowdDataset([sample], train=False)[0]
    assert image.shape == (3, 48, 64)


def test_training_code_uses_current_apis_and_separate_datasets():
    source = inspect.getsource(train_csrnet)
    assert "torch.cuda.amp" not in source and "torch.amp.GradScaler" in source and "torch.autocast" in source
    plateau = re.search(r"ReduceLROnPlateau\((.*?)\)", source).group(1)
    assert "verbose" not in plateau
    assert "random_split" not in source  # it would give train and validation the same (augmenting) transform
    assert re.search(r"DataLoader\(CrowdDataset\(train_samples, train=True", source)
    assert re.search(r"DataLoader\(CrowdDataset\(val_samples, train=False\), batch_size=1", source)


def test_the_same_seed_gives_the_same_training(processed, tmp_path):  # noqa: F811
    def run(seed, name):
        out = tmp_path / name
        train_main(["--data", str(processed), "--out", str(out), "--name", "s", "--epochs", "1", "--max-steps", "2",
                    "--seed", str(seed), *TINY])
        return torch.load(out / "s_best.pth", map_location="cpu", weights_only=True)["model"]

    first, second, other = run(42, "a"), run(42, "b"), run(7, "c")
    assert all(torch.allclose(first[k], second[k], atol=1e-6) for k in first)
    assert not all(torch.allclose(first[k], other[k], atol=1e-6) for k in first)


def test_early_stop_marks_training_finished_and_saves_atomically(processed, tmp_path, monkeypatch, capsys):  # noqa: F811
    maes = iter([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])  # the best is epoch 1, then it only gets worse

    def fake_evaluate(model, loader, device, amp):
        mae = next(maes)
        return {"mae": mae, "rmse": mae, "loss": mae, "preds": [], "gts": []}

    monkeypatch.setattr(train_csrnet, "evaluate_counts", fake_evaluate)
    out, log = tmp_path / "w", tmp_path / "log.csv"
    args = ["--data", str(processed), "--out", str(out), "--name", "e", "--epochs", "10", "--max-steps", "1",
            "--early-stop", "2", "--log", str(log), *TINY]
    train_main(args)
    assert [line.split(",")[0] for line in log.read_text().splitlines()[1:]] == ["1", "2", "3"]
    last = torch.load(out / "e_last.pth", map_location="cpu", weights_only=True)
    assert last["finished"].startswith("early stop at epoch 3") and last["best_epoch"] == 1
    assert not list(out.glob("*.tmp"))  # every checkpoint was written to a temp file and renamed

    train_main(args + ["--resume"])  # the loop resumes after any exit: a finished run must stay finished
    assert "training already finished" in capsys.readouterr().out
    assert len(log.read_text().splitlines()) == 4
