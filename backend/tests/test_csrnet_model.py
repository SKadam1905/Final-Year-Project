import numpy as np
import pytest
import torch

from app.csrnet_model import (
    CSRNet,
    DensityEstimator,
    extract_state_dict,
    load_weights,
    stride_aligned_size,
)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return CSRNet(pretrained_frontend=False).eval()


def test_output_is_one_eighth_resolution(model):
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 64, 96)).shape == (2, 1, 8, 12)


def test_parameter_count_matches_paper(model):
    assert sum(p.numel() for p in model.parameters()) == 16_263_489  # "16.26M" in Li et al. 2018


def test_layer_names_match_authors_checkpoints(model):
    keys = set(model.state_dict())
    for key in ("frontend.0.weight", "frontend.21.weight", "backend.0.weight", "backend.10.weight",
                "output_layer.weight"):
        assert key in keys


def test_extract_state_dict_handles_wrappers(model):
    state = model.state_dict()
    data_parallel = {f"module.{k}": v for k, v in state.items()}
    for checkpoint in ({"model": state}, {"state_dict": data_parallel, "epoch": 3}, state):
        assert extract_state_dict(checkpoint).keys() == state.keys()


def test_save_and_load_roundtrip(tmp_path, model):
    path = tmp_path / "w.pth"
    torch.save({"model": model.state_dict(), "epoch": 1, "val_mae": 70.0}, path)
    other = CSRNet(pretrained_frontend=False)
    assert load_weights(other, path)["epoch"] == 1
    for a, b in zip(model.state_dict().values(), other.state_dict().values()):
        assert torch.equal(a, b)


def test_frontend_freezing(model):
    model.set_frontend_trainable(False)
    assert not any(p.requires_grad for p in model.frontend.parameters())
    assert all(p.requires_grad for p in model.backend.parameters())
    model.set_frontend_trainable(True)


@pytest.mark.parametrize("size, max_width, expected", [((150, 100), 1280, (152, 96)), ((2000, 1000), 1280, (1280, 640)),
                                                        ((1920, 1080), None, (1920, 1080)), ((5, 5), None, (8, 8))])
def test_stride_aligned_size(size, max_width, expected):
    assert stride_aligned_size(*size, max_width) == expected


def test_estimator_predicts_on_cpu(tmp_path, model):
    path = tmp_path / "w.pth"
    torch.save({"model": model.state_dict()}, path)
    estimator = DensityEstimator(path, device="cpu", max_width=128)
    frame = np.random.default_rng(0).integers(0, 255, (120, 200, 3), dtype=np.uint8)
    result = estimator.predict(frame)
    assert result.input_size == (128, 80) and result.frame_size == (200, 120)
    assert result.density.shape == (10, 16)
    assert result.count == pytest.approx(float(result.density.sum()), rel=1e-5)


def test_weights_describe_their_provenance():
    from app.csrnet_model import describe_weights

    public = {"provenance": {"source": "public_pretrained", "dataset": "ShanghaiTech Part A", "origin": "authors",
                             "authors_reported_mae": 66.4, "test_mae": None, "test_rmse": None}}
    info = describe_weights(public, "csrnet_partA_public.pth")
    assert info["kind"] == "public_pretrained" and info["dataset"] == "ShanghaiTech Part A"
    assert "not yet evaluated on this machine" in info["label"] and info["test_mae"] is None
    public["provenance"].update(test_mae=70.1, test_rmse=110.2)
    assert "test MAE 70.1 / RMSE 110.2 measured here" in describe_weights(public, "x.pth")["label"]

    ours = {"provenance": {"source": "trained_here", "dataset": "shanghaitech_a", "val_mae": 61.2, "test_mae": 69.0}}
    assert describe_weights(ours, "parta_best.pth")["kind"] == "trained"
    assert describe_weights({"demo": True}, "d.pth")["kind"] == "demo"
    assert describe_weights({"model": {}}, "random.pth")["kind"] == "untrained"  # no provenance: not trusted
