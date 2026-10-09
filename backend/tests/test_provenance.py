"""Phase 5A step 1: no fake numbers at runtime. Only weights with a trusted provenance record are
loaded; demo or provenance-less weights need DEMO_MODE=1, and their counts are never shown otherwise."""

import json
from dataclasses import replace

import pytest
import torch
from fastapi.testclient import TestClient

from app.config import get_settings
from app.csrnet_model import CSRNet
from app.main import create_app
from app.video_jobs import VideoService
from app.worker import build_services
from tests.fakes import make_services


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    """The same random CSRNet saved four ways: public (with provenance), trained here, demo, and bare."""
    folder = tmp_path_factory.mktemp("weights")
    torch.manual_seed(0)
    state = CSRNet(pretrained_frontend=False).state_dict()
    files = {
        "public": {"model": state, "provenance": {"source": "public_pretrained", "dataset": "ShanghaiTech Part A",
                                                    "origin": "test", "authors_reported_mae": 66.4,
                                                    "test_mae": 71.8, "test_rmse": 120.2}},
        "trained": {"model": state, "provenance": {"source": "trained_here", "dataset": "shanghaitech_a",
                                                     "val_mae": 61.0, "test_mae": None}},
        "demo": {"model": state, "demo": True, "provenance": {"source": "demo"}},
        "bare": {"model": state, "val_mae": 60.0},  # old-style checkpoint: no provenance record
    }
    paths = {}
    for name, checkpoint in files.items():
        paths[name] = folder / f"{name}.pth"
        torch.save(checkpoint, paths[name])
    return paths


def settings_for(tmp_path, weights_path, demo_mode=False):
    zones = tmp_path / "zones.json"
    zones.write_text(json.dumps({"zones": [{"id": "Z", "name": "Zone", "source": "none.mp4", "area_m2": 100}]}))
    return replace(get_settings(), zones_config=zones, csrnet_weights=weights_path, demo_mode=demo_mode,
                   device="cpu", frame_source="local", mongodb_uri="", upload_dir=tmp_path / "uploads")


@pytest.mark.parametrize("kind", ["public", "trained"])
def test_weights_with_a_trusted_provenance_are_loaded(tmp_path, weights, kind):
    services = build_services(settings_for(tmp_path, weights[kind]))
    assert services.worker is not None and services.model_error is None
    assert services.estimator.info["kind"] == {"public": "public_pretrained", "trained": "trained"}[kind]


@pytest.mark.parametrize("kind", ["demo", "bare"])
def test_demo_or_provenance_less_weights_are_refused_without_demo_mode(tmp_path, weights, kind):
    services = build_services(settings_for(tmp_path, weights[kind]))
    assert services.worker is None and services.estimator is None
    assert services.model_error.startswith("No trained model: counts unavailable")
    assert "DEMO_MODE=1" in services.model_error

    allowed = build_services(settings_for(tmp_path, weights[kind], demo_mode=True))
    assert allowed.worker is not None  # a demo is possible, but only when asked for explicitly


def test_health_says_model_not_loaded_and_names_the_active_weights(tmp_path, weights):
    for path, loaded in ((weights["demo"], False), (weights["public"], True)):
        settings = settings_for(tmp_path, path)
        app = create_app(settings, services_factory=build_services, run_loop=False)
        with TestClient(app) as client:
            health = client.get("/health").json()
        assert health["model_loaded"] is loaded
        assert health["model_status"] == ("loaded" if loaded else "model not loaded")
        if loaded:
            assert health["model"]["kind"] == "public_pretrained" and health["model"]["test_mae"] == 71.8
        else:
            assert health["model"] is None and "counts unavailable" in health["model_error"]


def test_no_trained_model_means_no_live_readings(tmp_path, weights):
    services = build_services(settings_for(tmp_path, weights["demo"]))
    app = create_app(services.settings, services_factory=lambda _: services, run_loop=False)
    with TestClient(app) as client:
        assert client.get("/api/zones/Z/status").json()["risk_level"] == "UNKNOWN"  # no count, density or level
        assert client.post("/api/zones/Z/infer").status_code == 503
        options = client.get("/api/video/options").json()
        assert not options["model_loaded"] and options["model_error"].startswith("No trained model")


def test_video_results_made_with_demo_weights_are_not_served_without_demo_mode(tmp_path):
    services = make_services(count=300)  # the fake estimator: no provenance, like a demo model
    video = VideoService(services.settings, services)
    video.start()
    clip = tmp_path / "c.avi"
    from tests.test_video import textured_video, wait_for

    textured_video(clip, seconds=1)
    meta = video.register_upload("v1", clip, "c.avi", 1)
    from app.video_jobs import VideoZoneRequest

    job = video.create_job(meta["id"], VideoZoneRequest("Z", 100.0))
    wait_for(lambda: job.state == "done")
    video.stop()

    for demo_mode, expected in ((True, 200), (False, 409)):
        settings = replace(services.settings, demo_mode=demo_mode)
        app = create_app(settings, services_factory=lambda _: replace(services, settings=settings), run_loop=False)
        with TestClient(app) as client:
            response = client.get(f"/api/jobs/{job.id}/results")
            assert response.status_code == expected
            if expected == 409:
                assert "counts are not real" in response.json()["detail"]
                assert client.get(f"/api/jobs/{job.id}/results.csv").status_code == 409
