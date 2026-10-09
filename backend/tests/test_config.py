import json

import pytest

from app.config import REPO_ROOT, get_settings, load_zones


@pytest.fixture
def fresh_settings():
    get_settings.cache_clear()
    yield get_settings
    get_settings.cache_clear()


def test_alternative_env_names(monkeypatch, fresh_settings):
    for name in ("CSRNET_WEIGHTS", "MONGODB_URI", "ALERT_COOLDOWN_S", "RISK_RED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODEL_PATH", "backend/ml/weights/other.pth")
    monkeypatch.setenv("MONGO_URI", "mongodb+srv://user:pw@cluster0.example.mongodb.net/")
    monkeypatch.setenv("ALERT_COOLDOWN_SECONDS", "120")
    monkeypatch.setenv("THRESHOLD_RED", "7.5")
    s = fresh_settings()
    assert s.csrnet_weights == REPO_ROOT / "backend/ml/weights/other.pth"
    assert s.mongodb_uri == "mongodb+srv://user:pw@cluster0.example.mongodb.net/"
    assert (s.alert_cooldown_s, s.risk_red) == (120.0, 7.5)


def test_alert_channel_list_and_flags(monkeypatch, fresh_settings):
    monkeypatch.setenv("ALERT_CHANNEL", "Console, Telegram")
    monkeypatch.setenv("CSRNET_ALLOW_PICKLE", "true")
    monkeypatch.setenv("FRAME_SOURCE", "LOCAL")
    monkeypatch.setenv("FLOW_MAX_WIDTH", "480")
    s = fresh_settings()
    assert s.alert_channels == ("console", "telegram") and s.csrnet_allow_pickle and s.frame_source == "local"
    assert s.flow_max_width == 480


def test_frames_are_read_locally_by_default_so_no_docker_is_needed(monkeypatch, fresh_settings):
    monkeypatch.delenv("FRAME_SOURCE", raising=False)
    assert fresh_settings().frame_source == "local"


def test_flow_width_defaults_to_320(monkeypatch, fresh_settings):
    monkeypatch.delenv("FLOW_MAX_WIDTH", raising=False)
    assert fresh_settings().flow_max_width == 320


def test_no_mongodb_uri_means_in_memory_storage(monkeypatch, fresh_settings):
    for name in ("MONGODB_URI", "MONGO_URI", "MONGODB_URL", "MONGODB_DB"):
        monkeypatch.delenv(name, raising=False)
    assert (fresh_settings().mongodb_uri, fresh_settings().mongodb_db) == ("", "kumbhmela")


def write_zones(tmp_path, zones):
    path = tmp_path / "zones.json"
    path.write_text(json.dumps({"zones": zones}), encoding="utf-8")
    return path


def test_load_zones_reads_graph_node_and_resolves_file_sources(tmp_path):
    [zone] = load_zones(write_zones(tmp_path, [
        {"id": "cam1", "source": "data/clip.mp4", "area_m2": 50, "graph_node": "GHAT_2"}]))
    assert zone.graph_node == "GHAT_2" and zone.name == "cam1"
    assert zone.source == str(REPO_ROOT / "data/clip.mp4")


def test_load_zones_rejects_bad_configs(tmp_path):
    with pytest.raises(ValueError, match="area_m2"):
        load_zones(write_zones(tmp_path, [{"id": "a", "source": "rtsp://x"}]))
    with pytest.raises(ValueError, match="duplicate"):
        load_zones(write_zones(tmp_path, [{"id": "a", "source": "rtsp://x", "area_m2": 1},
                                          {"id": "a", "source": "rtsp://y", "area_m2": 1}]))


def test_repo_zones_config_is_valid():
    zones = load_zones(REPO_ROOT / "config" / "zones.json")
    assert {z.id for z in zones} == {"RAMKUND_NORTH", "RAMKUND_SOUTH", "GHAT_1"}
