"""P3: the schematic site map for the Command Centre (GET /api/site-map, config/site_map.json)."""

import itertools
import json

from fastapi.testclient import TestClient

from app.alert_dispatcher import AlertDispatcher
from app.config import REPO_ROOT, ZoneConfig
from app.exit_router import DEFAULT_EDGES, ExitRouter
from app.main import create_app
from app.pipeline import ZoneAnalyzer
from app.worker import ZoneOverride
from tests.fakes import FakeEstimator, auth_headers, make_services


def client_for(services):
    app = create_app(services.settings, services_factory=lambda _: services, run_loop=False)
    return TestClient(app, headers=auth_headers(services.settings, role="officer"))


def test_site_map_places_every_node_and_walkway():
    services = make_services()
    with client_for(services) as client:
        body = client.get("/api/site-map").json()
    assert body["schematic"] is True and body["note"] == "Schematic map, not to scale"
    assert body["unplaced"] == []
    nodes = {n["id"]: n for n in body["nodes"]}
    assert set(nodes) == set(services.router.graph.nodes)
    assert nodes["EXIT_GATE_A"]["exit"] and nodes["EXIT_GATE_A"]["kind"] == "gate" and not nodes["GHAT_1"]["exit"]
    assert len(body["edges"]) == len(DEFAULT_EDGES)
    assert {"a": "GHAT_1", "b": "RAMKUND_NORTH", "metres": 50.0} in body["edges"]
    assert all(0 <= n["x"] <= body["width"] and 0 <= n["y"] <= body["height"] for n in body["nodes"])


def test_site_map_needs_a_login():
    services = make_services()
    app = create_app(services.settings, services_factory=lambda _: services, run_loop=False)
    with TestClient(app) as client:
        assert client.get("/api/site-map").status_code == 401


def _cross(p, q, r, s):
    """True if segments pq and rs properly intersect (shared end points do not count)."""
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return (v > 0) - (v < 0)
    if {p, q} & {r, s}:
        return False
    return orient(p, q, r) * orient(p, q, s) < 0 and orient(r, s, p) * orient(r, s, q) < 0


def test_the_repo_layout_has_no_crossing_walkways():
    placed = json.loads((REPO_ROOT / "config" / "site_map.json").read_text(encoding="utf-8"))["nodes"]
    pos = {n: (v["x"], v["y"]) for n, v in placed.items()}
    segments = [(pos[a], pos[b], a, b) for a, b, _ in DEFAULT_EDGES]
    crossings = [(s1[2:], s2[2:]) for s1, s2 in itertools.combinations(segments, 2) if _cross(s1[0], s1[1], s2[0], s2[1])]
    assert crossings == []


def test_a_test_video_reports_its_place_on_the_map():
    services = make_services()
    zone = ZoneConfig("VIDEO_1", "Test camera", "clip.mp4", area_m2=20.0)
    services.zones[zone.id] = zone
    override = ZoneOverride(AlertDispatcher([]), ExitRouter(), "GHAT_2")
    services.worker.add_zone(zone, ZoneAnalyzer(zone, FakeEstimator(), services.settings), override)
    with client_for(services) as client:
        zones = {z["id"]: z for z in client.get("/api/zones").json()}
    assert zones["VIDEO_1"]["graph_node"] == "GHAT_2" and zones["RAMKUND_NORTH"]["graph_node"] == "RAMKUND_NORTH"
