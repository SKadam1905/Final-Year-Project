import csv
import io
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.alert_dispatcher import AlertDispatcher, AlertPolicy
from app.calibration import ZoneGeometry
from app.config import ZoneConfig
from app.main import create_app
from app.video_analysis import CSV_COLUMNS, analyse_video, bursts, safe_filename, write_csv
from app.video_jobs import LiveGate, VideoService, VideoZoneRequest
from app.visualize import density_rgba, density_scale
from app.worker import WithUploadedVideos
from tests.conftest import texture, write_video
from tests.fakes import FakeEstimator, FakeSource, RecordingChannel, make_services


def numbered_video(path, fps, seconds, size=(64, 48)):
    """Frame i is a flat grey of value i (mod 250), so a burst's frames can be identified."""
    frames = [np.full((size[1], size[0], 3), i % 250, np.uint8) for i in range(round(fps * seconds))]
    return write_video(path, frames, fps)


def textured_video(path, fps=25, seconds=5):
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.uniform(0, 255, (120, 160)).astype(np.float32), (0, 0), 2)
    base = cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return write_video(path, [np.dstack([base] * 3)] * round(fps * seconds), fps)


def wait_for(predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    raise AssertionError("timed out")


class Recording:
    """channel_factory that records which channels each dispatcher was given."""

    def __init__(self):
        self.telegram = RecordingChannel()
        self.requests = []

    def __call__(self, telegram, console):
        self.requests.append((telegram, console))
        return [self.telegram] if telegram else []


def client_for(services, channels=None):
    app = create_app(services.settings, services_factory=lambda _: services, run_loop=False,
                     video_factory=lambda settings, s, emit: VideoService(
                         settings, s, lambda name, payload: emitted.append((name, payload)), channel_factory=channels))
    emitted = []
    return TestClient(app), emitted


def upload(client, path, name=None):
    return client.post(f"/api/videos?filename={name or path.name}", content=path.read_bytes())


def analyse_job(client, video_id, **body):
    job = client.post(f"/api/videos/{video_id}/analyse",
                      json={"name": "Test ghat", "area_m2": 100.0, **body}).json()
    wait_for(lambda: client.get(f"/api/jobs/{job['id']}").json()["state"] in ("done", "failed"))
    return client.get(f"/api/jobs/{job['id']}").json()


# Upload validation ---------------------------------------------------------------------

def test_safe_filename():
    assert safe_filename("../../evil name!!.MP4") == "evil_name.mp4"
    assert safe_filename("C:\\Users\\me\\Kumbh clip (1).mov") == "Kumbh_clip_1.mov"
    assert safe_filename("...") == "video"


def test_upload_rejects_wrong_type_bad_files_and_oversized_uploads(tmp_path):
    services = make_services(video_max_mb=1)
    client, _ = client_for(services)
    videos = services.settings.upload_dir / "videos"
    with client:
        wrong = client.post("/api/videos?filename=notes.txt", content=b"hello")
        assert wrong.status_code == 415 and "Unsupported file type '.txt'" in wrong.json()["detail"]

        fake = client.post("/api/videos?filename=crowd.mp4", content=np.random.bytes(50_000))
        assert fake.status_code == 422 and "could not be" in fake.json()["detail"]

        declared = client.post("/api/videos?filename=big.mp4", content=b"x" * (1024 * 1024 + 1))
        assert declared.status_code == 413 and "1 MB limit" in declared.json()["detail"]
        streamed = client.post("/api/videos?filename=big.mp4", content=iter([b"x" * 600_000] * 3))  # no length
        assert streamed.status_code == 413

        assert not [p for p in videos.iterdir() if p.suffix != ".json"]  # nothing rejected is kept

        good = upload(client, numbered_video(tmp_path / "ok clip.avi", 25, 2.4))
        assert good.status_code == 201
        info = good.json()
        assert (info["fps"], info["frames"], info["duration_s"], info["width"]) == (25.0, 60, 2.4, 64)
        assert info["filename"] == "ok clip.avi" and len(list(videos.glob("*ok_clip.avi"))) == 1


# Sampling, density, results ---------------------------------------------------------------

@pytest.mark.parametrize("fps, middle_frames", [(25, [2, 52, 102, 152]), (30, [2, 62, 122, 182])])
def test_bursts_follow_video_time(tmp_path, fps, middle_frames):
    cap = cv2.VideoCapture(str(numbered_video(tmp_path / f"v{fps}.avi", fps, 7)))
    samples = list(bursts(cap, interval_s=2.0, burst=5))
    assert [b.frame_index for b in samples] == middle_frames  # one burst per 2 s of video
    assert [b.time_s for b in samples] == pytest.approx([i / fps for i in middle_frames], abs=1e-3)
    assert [b.start_s for b in samples] == pytest.approx([0, 2, 4, 6], abs=1e-3)
    for b in samples:  # 5 consecutive frames, as a live camera sends
        values = [int(f.mean()) for f in b.frames]
        assert values == pytest.approx(list(range(b.frame_index - 2, b.frame_index + 3)), abs=3)


def test_density_uses_the_given_area_and_level_times_add_up(tmp_path):
    services = make_services()
    zone = ZoneConfig("VIDEO_x", "Test", "clip", area_m2=150.0)
    results = analyse_video(textured_video(tmp_path / "still.avi", seconds=7), zone, FakeEstimator(300), services.settings,
                            tmp_path / "job", AlertDispatcher([], AlertPolicy()), graph_node="GHAT_1")
    samples = results["samples"]
    assert len(samples) == 4 and all(s["count"] == pytest.approx(300, rel=1e-3) for s in samples)
    assert all(s["density"] == pytest.approx(2.0, rel=1e-3) for s in samples)  # 300 people / 150 m²
    assert all(s["turbulence"] == 0 and s["risk_level"] == "YELLOW" for s in samples)  # still frames
    summary = results["summary"]
    assert summary["highest_level"] == "YELLOW" and summary["peak_count"] == pytest.approx(300, rel=1e-3)
    assert sum(summary["time_at_level_s"].values()) == pytest.approx(7.0, abs=0.01)
    assert results["exits_at_peak"]["exits"][0]["path"][0] == "GHAT_1"
    assert len(list((tmp_path / "job" / "frames").glob("*.jpg"))) == 4


def two_shot_video(path, cut_frame, fps=25, seconds=7):
    """Shot A (a greenish scene drifting right) until `cut_frame`, then shot B (a reddish scene)."""
    a = cv2.cvtColor(texture(120, 160, seed=1, blur=3), cv2.COLOR_GRAY2BGR) * np.array([0.5, 1.0, 0.4])
    b = cv2.cvtColor(texture(120, 160, seed=2, blur=3), cv2.COLOR_GRAY2BGR) * np.array([0.4, 0.5, 1.0])
    frames = [np.roll(a if i < cut_frame else b, i, axis=1).astype(np.uint8) for i in range(fps * seconds)]
    return write_video(path, frames, fps)


def test_a_burst_spanning_a_hard_cut_is_skipped_and_kept_out_of_the_risk_average(tmp_path):
    services = make_services()
    zone = ZoneConfig("VIDEO_x", "Test", "clip", area_m2=150.0)
    # Bursts start at frames 0, 50, 100, 150; the cut between frames 51 and 52 falls in the second.
    results = analyse_video(two_shot_video(tmp_path / "cut.avi", cut_frame=52), zone, FakeEstimator(300),
                            services.settings, tmp_path / "job", AlertDispatcher([], AlertPolicy()))
    samples = results["samples"]
    assert [s["scene_cut"] for s in samples] == [False, True, False, False]
    assert samples[1]["count"] is None and samples[1]["risk_level"] is None
    assert [s["samples"] for s in samples] == [1, None, 2, 3]  # the cut never entered the rolling mean
    assert not (tmp_path / "job" / "density" / "00001.npy").exists()
    summary = results["summary"]
    assert (summary["samples"], summary["scene_cut_samples"], summary["time_scene_cut_s"]) == (3, 1, 2.0)
    assert sum(summary["time_at_level_s"].values()) == pytest.approx(5.0, abs=0.01)  # 7 s minus the cut

    rows = list(csv.DictReader(open(write_csv(results, tmp_path / "r.csv"), encoding="utf-8")))
    # Both shots drift 1 px/frame, which is a pan: those samples are "moving camera"
    assert [r["flag"] for r in rows] == ["moving camera", "scene cut", "moving camera", "moving camera"]
    assert rows[1]["count"] == "" and rows[1]["time_s"] == "2.08"


def test_heatmap_uses_a_fixed_persons_per_m2_scale():
    scale = density_scale(2.0, 4.0, 6.0)
    assert [v for v, _ in scale.stops] == [0, 2, 4, 6, 7]

    def colour(per_m2):  # (B, G, R, alpha) in the middle of a uniform map
        return density_rgba(np.full((45, 80), per_m2, np.float32), (160, 90), scale)[45, 80].astype(int)

    sparse, red, beyond = colour(0.23), colour(6.0), colour(12.0)
    assert sparse[1] > sparse[2] and sparse[3] < 0.2 * 255  # faint green, never red
    assert red[2] > 150 and red[1] < 60 and red[0] < 60 and red[3] > 0.65 * 255  # red at the RED threshold
    assert (beyond == colour(7.0)).all() and beyond[2] > 2 * beyond[1]  # 7 and above: the same dark red
    b, g, r, _ = colour(4.0)
    assert r > 200 and 90 < g < 140 and b < 40  # orange at the ORANGE threshold

    # The colour depends on persons/m² only: one dense spot elsewhere does not repaint the rest.
    spot = np.full((45, 80), 0.23, np.float32)
    spot[5, 5] = 9.0
    assert (density_rgba(spot, (160, 90), scale)[60, 120] == colour(0.23)).all()

    # 700 people spread over 2,997 m² show 0.23 /m² in every cell, the zone's own density.
    cells = ZoneGeometry(area_m2=2997.0).cell_density(np.full((45, 80), 700 / 3600, np.float32), (640, 360))
    assert cells == pytest.approx(np.full((45, 80), 700 / 2997), rel=1e-4)


# Jobs ------------------------------------------------------------------------------------

def test_jobs_run_one_at_a_time_in_order(tmp_path):
    services = make_services()
    video = VideoService(services.settings, services)
    video.start()
    meta = video.register_upload(*video.new_upload("clip.avi")[:1], numbered_video(tmp_path / "c.avi", 25, 1), "c.avi", 1)
    running, peak, order, lock = [0], [0], [], threading.Lock()

    def slow_analyse(job):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
            order.append(job.name)
        time.sleep(0.15)
        with lock:
            running[0] -= 1
        job.state = "done"

    video._analyse = slow_analyse
    jobs = [video.create_job(meta["id"], VideoZoneRequest(f"job{i}", 100.0)) for i in range(3)]
    assert video.queue_position(jobs[2].id) >= 1
    wait_for(lambda: all(j.state == "done" for j in jobs))
    video.stop()
    assert peak[0] == 1 and order == ["job0", "job1", "job2"]


def test_alerts_are_suppressed_by_default_and_prefixed_when_enabled(tmp_path):
    services = make_services(count=700, telegram_bot_token="123:abc", telegram_chat_id="42")  # 7 /m² -> RED
    channels = Recording()
    client, _ = client_for(services, channels)
    with client:
        video_id = upload(client, textured_video(tmp_path / "dense.avi", seconds=3)).json()["id"]
        quiet = client.get(f"/api/jobs/{analyse_job(client, video_id)['id']}/results").json()
        assert [a["risk_level"] for a in quiet["alerts"]] == ["RED"]
        assert not quiet["alerts"][0]["sent"] and quiet["alerts"][0]["deliveries"] == []
        assert channels.telegram.sent == []

        loud = client.get(f"/api/jobs/{analyse_job(client, video_id, telegram_alerts=True)['id']}/results").json()
        assert loud["alerts"][0]["sent"]
        [message] = channels.telegram.sent
        assert message.startswith("[VIDEO TEST] RED: Test ghat") and loud["alerts"][0]["message"] == message
        assert channels.requests == [(False, False), (True, False)]
    assert services.store.recent_alerts() == []  # never in the live alert list


def test_telegram_alerts_need_telegram_configured(tmp_path):
    services = make_services(telegram_bot_token="", telegram_chat_id="")
    client, _ = client_for(services)
    with client:
        video_id = upload(client, textured_video(tmp_path / "v.avi", seconds=1)).json()["id"]
        body = {"name": "x", "area_m2": 100.0, "telegram_alerts": True}
        response = client.post(f"/api/videos/{video_id}/analyse", json=body)
        assert response.status_code == 400 and "Telegram is not set up" in response.json()["detail"]


def test_analyse_once_end_to_end_with_csv_frames_annotated_video_and_delete(tmp_path):
    services = make_services(count=300)
    client, emitted = client_for(services)
    with client:
        options = client.get("/api/video/options").json()
        assert options["max_upload_mb"] == 500 and "GHAT_1" in options["locations"]
        assert "EXIT_GATE_A" not in options["locations"]
        video_id = upload(client, textured_video(tmp_path / "v.avi", seconds=5)).json()["id"]
        job = analyse_job(client, video_id, graph_node="GHAT_1")
        assert job["state"] == "done" and job["progress"] == 100
        assert any(name == "video_job" and p["state"] == "running" for name, p in emitted)

        results = client.get(f"/api/jobs/{job['id']}/results").json()
        assert results["summary"]["samples"] == 3 and results["zone"]["area_m2"] == 100.0

        response = client.get(f"/api/jobs/{job['id']}/results.csv")
        assert response.headers["content-disposition"].startswith("attachment")
        rows = list(csv.reader(io.StringIO(response.text)))
        assert rows[0] == CSV_COLUMNS == ["time_s", "frame", "count", "count_method", "density_count",
                                          "detection_count", "density_per_m2", "turbulence", "raw_pressure",
                                          "pressure", "risk_level", "readings_in_mean", "hotspot", "hotspot_count",
                                          "camera_motion_px", "flag", "inference_ms", "detection_ms", "flow_ms"]
        row = dict(zip(rows[0], rows[1]))
        assert len(rows) == 4 and row["risk_level"] == "YELLOW" and row["flag"] == ""  # fixed camera, no cut
        assert row["count_method"] == "density" and row["detection_count"] == ""  # no detector in this test
        assert row["hotspot"]  # the most crowded grid cell, e.g. "top-left"

        frame = client.get(f"/api/jobs/{job['id']}/samples/1/frame.jpg")
        overlay = client.get(f"/api/jobs/{job['id']}/samples/1/density.png")
        assert frame.headers["content-type"] == "image/jpeg" and overlay.headers["content-type"] == "image/png"
        assert client.get(f"/api/jobs/{job['id']}/samples/9/frame.jpg").status_code == 404

        assert client.get(f"/api/jobs/{job['id']}/annotated.mp4").status_code == 409
        client.post(f"/api/jobs/{job['id']}/annotated")
        wait_for(lambda: client.get(f"/api/jobs/{job['id']}").json()["annotated"]["state"] == "done")
        mp4 = client.get(f"/api/jobs/{job['id']}/annotated.mp4")
        assert mp4.headers["content-type"] == "video/mp4" and len(mp4.content) > 1000

        assert client.delete(f"/api/jobs/{job['id']}").status_code == 204
        assert client.get(f"/api/jobs/{job['id']}").status_code == 404
        assert not (services.settings.upload_dir / "jobs" / job["id"]).exists()
        assert not list((services.settings.upload_dir / "videos").glob("*.avi"))  # unused upload removed too


def test_finished_jobs_survive_a_restart_and_unfinished_ones_are_marked_failed(tmp_path):
    services = make_services()
    first = VideoService(services.settings, services)
    first.start()
    meta = first.register_upload(*first.new_upload("c.avi")[:1], textured_video(tmp_path / "c.avi", seconds=1), "c.avi", 1)
    done = first.create_job(meta["id"], VideoZoneRequest("done", 100.0))
    wait_for(lambda: done.state == "done")
    first.stop()
    waiting = first.create_job(meta["id"], VideoZoneRequest("waiting", 100.0))  # queued, never run

    second = VideoService(services.settings, services)
    second.start()
    states = {j["name"]: (j["state"], j["error"]) for j in second.list_jobs()}
    assert states["done"] == ("done", None) and states["waiting"][0] == "failed"
    assert "restarted" in states["waiting"][1] and second.results(done.id)["summary"]["samples"] == 1
    second.stop()


# Video as a live zone ------------------------------------------------------------------------

def test_video_runs_as_a_live_zone_with_its_own_exit_graph_and_can_be_removed(tmp_path):
    services = make_services(count=700, source=WithUploadedVideos(FakeSource()))
    client, emitted = client_for(services)
    with client:
        video_id = upload(client, textured_video(tmp_path / "loop.avi", seconds=2)).json()["id"]
        added = client.post(f"/api/videos/{video_id}/live",
                            json={"name": "Phone video", "area_m2": 100.0, "graph_node": "GHAT_3"})
        assert added.status_code == 201 and added.json()["id"] == "VIDEO_1"
        zones = {z["id"]: z for z in client.get("/api/zones").json()}
        assert zones["VIDEO_1"]["kind"] == "video" and zones["RAMKUND_NORTH"]["kind"] == "camera"
        assert ("zones_changed", {"added": "VIDEO_1"}) in emitted

        events = []
        wait_for(lambda: events.extend(services.worker.step_zone("VIDEO_1")) or events)
        update = next(p for name, p in events if name == "density_update")
        assert update["zone_id"] == "VIDEO_1" and update["risk_level"] == "RED"
        assert update["exits"][0]["path"][0] == "GHAT_3"
        assert services.router.graph.nodes["GHAT_3"]["pressure"] == 0  # live exit advice untouched
        alert = next(p for name, p in events if name == "alert")
        assert alert["message"].startswith("[VIDEO TEST] RED: Phone video")
        assert [d["channel"] for d in alert["deliveries"]] == ["console"]  # no Telegram unless ticked

        assert client.delete("/api/zones/RAMKUND_NORTH").status_code == 409  # configured camera
        assert client.delete("/api/zones/NOPE").status_code == 404
        assert client.delete("/api/zones/VIDEO_1").status_code == 204
        assert "VIDEO_1" not in [z["id"] for z in client.get("/api/zones").json()]
        assert "VIDEO_1" not in services.worker.zones and services.worker.step_zone("VIDEO_1") == []
        assert ("zones_changed", {"removed": "VIDEO_1"}) in emitted


# Live monitoring keeps priority ----------------------------------------------------------------

def test_live_gate_waits_for_the_live_cycle_and_backs_off_while_live_is_busy():
    gate = LiveGate(interval_s=2.0)
    slept = []
    gate.wait_turn(sleep=slept.append)
    assert slept == []  # live is idle and on time: no delay

    gate.cycle_started()
    released = threading.Event()
    threading.Thread(target=lambda: (gate.wait_turn(sleep=slept.append), released.set())).start()
    assert not released.wait(0.2)  # a job never runs during a live cycle
    gate.cycle_finished(1.9)  # ...and this one nearly overran its 2 s
    assert released.wait(2)
    gate.wait_turn(sleep=slept.append)
    assert slept == [0.5, 1.0] and gate.backoffs == 2  # backing off harder while live stays busy
    for _ in range(3):
        gate.cycle_started()
        gate.cycle_finished(0.5)
    gate.wait_turn(sleep=slept.append)
    assert slept == [0.5, 1.0]  # live recovered: full speed again
