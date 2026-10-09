"""Uploaded videos: the analysis queue and videos looping as extra live zones.

One background thread runs every job (analyses and annotated-video renders) one at a time.
Live monitoring keeps priority through LiveGate: a job works only between live cycles, and
backs off while live cycles run close to their 2 s budget, so a long analysis slows down
rather than making the live cameras late.

On disk, under UPLOAD_DIR (data/uploads, git-ignored):
    videos/<video_id>__<name>.<ext> + <video_id>.json   uploaded file and its probe info
    jobs/<job_id>/job.json, results.json, results.csv     job state and results
    jobs/<job_id>/frames/, density/, gradcam/             each analysed frame and its overlays
    jobs/<job_id>/annotated.mp4                           made on request
Finished jobs survive a restart; one that was queued or running is marked failed. Video zones
are not restored after a restart (add them again), and an upload that no job or zone uses is
deleted at start-up.
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from app.alert_dispatcher import AlertDispatcher, AlertPolicy, Channel, build_channels
from app.config import Settings, ZoneConfig
from app.exit_router import ExitRouter
from app.video_analysis import Cancelled, VideoRejected, analyse_video, probe_video, render_annotated, write_csv

log = logging.getLogger(__name__)

VIDEO_PREFIX = "[VIDEO TEST] "
MAX_VIDEO_ZONES = 4


class LiveGate:
    """Lets background jobs run only when live monitoring can spare the time.

    The live loop calls cycle_started()/cycle_finished(); a job calls wait_turn() before each
    unit of work. It waits for the current live cycle to finish, then, if any of the last
    three cycles took more than `busy_fraction` of the interval, sleeps (0.5 s, doubling up to
    the interval) so the live cameras catch up.
    """

    def __init__(self, interval_s: float = 2.0, busy_fraction: float = 0.8, max_wait_s: float = 10.0) -> None:
        self.interval_s = interval_s
        self.busy_fraction = busy_fraction
        self.max_wait_s = max_wait_s
        self._idle = threading.Event()
        self._idle.set()
        self._recent: deque[float] = deque(maxlen=3)
        self._backoff = 0.0
        self.backoffs = 0  # how often a job had to wait for live monitoring

    def cycle_started(self) -> None:
        self._idle.clear()

    def cycle_finished(self, seconds: float) -> None:
        self._recent.append(seconds)
        self._idle.set()

    def live_is_busy(self) -> bool:
        return any(s > self.busy_fraction * self.interval_s for s in self._recent)

    def wait_turn(self, sleep: Callable[[float], None] = time.sleep) -> None:
        self._idle.wait(self.max_wait_s)
        if self.live_is_busy():
            self._backoff = min(self.interval_s, max(0.5, self._backoff * 2))
            self.backoffs += 1
            sleep(self._backoff)
            self._idle.wait(self.max_wait_s)
        else:
            self._backoff = 0.0


@dataclass
class Job:
    id: str
    video_id: str
    video_name: str
    name: str
    area_m2: float
    graph_node: str | None
    telegram_alerts: bool
    created_at: float
    state: str = "queued"  # queued | running | done | failed
    progress: float = 0.0
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    annotated: dict = field(default_factory=lambda: {"state": "none", "progress": 0.0, "error": None})
    slowed_for_live: int = 0  # times the job paused so live cameras stayed on time

    def public(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class VideoZoneRequest:
    name: str
    area_m2: float
    graph_node: str | None = None
    telegram_alerts: bool = False


class VideoService:
    def __init__(self, settings: Settings, services, emit: Callable[[str, dict], None] = lambda name, payload: None,
                 channel_factory: Callable[[bool, bool], list[Channel]] | None = None,
                 gate: LiveGate | None = None) -> None:
        self.settings = settings
        self.services = services
        self.emit = emit
        self.gate = gate or LiveGate(settings.burst_interval_s)
        self.channel_factory = channel_factory or self._channels
        self.root = settings.upload_dir
        self.videos_dir, self.jobs_dir = self.root / "videos", self.root / "jobs"
        self.videos: dict[str, dict] = {}
        self.jobs: dict[str, Job] = {}
        self.video_zones: dict[str, str] = {}  # zone id -> video id
        self._queue: queue.Queue[tuple[str, str] | None] = queue.Queue()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._current: str | None = None  # the job the worker thread is on
        self._stopping = threading.Event()
        self._last_emit: dict[str, float] = {}

    # Set-up -----------------------------------------------------------------------------

    def start(self) -> None:
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        for meta in self.videos_dir.glob("*.json"):
            try:
                video = json.loads(meta.read_text(encoding="utf-8"))
                self.videos[video["id"]] = {**video, "path": Path(video["path"])}
            except (ValueError, KeyError, OSError):
                log.warning("ignoring unreadable upload record %s", meta)
        for job_file in self.jobs_dir.glob("*/job.json"):
            try:
                job = Job(**json.loads(job_file.read_text(encoding="utf-8")))
            except (ValueError, TypeError, OSError):
                log.warning("ignoring unreadable job %s", job_file)
                continue
            if job.state in ("queued", "running"):
                job.state, job.error = "failed", "Interrupted: the backend restarted while this job was waiting or running."
            if job.annotated["state"] in ("queued", "running"):
                job.annotated = {"state": "none", "progress": 0.0, "error": None}
            self.jobs[job.id] = job
            self._save(job)
        for video_id in list(self.videos):
            self._delete_video_if_unused(video_id)
        for leftover in self.videos_dir.glob("*__*"):  # files whose record is gone (e.g. locked at deletion)
            if leftover.name.split("__", 1)[0] not in self.videos:
                leftover.unlink(missing_ok=True)
        self._thread = threading.Thread(target=self._run, name="video-jobs", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()  # a running job stops at its next sample and is marked interrupted
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=10)
        for zone_id in list(self.video_zones):
            self.remove_live(zone_id, delete_unused=False)

    def options(self) -> dict:
        router = self.services.router
        estimator = self.services.estimator
        return {
            "max_upload_mb": self.settings.video_max_mb, "file_types": [".mp4", ".avi", ".mov", ".mkv"],
            "telegram_available": bool(self.settings.telegram_bot_token and self.settings.telegram_chat_id),
            "model_loaded": estimator is not None, "model_error": self.services.model_error,
            "model": getattr(estimator, "info", None),
            "locations": sorted(n for n in router.graph if n not in router.exits),
            "burst_interval_s": self.settings.burst_interval_s, "burst_size": self.settings.burst_size,
            "max_video_zones": MAX_VIDEO_ZONES,
        }

    # Uploads ----------------------------------------------------------------------------

    def new_upload(self, filename: str) -> tuple[str, Path]:
        video_id = uuid.uuid4().hex[:12]
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        return video_id, self.videos_dir / f"{video_id}__{filename}"

    def register_upload(self, video_id: str, path: Path, original_name: str, size: int) -> dict:
        """Probe the saved file (VideoRejected if unusable) and remember it."""
        try:
            info = probe_video(path)
        except VideoRejected:
            path.unlink(missing_ok=True)
            raise
        video = {"id": video_id, "filename": original_name, "path": path, "size_bytes": size,
                 "uploaded_at": time.time(), **info}
        (self.videos_dir / f"{video_id}.json").write_text(json.dumps({**video, "path": str(path)}), encoding="utf-8")
        with self._lock:
            self.videos[video_id] = video
        return self.public_video(video)

    @staticmethod
    def public_video(video: dict) -> dict:
        return {k: v for k, v in video.items() if k != "path"}

    def video(self, video_id: str) -> dict:
        video = self.videos.get(video_id)
        if video is None or not video["path"].exists():
            raise KeyError(video_id)
        return video

    def _delete_video_if_unused(self, video_id: str) -> None:
        with self._lock:
            if any(j.video_id == video_id for j in self.jobs.values()) or video_id in self.video_zones.values():
                return
            video = self.videos.pop(video_id, None)
        if video:
            try:
                video["path"].unlink(missing_ok=True)
            except OSError as exc:  # still open somewhere (Windows): removed at the next start-up
                log.warning("could not delete upload %s yet: %s", video["path"].name, exc)
            (self.videos_dir / f"{video_id}.json").unlink(missing_ok=True)

    # Jobs -------------------------------------------------------------------------------

    def create_job(self, video_id: str, request: VideoZoneRequest) -> Job:
        video = self.video(video_id)
        job = Job(id=uuid.uuid4().hex[:12], video_id=video_id, video_name=video["filename"], name=request.name,
                  area_m2=request.area_m2, graph_node=request.graph_node, telegram_alerts=request.telegram_alerts,
                  created_at=time.time())
        with self._lock:
            self.jobs[job.id] = job
        self._save(job)
        self._queue.put(("analyse", job.id))
        self._publish(job, force=True)
        return job

    def list_jobs(self) -> list[dict]:
        return [j.public() for j in sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)]

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def results(self, job_id: str) -> dict:
        return json.loads((self.job_dir(job_id) / "results.json").read_text(encoding="utf-8"))

    def delete_job(self, job_id: str) -> None:
        with self._lock:
            job = self.jobs.pop(job_id, None)
            running = self._current == job_id
        if job is None:
            raise KeyError(job_id)
        if not running:  # a running job notices at its next sample, stops and cleans up itself
            self._remove_job_files(job)
        self.emit("video_job", {**job.public(), "state": "deleted"})

    def _remove_job_files(self, job: Job) -> None:
        shutil.rmtree(self.job_dir(job.id), ignore_errors=True)
        self._delete_video_if_unused(job.video_id)

    def request_annotated(self, job_id: str) -> Job:
        job = self.jobs[job_id]
        if job.state != "done":
            raise ValueError("the analysis has not finished")
        if job.annotated["state"] in ("queued", "running", "done"):
            return job
        job.annotated = {"state": "queued", "progress": 0.0, "error": None}
        self._save(job)
        self._queue.put(("annotate", job.id))
        self._publish(job, force=True)
        return job

    def queue_position(self, job_id: str) -> int:
        """How many tasks run before this one (0 = running or next)."""
        waiting = [item[1] for item in list(self._queue.queue) if item]
        return waiting.index(job_id) if job_id in waiting else 0

    def _cancelled(self, job_id: str) -> Callable[[], bool]:
        return lambda: job_id not in self.jobs or self._stopping.is_set()

    def _run(self) -> None:
        while (item := self._queue.get()) is not None:
            kind, job_id = item
            with self._lock:
                job = self.jobs.get(job_id)
                self._current = job_id if job else None
            if job is None:
                continue  # deleted while queued
            error = None
            try:
                if kind == "analyse":
                    self._analyse(job)
                else:
                    self._annotate(job)
            except Cancelled:
                if self._stopping.is_set():
                    error = "Interrupted: the backend shut down while this job was running."
            except Exception as exc:  # report it on the job; the queue carries on
                if job_id in self.jobs:
                    log.exception("video job %s failed", job_id)
                    error = str(exc)
            if error and kind == "analyse":
                job.state, job.error, job.finished_at = "failed", error, time.time()
            elif error:
                job.annotated = {"state": "failed", "progress": 0.0, "error": error}
            with self._lock:
                self._current = None
                deleted = job_id not in self.jobs
            if deleted:
                self._remove_job_files(job)
            else:
                self._save(job)
                self._publish(job, force=True)

    def _wait_turn(self, job: Job) -> Callable[[], None]:
        def wait() -> None:
            before = self.gate.backoffs
            self.gate.wait_turn()
            job.slowed_for_live += self.gate.backoffs - before

        return wait

    def _analyse(self, job: Job) -> None:
        estimator = self.services.estimator
        if estimator is None:
            raise RuntimeError(f"no crowd model loaded: {self.services.model_error}")
        video = self.video(job.video_id)
        job.state, job.started_at, job.progress = "running", time.time(), 0.0
        self._save(job)
        self._publish(job, force=True)
        zone = ZoneConfig(id=f"VIDEO_{job.id}", name=job.name, source=str(video["path"]), area_m2=job.area_m2)
        dispatcher = AlertDispatcher(self.channel_factory(job.telegram_alerts, False),
                                     AlertPolicy(self.settings.alert_cooldown_s), prefix=VIDEO_PREFIX)

        def progress(percent: float) -> None:
            job.progress = round(percent, 1)
            self._publish(job)

        try:
            results = analyse_video(video["path"], zone, estimator, self.settings, self.job_dir(job.id), dispatcher,
                                    job.graph_node, progress, self._cancelled(job.id), self._wait_turn(job),
                                    detector=self.services.detector)
        except VideoRejected as exc:
            raise RuntimeError(str(exc)) from exc
        results["job"] = {"id": job.id, "video_name": job.video_name, "telegram_alerts": job.telegram_alerts}
        (self.job_dir(job.id) / "results.json").write_text(json.dumps(results), encoding="utf-8")
        write_csv(results, self.job_dir(job.id) / "results.csv")
        job.state, job.progress, job.finished_at = "done", 100.0, time.time()

    def _annotate(self, job: Job) -> None:
        job.annotated = {"state": "running", "progress": 0.0, "error": None}
        self._publish(job, force=True)

        def progress(percent: float) -> None:
            job.annotated["progress"] = round(percent, 1)
            self._publish(job)

        folder = self.job_dir(job.id)
        render_annotated(self.video(job.video_id)["path"], self.results(job.id), folder / "density",
                         folder / "annotated.mp4", progress, self._cancelled(job.id), self._wait_turn(job))
        job.annotated = {"state": "done", "progress": 100.0, "error": None}

    def _save(self, job: Job) -> None:
        folder = self.job_dir(job.id)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "job.json").write_text(json.dumps(job.public()), encoding="utf-8")

    def _publish(self, job: Job, force: bool = False) -> None:
        """Progress over Socket.IO, at most twice a second per job (state changes always)."""
        now = time.monotonic()
        if force or now - self._last_emit.get(job.id, 0.0) >= 0.5:
            self._last_emit[job.id] = now
            self.emit("video_job", job.public())

    # Video zones (live) -----------------------------------------------------------------

    def _channels(self, telegram: bool, console: bool) -> list[Channel]:
        names = (["console"] if console else []) + (["telegram"] if telegram else [])
        return build_channels(self.settings, names) if names else []

    def add_live(self, video_id: str, request: VideoZoneRequest) -> ZoneConfig:
        """Loop the video as a new live zone. Call from the event loop thread (zones must not
        change while a cycle is being scheduled)."""
        s = self.services
        if s.worker is None:
            raise RuntimeError(f"no crowd model loaded: {s.model_error}")
        if len(self.video_zones) >= MAX_VIDEO_ZONES:
            raise OverflowError(f"at most {MAX_VIDEO_ZONES} videos can run as live cameras at once")
        video = self.video(video_id)
        n = 1
        while f"VIDEO_{n}" in s.zones:
            n += 1
        zone = ZoneConfig(id=f"VIDEO_{n}", name=request.name, source=str(video["path"]), area_m2=request.area_m2)
        from app.pipeline import ZoneAnalyzer
        from app.worker import ZoneOverride

        router = ExitRouter()
        dispatcher = AlertDispatcher(self.channel_factory(request.telegram_alerts, True),
                                     AlertPolicy(self.settings.alert_cooldown_s), prefix=VIDEO_PREFIX)
        override = ZoneOverride(dispatcher, router, request.graph_node if request.graph_node in router.graph else None)
        s.source.add_video(zone)
        s.worker.add_zone(zone, ZoneAnalyzer(zone, s.estimator, self.settings, detector=s.detector), override)
        s.zones[zone.id] = zone
        with self._lock:
            self.video_zones[zone.id] = video_id
        try:
            s.store.sync_zones([zone])
        except Exception as exc:  # database down: the zone still runs
            log.warning("could not save video zone %s: %s", zone.id, exc)
        self.emit("zones_changed", {"added": zone.id})
        return zone

    def remove_live(self, zone_id: str, delete_unused: bool = True) -> None:
        s = self.services
        with self._lock:
            video_id = self.video_zones.pop(zone_id)
        s.zones.pop(zone_id, None)
        if s.worker is not None:
            s.worker.remove_zone(zone_id)
        s.source.remove_video(zone_id)
        if delete_unused:
            self._delete_video_if_unused(video_id)
        self.emit("zones_changed", {"removed": zone_id})
