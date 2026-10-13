"""FastAPI + Socket.IO backend (plan Layer 4), running the inference loop every 2 s.

    cd backend
    uvicorn app.main:app --port 8000          # interactive docs: http://localhost:8000/docs

Every /api/ route needs a session token (app/auth.py) except POST /api/auth/login; video, jobs, reports,
removing a zone and /infer need the admin role. 401 = not logged in or expired, 403 = wrong role.

REST
    POST /api/auth/login | /logout, GET /api/auth/me   session token for {username, password}; who is logged in
    GET  /health                            status, model_loaded, zones, storage, frame source, cameras (public)
    GET  /api/config                        risk thresholds and timing (for the dashboard)
    GET  /api/zones                         zone configs (camera passwords hidden)
    GET  /api/zones/{zone_id}/status        latest reading
    GET  /api/zones/{zone_id}/history       last N readings, oldest first (?limit=150)
    GET  /api/zones/{zone_id}/exits         best exits now (Dijkstra over live crowd pressure)
    GET  /api/zones/{zone_id}/snapshot.jpg  latest analysed camera frame
    GET  /api/zones/{zone_id}/density.png   persons/m² heatmap (fixed scale, /api/config) sized to the snapshot
    GET  /api/zones/{zone_id}/detections.png  YOLO person boxes: transparent overlay sized to the snapshot
    GET  /api/zones/{zone_id}/gradcam.png   Grad-CAM of the count: transparent overlay sized to the snapshot
    POST /api/zones/{zone_id}/infer         run inference now on the zone's newest frames
    GET  /api/alerts/recent                 last 20 alerts, newest first
    POST /api/alerts/{alert_id}/ack         acknowledge an alert (who and when are stored); emits "alert_ack"
    GET  /api/incidents                     last 50 incidents (ORANGE/RED periods)
    GET  /api/site-map                      schematic node positions (config/site_map.json) + the walkways
Video analysis (upload, jobs, video zones): see app/video_api.py. Reports (filtered history, summaries): app/reports.py.
Socket.IO (path /socket.io; connect with auth {token}, refused without a valid one)
    density_update  {zone_id, name, timestamp, server_time, density, turbulence, pressure, raw_pressure,
                     risk_level, count, samples, exits, count_method ("density" | "detection"),
                     density_count, detection_count, grid {rows, cols, counts, density, hotspot}, hotspot,
                     speed, surge_ratio, surge_fired, escalation ("speed surge" when the surge trigger raised the level)}
    alert           {id, zone_id, risk_level, message, timestamp, incident_id, delivered, deliveries, hotspot,
                     gate_actions, reason, acknowledged_by, acknowledged_at}; "alert_ack" (same shape) once acknowledged
New clients get the latest density_update of every zone as soon as they connect.

Timestamps are Unix epoch seconds; `server_time` lets a client judge a reading's age without
trusting its own clock. Zone endpoints return 404 for unknown zones; the image endpoints return
503 with a reason when there is no frame or no model yet, and an X-Frame-Timestamp header with
the frame's capture time.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import threading
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
import socketio
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.auth import ROLES, Sessions, TokenError, UserStore, required_role, user_from_claims
from app.config import Settings, ZoneConfig, get_settings, redact
from app.reports import add_report_routes
from app.video_api import add_video_routes
from app.video_jobs import VideoService
from app.visualize import colour_overlay, density_rgba, density_scale, detections_rgba
from app.worker import Event, Services, ZoneFrame, build_services, graph_node

# Our modules log at INFO; libraries stay at WARNING (httpx would log Telegram URLs, which
# contain the bot token, at INFO).
logging.basicConfig(format="%(levelname)s:     %(name)s - %(message)s")
logging.getLogger("app").setLevel(logging.INFO)
log = logging.getLogger(__name__)


class HideTokens(logging.Filter):
    """Keep ?token=... (session tokens on image and download links) out of the access log."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(re.sub(r"token=[^&\s]+", "token=***", a) if isinstance(a, str) else a
                                for a in record.args)
        return True


logging.getLogger("uvicorn.access").addFilter(HideTokens())


class LoginBody(BaseModel):
    username: str
    password: str

DISPLAY_WIDTH = 960  # snapshot and overlays share this width, so they line up
GRADCAM_WIDTH = 640  # Grad-CAM needs a backward pass: keep its input small


def zone_info(zone: ZoneConfig, node: str | None) -> dict:
    cal = zone.calibration
    return {"id": zone.id, "name": zone.name, "source": redact(zone.source), "graph_node": node,
            "area_m2": zone.area_m2, "area_source": zone.area_source, "area_m2_estimated": zone.area_m2_estimated,
            "calibrated": cal is not None, "calibration_source": cal.source if cal else None, "roi": zone.roi,
            "frame_size": zone.frame_size}


def with_server_time(payload: dict) -> dict:
    return {**payload, "server_time": time.time()}


def display_size(frame: np.ndarray) -> tuple[int, int]:
    h, w = frame.shape[:2]
    return (w, h) if w <= DISPLAY_WIDTH else (DISPLAY_WIDTH, round(h * DISPLAY_WIDTH / w))


def encode(image: np.ndarray, fmt: str) -> bytes:
    ok, buffer = cv2.imencode(".jpg" if fmt == "jpeg" else ".png", image,
                              [cv2.IMWRITE_JPEG_QUALITY, 85] if fmt == "jpeg" else [])
    if not ok:
        raise HTTPException(500, "image encoding failed")
    return buffer.tobytes()


def image_response(data: bytes, fmt: str, timestamp: float) -> Response:
    return Response(data, media_type=f"image/{fmt}",
                    headers={"Cache-Control": "no-store", "X-Frame-Timestamp": f"{timestamp:.3f}"})


def quiet_connection_resets(loop, context: dict) -> None:
    """Event-loop exception handler. On Windows the proactor loop logs a full traceback whenever a
    browser drops its connection (ConnectionResetError in _call_connection_lost): normal, not an
    error. Everything else goes to the default handler."""
    if isinstance(context.get("exception"), ConnectionResetError):
        log.debug("client connection reset: %s", context.get("message"))
        return
    loop.default_exception_handler(context)


async def run_cycle(worker, emit) -> None:
    """One inference cycle: every zone in parallel threads (optical flow is CPU-bound and OpenCV
    releases the GIL); each zone's events are emitted as soon as that zone is done, so a
    reading's age does not grow with the number of cameras."""

    async def run_zone(zone_id: str) -> None:
        try:
            await emit(await asyncio.to_thread(worker.step_zone, zone_id))
        except Exception:
            log.exception("inference loop error in zone %s", zone_id)

    await asyncio.gather(*(run_zone(zone_id) for zone_id in list(worker.zones)))


def create_app(settings: Settings | None = None, services_factory=build_services, run_loop: bool = True,
               video_factory=VideoService):
    settings = settings or get_settings()
    origins = list(settings.cors_origins)
    heatmap_scale = density_scale(settings.risk_yellow, settings.risk_orange, settings.risk_red)
    sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*" if origins == ["*"] else origins)

    async def emit(events: list[Event]) -> None:
        for name, payload in events:
            await sio.emit(name, with_server_time(payload) if name == "density_update" else payload)

    cycle_times: list[float] = []  # the last live cycles' durations, seconds

    async def inference_loop(services: Services, gate) -> None:
        while True:
            started = time.monotonic()
            gate.cycle_started()  # video analysis waits while a live cycle runs
            try:
                await run_cycle(services.worker, emit)
            finally:
                elapsed = time.monotonic() - started
                gate.cycle_finished(elapsed)
                cycle_times[:] = [*cycle_times[-29:], elapsed]
            await asyncio.sleep(max(0.0, settings.burst_interval_s - (time.monotonic() - started)))

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        services = services_factory(settings)
        api.state.services = services
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(quiet_connection_resets)

        def emit_from_thread(name: str, payload: dict) -> None:
            asyncio.run_coroutine_threadsafe(sio.emit(name, payload), loop)

        video = video_factory(settings, services, emit_from_thread)
        api.state.video = video
        services.source.start()
        video.start()
        task = asyncio.create_task(inference_loop(services, video.gate)) if run_loop and services.worker else None
        try:
            yield
        finally:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await asyncio.to_thread(video.stop)
            services.source.stop()
            services.store.close()  # saves writes still queued for MongoDB

    api = FastAPI(title="Kumbh Mela crowd-safety API", version="0.4.0", lifespan=lifespan)
    sessions = Sessions(settings.secret_key, settings.session_hours)
    users = UserStore(settings.users_config)
    if settings.secret_key_ephemeral:
        log.warning("SECRET_KEY is not set in .env: using a random one, so everyone must log in again after a "
                    "restart. Run: python -m tools.create_user --list (it adds SECRET_KEY to .env)")

    @api.middleware("http")
    async def require_login(request: Request, call_next):
        """Every /api/ route needs a valid session token (header or ?token=); some need the admin role.
        Added before CORS, so CORS wraps it and the browser can read the 401/403 replies."""
        rule = required_role(request.method, request.url.path)
        if rule is None or request.method == "OPTIONS":
            return await call_next(request)
        header = request.headers.get("authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else request.query_params.get("token")
        try:
            claims = sessions.verify(token)
        except TokenError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=401)
        if rule == "admin" and claims["role"] != "admin":
            return JSONResponse({"detail": f"This needs the {ROLES['admin']} role; you are logged in as "
                                           f"{ROLES[claims['role']]}."}, status_code=403)
        request.state.user = claims
        return await call_next(request)

    api.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["X-Frame-Timestamp"])

    @api.post("/api/auth/login")
    async def login(body: LoginBody):
        if users.count() == 0:
            raise HTTPException(503, "No user accounts are set up yet. On the server run: "
                                     "cd backend; ..\\venv\\Scripts\\python.exe -m tools.create_user")
        user = await asyncio.to_thread(users.authenticate, body.username, body.password)  # PBKDF2: ~0.2 s
        if user is None:
            raise HTTPException(401, "Wrong username or password.")
        token, claims = sessions.issue(user)
        return {"token": token, "expires_at": claims["exp"], "user": user.public()}

    @api.get("/api/auth/me")
    def me(request: Request):
        claims = request.state.user
        return {**user_from_claims(claims).public(), "expires_at": claims["exp"]}

    @api.post("/api/auth/logout", status_code=204)
    def logout(request: Request):
        sessions.revoke(request.state.user)
        return Response(status_code=204)

    def services() -> Services:
        return api.state.services

    def zone_or_404(zone_id: str) -> ZoneConfig:
        zone = services().zones.get(zone_id)
        if zone is None:
            raise HTTPException(404, f"unknown zone {zone_id!r}")
        return zone

    def camera_status(s: Services, zone_id: str) -> dict | None:
        try:
            return s.source.camera_status(zone_id)
        except Exception:  # e.g. Redis unreachable
            return None

    def latest_frame(s: Services, zone_id: str) -> ZoneFrame:
        """The last analysed frame, or (without a model) the camera's newest frame."""
        if s.worker and zone_id in s.worker.frames:
            return s.worker.frames[zone_id]
        try:
            burst = s.source.peek_burst(zone_id)
        except Exception:
            burst = None
        if burst is None:
            raise HTTPException(503, f"no frame received yet from the camera of zone {zone_id!r}")
        frames, timestamp = burst
        return ZoneFrame(frames[len(frames) // 2], None, timestamp)

    def analysed_frame(s: Services, zone_id: str) -> ZoneFrame:
        if s.worker is None:
            raise HTTPException(503, f"model not loaded: {s.model_error}")
        record = s.worker.frames.get(zone_id)
        if record is None:
            raise HTTPException(503, f"zone {zone_id!r} has not been analysed yet")
        return record

    @api.get("/health")
    def health():
        s = services()
        return {
            "status": "ok", "model_loaded": s.worker is not None, "model_error": s.model_error,
            "model_status": "loaded" if s.worker is not None else "model not loaded",
            "demo_mode": settings.demo_mode,
            "model": getattr(s.estimator, "info", None),  # which weights: source, dataset, test MAE
            "detector": getattr(s.detector, "info", None), "detector_error": s.detector_error,
            "hybrid": {"threshold": settings.hybrid_threshold, "rule": settings.hybrid_rule},
            "last_cycle_s": round(cycle_times[-1], 3) if cycle_times else None,
            "max_cycle_s_last_30": round(max(cycle_times), 3) if cycle_times else None,
            "zones": list(s.zones), "storage": s.store.kind, "frame_source": settings.frame_source,
            "last_error": s.worker.last_error if s.worker else None, "server_time": time.time(),
            "cameras": {zone_id: camera_status(s, zone_id) for zone_id in s.zones},
        }

    @api.get("/api/config")
    def config():
        return {
            "thresholds": {"yellow": settings.risk_yellow, "orange": settings.risk_orange, "red": settings.risk_red},
            "burst_interval_s": settings.burst_interval_s, "rolling_window_s": settings.rolling_window_s,
            "pressure_k": settings.pressure_k, "alert_cooldown_s": settings.alert_cooldown_s,
            "density_scale": heatmap_scale.to_dict(),
            "hybrid": {"threshold": settings.hybrid_threshold, "rule": settings.hybrid_rule},
            "grid": {"rows": settings.grid_rows, "cols": settings.grid_cols},
            "surge": {"trigger": settings.surge_trigger, "ratio": settings.surge_ratio,
                      "consecutive": settings.surge_consecutive, "history_s": settings.surge_history_s,
                      "min_level": settings.surge_min_level},
        }

    @api.get("/api/zones")
    def list_zones():
        s = services()
        video_zones = api.state.video.video_zones
        overrides = s.worker.overrides if s.worker else {}  # a test video's place on the exit map

        def node(z: ZoneConfig) -> str | None:
            return overrides[z.id].node if z.id in overrides else graph_node(z, s.router)

        return [{**zone_info(z, node(z)), "kind": "video" if z.id in video_zones else "camera"}
                for z in list(s.zones.values())]

    @api.get("/api/site-map")
    def site_map():
        """The exit graph for the Command Centre map: node positions from config/site_map.json
        (schematic until real coordinates are entered) and every walkway with its length."""
        try:
            data = json.loads(settings.site_map.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HTTPException(503, f"site map unavailable ({settings.site_map.name}): {exc}") from exc
        router = services().router
        placed = data.get("nodes", {})
        nodes = [{"id": n, "label": placed[n].get("label", n), "kind": placed[n].get("kind", "zone"),
                  "x": placed[n]["x"], "y": placed[n]["y"], "label_at": placed[n].get("label_at", "below"),
                  "exit": n in router.exits}
                 for n in router.graph.nodes if n in placed]
        edges = [{"a": a, "b": b, "metres": d["base_weight"]}
                 for a, b, d in router.graph.edges(data=True) if a < b and a in placed and b in placed]
        return {"schematic": data.get("schematic", True), "note": data.get("note", "Schematic map, not to scale"),
                "width": data.get("width", 1000), "height": data.get("height", 600), "nodes": nodes, "edges": edges,
                "unplaced": sorted(set(router.graph.nodes) - set(placed))}

    @api.get("/api/zones/{zone_id}/status")
    def zone_status(zone_id: str):
        zone = zone_or_404(zone_id)
        s = services()
        latest = s.worker.latest.get(zone.id) if s.worker else None
        return with_server_time(latest or {"zone_id": zone.id, "name": zone.name, "risk_level": "UNKNOWN"})

    @api.get("/api/zones/{zone_id}/history")
    def zone_history(zone_id: str, limit: int = 150):
        zone = zone_or_404(zone_id)
        try:
            readings = services().store.history(zone.id, min(max(limit, 1), 1000))
        except Exception as exc:  # database down
            raise HTTPException(503, f"history unavailable: {exc}") from exc
        return {"zone_id": zone.id, "server_time": time.time(), "readings": readings}

    @api.get("/api/zones/{zone_id}/exits")
    def zone_exits(zone_id: str, top_k: int = 3):
        zone = zone_or_404(zone_id)
        s = services()
        node = graph_node(zone, s.router)
        if node is None:
            raise HTTPException(404, f"zone {zone_id!r} is not on the exit graph; set graph_node in zones.json")
        return {"zone_id": zone.id, "graph_node": node, "exits": s.router.get_best_exits(node, top_k)}

    @api.get("/api/zones/{zone_id}/snapshot.jpg")
    def snapshot(zone_id: str):
        zone = zone_or_404(zone_id)
        record = latest_frame(services(), zone.id)
        size = display_size(record.frame)
        frame = record.frame if size == record.frame.shape[1::-1] else cv2.resize(
            record.frame, size, interpolation=cv2.INTER_AREA)
        return image_response(encode(frame, "jpeg"), "jpeg", record.timestamp)

    @api.get("/api/zones/{zone_id}/density.png")
    def density_overlay(zone_id: str):
        zone = zone_or_404(zone_id)
        s = services()
        record = analysed_frame(s, zone.id)
        analyzer = s.worker.analyzers.get(zone.id)
        if analyzer is None:
            raise HTTPException(404, f"zone {zone_id!r} was removed")
        h, w = record.frame.shape[:2]
        per_m2 = analyzer.geometry.cell_density(record.density, (w, h))  # fixed persons/m² scale
        overlay = density_rgba(per_m2, display_size(record.frame), heatmap_scale)
        return image_response(encode(overlay, "png"), "png", record.timestamp)

    @api.get("/api/zones/{zone_id}/detections.png")
    def detections_overlay(zone_id: str):
        zone = zone_or_404(zone_id)
        s = services()
        record = analysed_frame(s, zone.id)
        if record.boxes is None:
            raise HTTPException(503, f"person detection is not available: {s.detector_error}")
        h, w = record.frame.shape[:2]
        overlay = detections_rgba(record.boxes, display_size(record.frame), (w, h))
        return image_response(encode(overlay, "png"), "png", record.timestamp)

    gradcam_cache: dict[str, tuple[float, bytes]] = {}  # zone id -> (frame timestamp, PNG)
    gradcam_lock = threading.Lock()

    @api.get("/api/zones/{zone_id}/gradcam.png")
    def gradcam_overlay(zone_id: str):
        zone = zone_or_404(zone_id)
        s = services()
        record = analysed_frame(s, zone.id)
        if s.gradcam is None:
            raise HTTPException(503, "Grad-CAM is not available for this model")
        from app.csrnet_model import to_input

        with gradcam_lock:  # one backward pass at a time; each frame is explained once
            cached = gradcam_cache.get(zone.id)
            if cached is None or cached[0] != record.timestamp:
                cam = s.gradcam(to_input(record.frame, GRADCAM_WIDTH).to(s.gradcam.device))
                cached = (record.timestamp, encode(colour_overlay(cam, display_size(record.frame)), "png"))
                gradcam_cache[zone.id] = cached
        return image_response(cached[1], "png", record.timestamp)

    @api.post("/api/zones/{zone_id}/infer")
    async def infer(zone_id: str):
        zone = zone_or_404(zone_id)
        s = services()
        if s.worker is None:
            raise HTTPException(503, f"model not loaded: {s.model_error}")
        events = await asyncio.to_thread(s.worker.infer_now, zone.id)
        if events is None:
            raise HTTPException(409, f"no frames received yet for zone {zone_id!r}")
        if not events:
            raise HTTPException(500, s.worker.last_error or "inference failed")
        await emit(events)
        return with_server_time(s.worker.latest[zone.id])

    @api.get("/api/alerts/recent")
    def recent_alerts(limit: int = 20):
        try:
            return services().store.recent_alerts(min(max(limit, 1), 200))
        except Exception as exc:  # database down
            raise HTTPException(503, f"alerts unavailable: {exc}") from exc

    @api.post("/api/alerts/{alert_id}/ack")
    async def acknowledge_alert(alert_id: str, request: Request):
        """A security officer or the admin confirms they are handling the alert."""
        try:
            alert = await asyncio.to_thread(services().store.acknowledge, alert_id, request.state.user["sub"])
        except Exception as exc:  # database down
            raise HTTPException(503, f"could not acknowledge: {exc}") from exc
        if alert is None:
            raise HTTPException(404, f"no alert {alert_id!r}")
        await sio.emit("alert_ack", alert)
        return alert

    @api.get("/api/incidents")
    def incidents(limit: int = 50):
        try:
            return services().store.recent_incidents(min(max(limit, 1), 200))
        except Exception as exc:  # database down
            raise HTTPException(503, f"incidents unavailable: {exc}") from exc

    add_video_routes(api, lambda: api.state.video)
    add_report_routes(api, services, settings)

    @sio.event
    async def connect(sid, environ, auth=None):
        try:
            sessions.verify(auth.get("token") if isinstance(auth, dict) else None)
        except TokenError as exc:
            raise socketio.exceptions.ConnectionRefusedError(str(exc)) from None
        s = getattr(api.state, "services", None)
        for payload in list(s.worker.latest.values()) if s and s.worker else []:
            await sio.emit("density_update", with_server_time(payload), to=sid)

    app = socketio.ASGIApp(sio, other_asgi_app=api)
    app.api, app.sio, app.sessions = api, sio, sessions  # handles for tests
    return app


app = create_app()
