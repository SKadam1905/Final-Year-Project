"""FastAPI + Socket.IO backend (plan Layer 4), running the inference loop every 2 s.

    cd backend
    uvicorn app.main:app --port 8000          # interactive docs: http://localhost:8000/docs

REST
    GET  /health                        status, model_loaded, zones, storage, frame source, cameras
    GET  /api/zones                     zone configs (camera passwords hidden)
    GET  /api/zones/{zone_id}/status    latest reading: density, turbulence, pressure, risk_level, count, exits
    GET  /api/zones/{zone_id}/exits     best exits now (Dijkstra over live crowd pressure)
    POST /api/zones/{zone_id}/infer     run inference now on the zone's newest frames
    GET  /api/alerts/recent             last 20 alerts (one row per channel and recipient)
    GET  /api/incidents                 last 50 incidents (ORANGE/RED periods)
Socket.IO (path /socket.io)
    density_update   {zone_id, name, density, turbulence, pressure, raw_pressure, risk_level, count, timestamp, exits}
    alert            {zone_id, risk_level, message, exits, timestamp, delivered}
New clients get the latest density_update of every zone as soon as they connect.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from contextlib import asynccontextmanager
from urllib.parse import urlsplit, urlunsplit

import socketio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, ZoneConfig, get_settings
from app.worker import Event, Services, build_services, graph_node

# Our modules log at INFO; libraries stay at WARNING (httpx would log Telegram URLs, which
# contain the bot token, at INFO).
logging.basicConfig(format="%(levelname)s:     %(name)s - %(message)s")
logging.getLogger("app").setLevel(logging.INFO)
log = logging.getLogger(__name__)


def redact(source: str) -> str:
    """Hide credentials in camera URLs such as rtsp://user:pass@host/stream."""
    parts = urlsplit(source)
    if parts.username is None and parts.password is None:
        return source
    host = parts.hostname or ""
    if parts.port:
        host += f":{parts.port}"
    return urlunsplit((parts.scheme, f"***@{host}", parts.path, parts.query, parts.fragment))


def zone_info(zone: ZoneConfig, node: str | None) -> dict:
    return {"id": zone.id, "name": zone.name, "source": redact(zone.source), "graph_node": node,
            "area_m2": zone.area_m2, "calibrated": zone.calibration is not None, "roi": zone.roi}


def create_app(settings: Settings | None = None, services_factory=build_services, run_loop: bool = True):
    settings = settings or get_settings()
    origins = list(settings.cors_origins)
    sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*" if origins == ["*"] else origins)

    async def emit(events: list[Event]) -> None:
        for name, payload in events:
            await sio.emit(name, payload)

    async def inference_loop(services: Services) -> None:
        while True:
            started = time.monotonic()
            try:
                await emit(await asyncio.to_thread(services.worker.step))
            except Exception:
                log.exception("inference loop error")
            await asyncio.sleep(max(0.0, settings.burst_interval_s - (time.monotonic() - started)))

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        services = services_factory(settings)
        api.state.services = services
        services.source.start()
        task = asyncio.create_task(inference_loop(services)) if run_loop and services.worker else None
        try:
            yield
        finally:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            services.source.stop()

    api = FastAPI(title="Kumbh Mela crowd-safety API", version="0.3.0", lifespan=lifespan)
    api.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

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

    @api.get("/health")
    def health():
        s = services()
        return {
            "status": "ok", "model_loaded": s.worker is not None, "model_error": s.model_error,
            "zones": list(s.zones), "storage": s.store.kind, "frame_source": settings.frame_source,
            "last_error": s.worker.last_error if s.worker else None,
            "cameras": {zone_id: camera_status(s, zone_id) for zone_id in s.zones},
        }

    @api.get("/api/zones")
    def list_zones():
        s = services()
        return [zone_info(z, graph_node(z, s.router)) for z in s.zones.values()]

    @api.get("/api/zones/{zone_id}/status")
    def zone_status(zone_id: str):
        zone = zone_or_404(zone_id)
        s = services()
        latest = s.worker.latest.get(zone.id) if s.worker else None
        return latest or {"zone_id": zone.id, "name": zone.name, "risk_level": "UNKNOWN"}

    @api.get("/api/zones/{zone_id}/exits")
    def zone_exits(zone_id: str, top_k: int = 3):
        zone = zone_or_404(zone_id)
        s = services()
        node = graph_node(zone, s.router)
        if node is None:
            raise HTTPException(404, f"zone {zone_id!r} is not on the exit graph; set graph_node in zones.json")
        return {"zone_id": zone.id, "graph_node": node, "exits": s.router.get_best_exits(node, top_k)}

    @api.post("/api/zones/{zone_id}/infer")
    async def infer(zone_id: str):
        zone = zone_or_404(zone_id)
        s = services()
        if s.worker is None:
            raise HTTPException(503, s.model_error)
        events = await asyncio.to_thread(s.worker.infer_now, zone.id)
        if events is None:
            raise HTTPException(409, f"no frames received yet for zone {zone_id!r}")
        if not events:
            raise HTTPException(500, s.worker.last_error or "inference failed")
        await emit(events)
        return s.worker.latest[zone.id]

    @api.get("/api/alerts/recent")
    def recent_alerts(limit: int = 20):
        return services().store.recent_alerts(min(max(limit, 1), 200))

    @api.get("/api/incidents")
    def incidents(limit: int = 50):
        return services().store.recent_incidents(min(max(limit, 1), 200))

    @sio.event
    async def connect(sid, environ, auth=None):
        s = getattr(api.state, "services", None)
        for payload in list(s.worker.latest.values()) if s and s.worker else []:
            await sio.emit("density_update", payload, to=sid)

    app = socketio.ASGIApp(sio, other_asgi_app=api)
    app.api, app.sio = api, sio  # handles for tests
    return app


app = create_app()
