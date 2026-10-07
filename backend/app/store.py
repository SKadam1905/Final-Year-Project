"""Storage for readings, incidents and alerts: PostgreSQL (SqlStore) or in memory.

An incident is a continuous stretch of ORANGE/RED readings for one zone: it opens on the
first such reading, tracks peak values, and closes when the zone drops below ORANGE.
MemoryStore keeps the same data in RAM so the system runs before Docker/PostGIS is set up
(nothing survives a restart).
"""

from __future__ import annotations

import itertools
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, Sequence

from sqlalchemy import create_engine, inspect, select
from sqlalchemy.orm import Session, sessionmaker

from app.alert_dispatcher import AlertEvent
from app.config import ZoneConfig
from app.models import AlertLog, Incident, Zone, ZoneReading
from app.risk_engine import RiskLevel, ZoneStatus

log = logging.getLogger(__name__)

TABLES = {"zones", "zone_readings", "incidents", "alert_logs"}


def to_datetime(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _iso(value: datetime | float | None) -> str | None:
    """ISO 8601 in UTC, whatever time zone the database session uses."""
    if value is None:
        return None
    return (to_datetime(value) if isinstance(value, (int, float)) else value.astimezone(timezone.utc)).isoformat()


class Store(Protocol):
    kind: str

    def sync_zones(self, zones: Sequence[ZoneConfig]) -> None: ...
    def record_reading(self, status: ZoneStatus) -> None: ...
    def record_alert(self, event: AlertEvent) -> None: ...
    def recent_alerts(self, limit: int = 20) -> list[dict]: ...
    def recent_incidents(self, limit: int = 50) -> list[dict]: ...


@dataclass
class _Incident:
    """In-memory twin of the Incident table row."""

    id: int
    zone_id: str
    started_at: float
    ended_at: float | None
    peak_level: RiskLevel
    peak_pressure: float
    peak_density: float
    peak_count: float


def update_peaks(incident: _Incident | Incident, status: ZoneStatus) -> None:
    incident.peak_level = max(incident.peak_level, status.level)
    incident.peak_pressure = max(incident.peak_pressure, status.smoothed_pressure)
    incident.peak_density = max(incident.peak_density, status.density)
    incident.peak_count = max(incident.peak_count, status.count)


def incident_dict(incident: _Incident | Incident) -> dict:
    return {
        "id": incident.id, "zone_id": incident.zone_id, "started_at": _iso(incident.started_at),
        "ended_at": _iso(incident.ended_at), "peak_level": str(incident.peak_level),
        "peak_pressure": incident.peak_pressure, "peak_density": incident.peak_density,
        "peak_count": incident.peak_count,
    }


def alert_dicts(event: AlertEvent, incident_id: int | None) -> list[dict]:
    """One row per delivery attempt (a channel/recipient pair)."""
    return [{
        "zone_id": event.zone_id, "incident_id": incident_id, "timestamp": _iso(event.triggered_at),
        "risk_level": str(event.level), "message": event.text, "channel": d.channel, "recipient": d.recipient,
        "delivered": d.ok, "latency_ms": d.latency_ms, "provider_id": d.provider_id, "error": d.error,
    } for d in event.deliveries]


class MemoryStore:
    kind = "memory"

    def __init__(self, max_items: int = 1000) -> None:
        self._incidents: deque[_Incident] = deque(maxlen=max_items)
        self._open: dict[str, _Incident] = {}
        self._alerts: deque[dict] = deque(maxlen=max_items)
        self._ids = itertools.count(1)
        self.readings: deque[ZoneStatus] = deque(maxlen=max_items)

    def sync_zones(self, zones: Sequence[ZoneConfig]) -> None:
        pass

    def record_reading(self, status: ZoneStatus) -> None:
        self.readings.append(status)
        incident = self._open.get(status.zone_id)
        if status.level >= RiskLevel.ORANGE:
            if incident is None:
                incident = _Incident(next(self._ids), status.zone_id, status.timestamp, None, status.level,
                                     status.smoothed_pressure, status.density, status.count)
                self._open[status.zone_id] = incident
                self._incidents.append(incident)
            else:
                update_peaks(incident, status)
        elif incident is not None:
            incident.ended_at = status.timestamp
            del self._open[status.zone_id]

    def record_alert(self, event: AlertEvent) -> None:
        incident = self._open.get(event.zone_id)
        self._alerts.extend(alert_dicts(event, incident.id if incident else None))

    def recent_alerts(self, limit: int = 20) -> list[dict]:
        return list(reversed(self._alerts))[:limit]

    def recent_incidents(self, limit: int = 50) -> list[dict]:
        newest = sorted(self._incidents, key=lambda i: i.started_at, reverse=True)[:limit]
        return [incident_dict(i) for i in newest]


class SqlStore:
    """PostgreSQL + PostGIS through the models in app.models (create the tables with Alembic)."""

    kind = "postgres"

    def __init__(self, database_url: str) -> None:
        connect_args = {"connect_timeout": 3} if database_url.startswith("postgresql") else {}
        self.engine = create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)
        missing = TABLES - set(inspect(self.engine).get_table_names())
        if missing:
            raise RuntimeError(f"tables missing ({', '.join(sorted(missing))}); run `alembic upgrade head` in backend/")
        self._sessions = sessionmaker(self.engine, expire_on_commit=False)

    def sync_zones(self, zones: Sequence[ZoneConfig]) -> None:
        with self._sessions.begin() as s:
            for z in zones:
                s.merge(Zone(id=z.id, name=z.name, camera_source=z.source, area_m2=z.area_m2))

    @staticmethod
    def _open_incident(session: Session, zone_id: str) -> Incident | None:
        query = (select(Incident).where(Incident.zone_id == zone_id, Incident.ended_at.is_(None))
                 .order_by(Incident.started_at.desc()).limit(1))
        return session.scalar(query)

    def record_reading(self, status: ZoneStatus) -> None:
        ts = to_datetime(status.timestamp)
        with self._sessions.begin() as s:
            s.add(ZoneReading(zone_id=status.zone_id, ts=ts, count=status.count, density=status.density,
                              turbulence=status.turbulence, pressure=status.pressure,
                              smoothed_pressure=status.smoothed_pressure, level=status.level))
            incident = self._open_incident(s, status.zone_id)
            if status.level >= RiskLevel.ORANGE:
                if incident is None:
                    s.add(Incident(zone_id=status.zone_id, started_at=ts, peak_level=status.level,
                                   peak_pressure=status.smoothed_pressure, peak_density=status.density,
                                   peak_count=status.count))
                else:
                    update_peaks(incident, status)
            elif incident is not None:
                incident.ended_at = ts

    def record_alert(self, event: AlertEvent) -> None:
        now = datetime.now(timezone.utc)
        with self._sessions.begin() as s:
            incident = self._open_incident(s, event.zone_id)
            for d in event.deliveries:
                s.add(AlertLog(zone_id=event.zone_id, incident_id=incident.id if incident else None,
                               level=event.level, channel=d.channel, recipient=d.recipient,
                               status="sent" if d.ok else "failed", message=event.text,
                               triggered_at=to_datetime(event.triggered_at), sent_at=now if d.ok else None,
                               latency_ms=d.latency_ms, provider_id=d.provider_id, error=d.error))

    def recent_alerts(self, limit: int = 20) -> list[dict]:
        with self._sessions() as s:
            rows = s.scalars(select(AlertLog).order_by(AlertLog.triggered_at.desc(), AlertLog.id.desc())
                             .limit(limit)).all()
            return [{
                "zone_id": r.zone_id, "incident_id": r.incident_id, "timestamp": _iso(r.triggered_at),
                "risk_level": str(r.level), "message": r.message, "channel": r.channel, "recipient": r.recipient,
                "delivered": r.status == "sent", "latency_ms": r.latency_ms, "provider_id": r.provider_id,
                "error": r.error,
            } for r in rows]

    def recent_incidents(self, limit: int = 50) -> list[dict]:
        with self._sessions() as s:
            rows = s.scalars(select(Incident).order_by(Incident.started_at.desc()).limit(limit)).all()
            return [incident_dict(r) for r in rows]


def open_store(database_url: str) -> Store:
    """SqlStore if the database is reachable and migrated, otherwise MemoryStore (with a warning)."""
    if not database_url:
        return MemoryStore()
    try:
        return SqlStore(database_url)
    except Exception as exc:  # server down, missing tables, bad URL, missing driver
        log.warning("database unavailable (%s); keeping readings, incidents and alerts in memory", exc)
        return MemoryStore()
