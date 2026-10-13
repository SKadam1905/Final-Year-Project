"""Storage for readings, incidents and alerts: MongoDB Atlas (MongoStore) or in memory.

An incident is a continuous stretch of ORANGE/RED readings for one zone: it opens on the
first such reading, tracks peak values, and closes when the zone drops below ORANGE.
MemoryStore keeps the same data in RAM so the system runs without a database (nothing
survives a restart).

Alerts are returned as one event per alert, with its deliveries (one per channel and
recipient) nested inside: the same dict the worker emits as the Socket.IO "alert" event.

MongoDB collections (database MONGODB_DB):
    zones           _id = zone id; name, camera_source (credentials hidden), area_m2, graph_node
    zone_readings   zone_id, ts, pressure (30 s mean), raw_pressure, density, turbulence, count, risk_level
Readings, incidents and alerts also store `model_kind`, the kind of weights that made them
(app.csrnet_model.describe_weights). Data made with demo or provenance-less weights is never
returned unless DEMO_MODE=1; tools/quarantine_demo_data.py moves old untagged demo data out.
    incidents       zone_id, started_at, ended_at (null while open), peak_level/pressure/density/count
    alerts          zone_id, incident_id, risk_level, message, triggered_at, delivered,
                    deliveries [{channel, recipient, delivered, latency_ms, provider_id, error}],
                    hotspot, gate_actions, acknowledged_by, acknowledged_at (null until acknowledged)
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import queue
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol, Sequence

from app.alert_dispatcher import AlertEvent, Delivery
from app.config import ZoneConfig, redact
from app.risk_engine import RiskLevel, ZoneStatus

log = logging.getLogger(__name__)

DEMO_KINDS = ("demo", "untrained")  # model kinds whose readings are not real counts


def to_datetime(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _utc(value: datetime) -> datetime:
    """Datetimes from the database are UTC; a naive one must not be read as local time."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _iso(value: datetime | float | None) -> str | None:
    """ISO 8601 in UTC."""
    if value is None:
        return None
    return (to_datetime(value) if isinstance(value, (int, float)) else _utc(value)).isoformat()


class Store(Protocol):
    kind: str

    def sync_zones(self, zones: Sequence[ZoneConfig]) -> None: ...
    def record_reading(self, status: ZoneStatus) -> None: ...
    def record_alert(self, event: AlertEvent) -> dict: ...
    def acknowledge(self, alert_id: str, username: str, now: float | None = None) -> dict | None: ...
    def recent_alerts(self, limit: int = 20) -> list[dict]: ...
    def recent_incidents(self, limit: int = 50) -> list[dict]: ...
    def history(self, zone_id: str, limit: int = 150) -> list[dict]: ...
    def alert_history(self, zone_id: str | None = None, levels: Sequence[str] = (), since: float | None = None,
                      until: float | None = None, limit: int = 500) -> list[dict]: ...
    def incident_history(self, zone_id: str | None = None, levels: Sequence[str] = (), since: float | None = None,
                         until: float | None = None, limit: int = 500) -> list[dict]: ...
    def readings_between(self, since: float | None = None, until: float | None = None,
                         zone_id: str | None = None, limit: int = 50_000) -> list[dict]: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


@dataclass
class _Incident:
    """An incident while it is open (both stores keep the open ones in memory)."""

    id: int | str
    zone_id: str
    started_at: float
    ended_at: float | None
    peak_level: RiskLevel
    peak_pressure: float
    peak_density: float
    peak_count: float


def update_peaks(incident: _Incident, status: ZoneStatus) -> None:
    incident.peak_level = max(incident.peak_level, status.level)
    incident.peak_pressure = max(incident.peak_pressure, status.smoothed_pressure)
    incident.peak_density = max(incident.peak_density, status.density)
    incident.peak_count = max(incident.peak_count, status.count)


def incident_dict(incident: _Incident) -> dict:
    return {
        "id": incident.id, "zone_id": incident.zone_id, "started_at": _iso(incident.started_at),
        "ended_at": _iso(incident.ended_at), "peak_level": str(incident.peak_level),
        "peak_pressure": incident.peak_pressure, "peak_density": incident.peak_density,
        "peak_count": incident.peak_count,
    }


def _in_range(ts: float, since: float | None, until: float | None) -> bool:
    return (since is None or ts >= since) and (until is None or ts <= until)


def _overlaps(incident: _Incident, since: float | None, until: float | None) -> bool:
    """An incident counts for a time range if any part of it falls inside."""
    started_in_time = until is None or incident.started_at <= until
    still_on = incident.ended_at is None or since is None or incident.ended_at >= since
    return started_in_time and still_on


def delivery_dict(d: Delivery) -> dict:
    return {"channel": d.channel, "recipient": d.recipient, "delivered": d.ok, "latency_ms": d.latency_ms,
            "provider_id": d.provider_id, "error": d.error}


def alert_dict(alert_id: int | str | None, event: AlertEvent, incident_id: int | str | None) -> dict:
    """An alert as served by GET /api/alerts/recent and emitted as the "alert" event."""
    deliveries = [delivery_dict(d) for d in event.deliveries]
    return {
        "id": alert_id, "zone_id": event.zone_id, "risk_level": str(event.level), "message": event.text,
        "timestamp": event.triggered_at, "incident_id": incident_id,
        "delivered": any(d["delivered"] for d in deliveries), "deliveries": deliveries,
        "hotspot": event.hotspot, "gate_actions": list(event.actions), "reason": event.reason,
        "acknowledged_by": None, "acknowledged_at": None,
    }


def reading_dict(timestamp: float, status: ZoneStatus) -> dict:
    """One point of a zone's history; field names match the "density_update" event."""
    return {
        "timestamp": timestamp, "pressure": status.smoothed_pressure, "raw_pressure": status.pressure,
        "density": status.density, "turbulence": status.turbulence, "count": status.count,
        "risk_level": str(status.level), "count_method": status.count_method,
        "density_count": status.density_count, "detection_count": status.detection_count,
        "hotspot": status.hotspot, "surge_ratio": status.surge_ratio, "escalation": status.escalation,
    }


class MemoryStore:
    kind = "memory"

    def __init__(self, max_items: int = 1000, max_readings: int = 5000) -> None:
        self._incidents: deque[_Incident] = deque(maxlen=max_items)
        self._open: dict[str, _Incident] = {}
        self._alerts: deque[dict] = deque(maxlen=max_items)
        self._incident_ids = itertools.count(1)
        self._alert_ids = itertools.count(1)
        self.readings: deque[ZoneStatus] = deque(maxlen=max_readings)

    def sync_zones(self, zones: Sequence[ZoneConfig]) -> None:
        pass

    def record_reading(self, status: ZoneStatus) -> None:
        self.readings.append(status)
        incident = self._open.get(status.zone_id)
        if status.level >= RiskLevel.ORANGE:
            if incident is None:
                incident = _Incident(next(self._incident_ids), status.zone_id, status.timestamp, None, status.level,
                                     status.smoothed_pressure, status.density, status.count)
                self._open[status.zone_id] = incident
                self._incidents.append(incident)
            else:
                update_peaks(incident, status)
        elif incident is not None:
            incident.ended_at = status.timestamp
            del self._open[status.zone_id]

    def record_alert(self, event: AlertEvent) -> dict:
        incident = self._open.get(event.zone_id)
        alert = alert_dict(next(self._alert_ids), event, incident.id if incident else None)
        self._alerts.append(alert)
        return alert

    def acknowledge(self, alert_id: str, username: str, now: float | None = None) -> dict | None:
        """Mark an alert as handled by `username`; the first acknowledgement stands. None if unknown."""
        alert = next((a for a in self._alerts if str(a["id"]) == str(alert_id)), None)
        if alert is not None and alert["acknowledged_at"] is None:
            alert["acknowledged_by"], alert["acknowledged_at"] = username, time.time() if now is None else now
        return alert

    def recent_alerts(self, limit: int = 20) -> list[dict]:
        return list(reversed(self._alerts))[:limit]

    def recent_incidents(self, limit: int = 50) -> list[dict]:
        newest = sorted(self._incidents, key=lambda i: i.started_at, reverse=True)[:limit]
        return [incident_dict(i) for i in newest]

    def history(self, zone_id: str, limit: int = 150) -> list[dict]:
        readings = [r for r in self.readings if r.zone_id == zone_id][-limit:]
        return [reading_dict(r.timestamp, r) for r in readings]

    def alert_history(self, zone_id=None, levels=(), since=None, until=None, limit=500) -> list[dict]:
        """Alerts matching the filters, newest first."""
        found = [a for a in reversed(self._alerts) if (zone_id is None or a["zone_id"] == zone_id)
                 and (not levels or a["risk_level"] in levels) and _in_range(a["timestamp"], since, until)]
        return found[:limit]

    def incident_history(self, zone_id=None, levels=(), since=None, until=None, limit=500) -> list[dict]:
        """ORANGE/RED periods overlapping the time range, newest first."""
        found = [i for i in sorted(self._incidents, key=lambda i: i.started_at, reverse=True)
                 if (zone_id is None or i.zone_id == zone_id) and (not levels or str(i.peak_level) in levels)
                 and _overlaps(i, since, until)]
        return [incident_dict(i) for i in found[:limit]]

    def readings_between(self, since=None, until=None, zone_id=None, limit=50_000) -> list[dict]:
        """Readings in the time range, oldest first, each with its zone_id (only those still kept in memory)."""
        found = [r for r in self.readings if (zone_id is None or r.zone_id == zone_id)
                 and _in_range(r.timestamp, since, until)]
        return [{"zone_id": r.zone_id, **reading_dict(r.timestamp, r)} for r in found[-limit:]]

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class MongoStore:
    """MongoDB (Atlas or local) through pymongo.

    Writes never block the inference loop: record_reading and record_alert queue them for one
    background thread and return at once, so a slow or dropped Wi-Fi link to Atlas delays the
    database, not the dashboard. Writes are applied in order. While Atlas is unreachable a
    write is retried every `retry_s` (it has a client-made _id, so a retry never duplicates
    it); one the server rejects (e.g. storage quota full) is logged and dropped. Open
    incidents are tracked in memory (loaded at start-up), so an alert gets its incident id
    without a database round trip. Reads (history, recent alerts/incidents) query the
    database directly.
    """

    kind = "mongodb"
    model_kind: str | None = None  # kind of the weights making the readings; set once the model has loaded

    def __init__(self, uri: str, db_name: str, client=None, timeout_ms: int = 5000, retry_s: float = 2.0,
                 max_queued: int = 50_000, show_demo: bool = False) -> None:
        self.show_demo = show_demo  # DEMO_MODE=1: also return data made with demo or provenance-less weights
        from bson import ObjectId
        from pymongo import ASCENDING, DESCENDING, MongoClient

        self._object_id = ObjectId
        self.client = client or MongoClient(uri, tz_aware=True, appname="kumbh-crowd-safety",
                                            serverSelectionTimeoutMS=timeout_ms, connectTimeoutMS=timeout_ms,
                                            socketTimeoutMS=2 * timeout_ms)
        self.db = self.client[db_name]
        self.client.admin.command("ping")  # unreachable or wrong password: fail now, open_store falls back
        self.db.zone_readings.create_index([("zone_id", ASCENDING), ("ts", DESCENDING)])
        self.db.incidents.create_index([("zone_id", ASCENDING), ("ended_at", ASCENDING)])
        self.db.incidents.create_index([("started_at", DESCENDING)])
        self.db.alerts.create_index([("triggered_at", DESCENDING)])
        self._open = {doc["zone_id"]: self._incident(doc)
                      for doc in self.db.incidents.find(self._visible({"ended_at": None}))}

        self.retry_s = retry_s
        self._queue: queue.Queue[Callable[[], object] | None] = queue.Queue(max_queued)
        self._unreachable = False
        self._alert_cache: OrderedDict[str, dict] = OrderedDict()  # recent alerts, for acknowledging at once
        self._writer = threading.Thread(target=self._write_loop, name="mongo-writer", daemon=True)
        self._writer.start()

    @staticmethod
    def _incident(doc: dict) -> _Incident:
        ended = doc.get("ended_at")
        return _Incident(str(doc["_id"]), doc["zone_id"], _utc(doc["started_at"]).timestamp(),
                         _utc(ended).timestamp() if ended else None, RiskLevel[doc["peak_level"]],
                         doc["peak_pressure"], doc["peak_density"], doc["peak_count"])

    # Writes -----------------------------------------------------------------------------

    def _submit(self, write: Callable[[], object]) -> None:
        try:
            self._queue.put_nowait(write)
        except queue.Full:
            log.warning("MongoDB write queue full (%d); dropping a write", self._queue.maxsize)

    def _write_loop(self) -> None:
        from pymongo.errors import ConnectionFailure, DuplicateKeyError, PyMongoError

        while (write := self._queue.get()) is not None:
            while True:
                try:
                    write()
                    if self._unreachable:
                        log.warning("MongoDB reachable again; %d writes queued", self._queue.qsize())
                        self._unreachable = False
                    break
                except DuplicateKeyError:  # a retried insert that had already landed
                    break
                except ConnectionFailure as exc:  # network down, Atlas paused, timeout: keep and retry
                    if not self._unreachable:
                        log.warning("MongoDB unreachable (%s); retrying every %.0f s", exc, self.retry_s)
                        self._unreachable = True
                    time.sleep(self.retry_s)
                except PyMongoError as exc:  # rejected by the server, e.g. storage quota full
                    log.warning("MongoDB rejected a write, dropped: %s", exc)
                    break
            self._queue.task_done()
        self._queue.task_done()

    def sync_zones(self, zones: Sequence[ZoneConfig]) -> None:
        for z in zones:
            self.db.zones.replace_one({"_id": z.id}, {
                "name": z.name, "camera_source": redact(z.source), "area_m2": z.area_m2,
                "calibrated": z.calibration is not None, "graph_node": z.graph_node,
            }, upsert=True)

    def record_reading(self, status: ZoneStatus) -> None:
        doc = {"_id": self._object_id(), "zone_id": status.zone_id, "ts": to_datetime(status.timestamp),
               **reading_dict(status.timestamp, status), "model_kind": self.model_kind}
        del doc["timestamp"]
        self._submit(lambda: self.db.zone_readings.insert_one(doc))

        incident = self._open.get(status.zone_id)
        if status.level >= RiskLevel.ORANGE:
            if incident is None:
                oid = self._object_id()
                incident = _Incident(str(oid), status.zone_id, status.timestamp, None, status.level,
                                     status.smoothed_pressure, status.density, status.count)
                self._open[status.zone_id] = incident
                new = {"_id": oid, "zone_id": status.zone_id, "started_at": to_datetime(status.timestamp),
                       "ended_at": None, **self._peaks(incident), "model_kind": self.model_kind}
                self._submit(lambda: self.db.incidents.insert_one(new))
            else:
                update_peaks(incident, status)
                oid, peaks = self._object_id(incident.id), self._peaks(incident)
                self._submit(lambda: self.db.incidents.update_one({"_id": oid}, {"$set": peaks}))
        elif incident is not None:
            del self._open[status.zone_id]
            oid, ended = self._object_id(incident.id), to_datetime(status.timestamp)
            self._submit(lambda: self.db.incidents.update_one({"_id": oid}, {"$set": {"ended_at": ended}}))

    @staticmethod
    def _peaks(incident: _Incident) -> dict:
        return {"peak_level": str(incident.peak_level), "peak_pressure": incident.peak_pressure,
                "peak_density": incident.peak_density, "peak_count": incident.peak_count}

    def record_alert(self, event: AlertEvent) -> dict:
        incident = self._open.get(event.zone_id)
        oid = self._object_id()
        alert = alert_dict(str(oid), event, incident.id if incident else None)
        doc = {"_id": oid, "zone_id": event.zone_id,
               "incident_id": self._object_id(incident.id) if incident else None,
               "risk_level": alert["risk_level"], "message": event.text, "triggered_at": to_datetime(event.triggered_at),
               "delivered": alert["delivered"], "deliveries": alert["deliveries"], "model_kind": self.model_kind,
               "hotspot": alert["hotspot"], "gate_actions": alert["gate_actions"], "reason": alert["reason"],
               "acknowledged_by": None, "acknowledged_at": None}
        self._submit(lambda: self.db.alerts.insert_one(doc))
        self._alert_cache[alert["id"]] = alert
        while len(self._alert_cache) > 500:
            self._alert_cache.popitem(last=False)
        return alert

    def acknowledge(self, alert_id: str, username: str, now: float | None = None) -> dict | None:
        """Mark an alert as handled. The update goes through the write queue, after the alert's own
        insert, so acknowledging an alert that is still being saved works too."""
        from bson.errors import InvalidId

        alert = self._alert_cache.get(alert_id)
        if alert is None:
            try:
                doc = self.db.alerts.find_one(self._visible({"_id": self._object_id(alert_id)}))
            except InvalidId:
                return None
            if doc is None:
                return None
            alert = self._alert(doc)
        if alert["acknowledged_at"] is None:
            when = time.time() if now is None else now
            alert["acknowledged_by"], alert["acknowledged_at"] = username, when
            oid, fields = self._object_id(alert_id), {"acknowledged_by": username, "acknowledged_at": to_datetime(when)}
            self._submit(lambda: self.db.alerts.update_one({"_id": oid, "acknowledged_at": None}, {"$set": fields}))
        return alert

    def flush(self) -> None:
        """Wait until every queued write has been applied (or dropped)."""
        self._queue.join()

    def close(self, timeout_s: float = 10.0) -> None:
        """Finish queued writes (giving up after `timeout_s` if Atlas is unreachable), then disconnect."""
        with contextlib.suppress(queue.Full):
            self._queue.put(None, timeout=timeout_s)
        self._writer.join(timeout_s)
        if self._writer.is_alive():
            log.warning("MongoDB: %d queued writes not saved before shutdown", self._queue.qsize())
        self.client.close()

    # Reads ------------------------------------------------------------------------------

    def _visible(self, query: dict | None = None) -> dict:
        """`query`, leaving out data made with demo or provenance-less weights unless show_demo.
        Untagged data (from before model_kind was stored) stays visible."""
        query = dict(query or {})
        if not self.show_demo:
            query["model_kind"] = {"$nin": list(DEMO_KINDS)}
        return query

    @staticmethod
    def _alert(d: dict) -> dict:
        return {"id": str(d["_id"]), "zone_id": d["zone_id"], "risk_level": d["risk_level"], "message": d["message"],
                "timestamp": _utc(d["triggered_at"]).timestamp(),
                "incident_id": str(d["incident_id"]) if d.get("incident_id") else None,
                "delivered": d["delivered"], "deliveries": d["deliveries"],
                "hotspot": d.get("hotspot"), "gate_actions": d.get("gate_actions", []), "reason": d.get("reason"),
                "acknowledged_by": d.get("acknowledged_by"),
                "acknowledged_at": _utc(d["acknowledged_at"]).timestamp() if d.get("acknowledged_at") else None}

    @staticmethod
    def _time_range(since: float | None, until: float | None) -> dict:
        query = {}
        if since is not None:
            query["$gte"] = to_datetime(since)
        if until is not None:
            query["$lte"] = to_datetime(until)
        return query

    def recent_alerts(self, limit: int = 20) -> list[dict]:
        docs = self.db.alerts.find(self._visible()).sort([("triggered_at", -1), ("_id", -1)]).limit(limit)
        return [self._alert(d) for d in docs]

    def alert_history(self, zone_id=None, levels=(), since=None, until=None, limit=500) -> list[dict]:
        query: dict = {}
        if zone_id:
            query["zone_id"] = zone_id
        if levels:
            query["risk_level"] = {"$in": list(levels)}
        if when := self._time_range(since, until):
            query["triggered_at"] = when
        docs = self.db.alerts.find(self._visible(query)).sort([("triggered_at", -1), ("_id", -1)]).limit(limit)
        return [self._alert(d) for d in docs]

    def incident_history(self, zone_id=None, levels=(), since=None, until=None, limit=500) -> list[dict]:
        query: dict = {}
        if zone_id:
            query["zone_id"] = zone_id
        if levels:
            query["peak_level"] = {"$in": list(levels)}
        if until is not None:
            query["started_at"] = {"$lte": to_datetime(until)}
        if since is not None:
            query["$or"] = [{"ended_at": None}, {"ended_at": {"$gte": to_datetime(since)}}]
        docs = self.db.incidents.find(self._visible(query)).sort([("started_at", -1), ("_id", -1)]).limit(limit)
        return [incident_dict(self._incident(d)) for d in docs]

    def readings_between(self, since=None, until=None, zone_id=None, limit=50_000) -> list[dict]:
        query: dict = {}
        if zone_id:
            query["zone_id"] = zone_id
        if when := self._time_range(since, until):
            query["ts"] = when
        docs = (self.db.zone_readings.find(self._visible(query), {"_id": 0, "model_kind": 0})
                .sort([("ts", 1), ("_id", 1)]).limit(limit))
        return [{"timestamp": _utc(d.pop("ts")).timestamp(), **d} for d in docs]

    def recent_incidents(self, limit: int = 50) -> list[dict]:
        docs = self.db.incidents.find(self._visible()).sort([("started_at", -1), ("_id", -1)]).limit(limit)
        return [incident_dict(self._incident(d)) for d in docs]

    def history(self, zone_id: str, limit: int = 150) -> list[dict]:
        docs = list(self.db.zone_readings.find(self._visible({"zone_id": zone_id}), {"_id": 0, "zone_id": 0, "model_kind": 0})
                    .sort([("ts", -1), ("_id", -1)]).limit(limit))
        return [{"timestamp": _utc(d.pop("ts")).timestamp(), **d} for d in reversed(docs)]


def open_store(mongodb_uri: str, db_name: str = "kumbhmela", timeout_ms: int = 5000, show_demo: bool = False) -> Store:
    """MongoStore if the database answers, otherwise MemoryStore (with a warning)."""
    if not mongodb_uri:
        return MemoryStore()
    try:
        return MongoStore(mongodb_uri, db_name, timeout_ms=timeout_ms, show_demo=show_demo)
    except Exception as exc:  # unreachable, wrong password, IP not on the Atlas access list, bad URI
        log.warning("MongoDB unavailable (%s); keeping readings, incidents and alerts in memory", exc)
        return MemoryStore()
