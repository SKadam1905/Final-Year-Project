import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.alert_dispatcher import AlertEvent, Delivery
from app.config import ZoneConfig
from app.risk_engine import RiskLevel, ZoneStatus
from app.store import MemoryStore, SqlStore, open_store

GREEN, YELLOW, ORANGE, RED = RiskLevel
BACKEND = Path(__file__).resolve().parents[1]


def reading(level, t, pressure=5.0, density=4.0, count=100.0, zone="Z"):
    return ZoneStatus(zone, t, count, density, 0.1, pressure, pressure, level, 15)


def iso(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()


def exercise(store):
    """ORANGE/RED from t=2 to t=6 makes one incident; RED again at t=10 opens a second."""
    store.sync_zones([ZoneConfig("Z", "Zone Z", "rtsp://cam/z", area_m2=100.0)])
    store.record_reading(reading(GREEN, 0))
    store.record_reading(reading(ORANGE, 2, pressure=4.5))
    store.record_reading(reading(RED, 4, pressure=6.5, density=5.5, count=300))
    alert = AlertEvent("Z", RED, "RED: Zone Z", 4.0, (
        Delivery("console", "stdout", True, 0.0), Delivery("sms", "+919999999999", False, 5.0, error="unverified")))
    store.record_alert(alert)
    store.record_reading(reading(ORANGE, 6, pressure=5.0))
    store.record_reading(reading(YELLOW, 8))
    store.record_reading(reading(RED, 10, pressure=7.0))

    second, first = store.recent_incidents()
    assert first["peak_level"] == "RED" and first["peak_pressure"] == 6.5 and first["peak_count"] == 300
    assert (first["started_at"], first["ended_at"]) == (iso(2), iso(8))
    assert second["ended_at"] is None and second["peak_level"] == "RED"

    alerts = store.recent_alerts()
    assert {a["channel"]: a["delivered"] for a in alerts} == {"console": True, "sms": False}
    assert {a["incident_id"] for a in alerts} == {first["id"]}
    assert all(a["risk_level"] == "RED" and a["timestamp"] == iso(4) for a in alerts)


def test_memory_store_incidents_and_alerts():
    exercise(MemoryStore())


def test_open_store_falls_back_to_memory():
    assert open_store("").kind == "memory"
    assert open_store("sqlite://").kind == "memory"  # reachable, but not migrated
    assert open_store("nosuchdriver://x").kind == "memory"


@pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"),
                    reason="set TEST_DATABASE_URL to a throwaway PostGIS database to run")
def test_sql_store_against_postgis():
    from alembic import command
    from alembic.config import Config

    url = os.environ["TEST_DATABASE_URL"]
    config = Config(str(BACKEND / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", url)
    command.downgrade(config, "base")
    command.upgrade(config, "head")
    exercise(SqlStore(url))
