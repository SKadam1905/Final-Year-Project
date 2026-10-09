import os
import threading
from datetime import datetime, timezone

import mongomock
import pytest
from pymongo.errors import AutoReconnect, OperationFailure

from app.alert_dispatcher import AlertEvent, Delivery
from app.config import ZoneConfig
from app.risk_engine import RiskLevel, ZoneStatus
from app.store import MemoryStore, MongoStore, open_store

GREEN, YELLOW, ORANGE, RED = RiskLevel


def reading(level, t, pressure=5.0, density=4.0, count=100.0, zone="Z"):
    return ZoneStatus(zone, t, count, density, 0.1, pressure, pressure, level, 15)


def iso(t):
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()


def mongo_store(client=None, **kwargs) -> MongoStore:
    return MongoStore("", "kumbh_test", client=client or mongomock.MongoClient(tz_aware=True), **kwargs)


def exercise(store):
    """ORANGE/RED from t=2 to t=6 makes one incident; RED again at t=10 opens a second."""
    store.sync_zones([ZoneConfig("Z", "Zone Z", "rtsp://cam/z", area_m2=100.0)])
    store.record_reading(reading(GREEN, 0))
    store.record_reading(reading(ORANGE, 2, pressure=4.5))
    store.record_reading(reading(RED, 4, pressure=6.5, density=5.5, count=300))
    alert = AlertEvent("Z", RED, "RED: Zone Z", 4.0, (
        Delivery("console", "stdout", True, 0.0),
        Delivery("telegram", "1234", False, 5.0, error="chat not found")))
    stored = store.record_alert(alert)
    store.record_reading(reading(ORANGE, 6, pressure=5.0))
    store.record_reading(reading(YELLOW, 8))
    store.record_reading(reading(RED, 10, pressure=7.0))
    later = store.record_alert(AlertEvent("Z", RED, "RED again", 10.0, (Delivery("console", "stdout", True, 0.0),)))
    store.flush()

    second, first = store.recent_incidents()
    assert first["peak_level"] == "RED" and first["peak_pressure"] == 6.5 and first["peak_count"] == 300
    assert (first["started_at"], first["ended_at"]) == (iso(2), iso(8))
    assert second["ended_at"] is None and second["peak_level"] == "RED"

    # One event per alert, newest first, deliveries nested; identical to what record_alert returned.
    assert store.recent_alerts() == [later, stored]
    assert store.recent_alerts(limit=1) == [later]
    assert stored["id"] is not None and stored["id"] != later["id"]
    assert (stored["zone_id"], stored["risk_level"], stored["timestamp"]) == ("Z", "RED", 4.0)
    assert stored["incident_id"] == first["id"] and later["incident_id"] == second["id"]
    assert stored["delivered"] and [(d["channel"], d["delivered"]) for d in stored["deliveries"]] == \
        [("console", True), ("telegram", False)]
    assert stored["deliveries"][1]["error"] == "chat not found"

    history = store.history("Z", limit=3)
    assert [h["timestamp"] for h in history] == [6.0, 8.0, 10.0]
    assert history[0]["pressure"] == 5.0 and history[0]["risk_level"] == "ORANGE"
    assert store.history("other-zone") == []


def test_memory_store_incidents_and_alerts():
    exercise(MemoryStore())


def test_mongo_store_incidents_and_alerts():
    store = mongo_store()
    exercise(store)
    store.close()


def test_mongo_store_hides_camera_passwords_and_keeps_peaks_per_incident():
    client = mongomock.MongoClient(tz_aware=True)
    store = mongo_store(client)
    store.sync_zones([ZoneConfig("Z", "Zone Z", "rtsp://admin:secret@10.0.0.5/live", area_m2=100.0)])
    store.record_reading(reading(ORANGE, 2, pressure=4.5))
    store.record_reading(reading(RED, 4, pressure=6.5))
    store.record_reading(reading(ORANGE, 6, pressure=5.0))  # lower again: peaks stay
    store.flush()
    db = client["kumbh_test"]
    assert db.zones.find_one({"_id": "Z"})["camera_source"] == "rtsp://***@10.0.0.5/live"
    (incident,) = db.incidents.find()
    assert (incident["peak_level"], incident["peak_pressure"], incident["ended_at"]) == ("RED", 6.5, None)
    assert db.zone_readings.count_documents({"zone_id": "Z"}) == 3


def test_mongo_store_continues_an_open_incident_after_a_restart():
    client = mongomock.MongoClient(tz_aware=True)
    before = mongo_store(client)
    before.record_reading(reading(RED, 2, pressure=6.5))
    before.close()

    after = mongo_store(client)
    after.record_reading(reading(ORANGE, 4, pressure=4.5))
    alert = after.record_alert(AlertEvent("Z", ORANGE, "ORANGE: Zone Z", 4.0, ()))
    after.record_reading(reading(GREEN, 6))
    after.flush()
    (incident,) = after.recent_incidents()
    assert alert["incident_id"] == incident["id"]
    assert (incident["started_at"], incident["ended_at"], incident["peak_level"]) == (iso(2), iso(6), "RED")
    after.close()


def test_mongo_writes_do_not_wait_for_the_database():
    gate = threading.Event()
    store = mongo_store()
    store._submit(gate.wait)  # the database "hangs" until the gate opens
    store.record_reading(reading(RED, 2))  # returns at once; queued behind the hung write
    alert = store.record_alert(AlertEvent("Z", RED, "RED: Zone Z", 2.0, ()))
    assert alert["id"] and alert["incident_id"]  # ids are made locally, no round trip
    assert store.history("Z") == []
    gate.set()
    store.flush()
    assert [h["risk_level"] for h in store.history("Z")] == ["RED"]
    assert store.recent_alerts() == [alert]
    store.close()


def test_mongo_store_retries_while_unreachable_without_duplicates_and_drops_rejected_writes():
    store = mongo_store(retry_s=0.01)
    collection = store.db.zone_readings
    attempts = []

    def lands_but_ack_is_lost():
        attempts.append(1)
        collection.insert_one({"_id": "r1", "n": 1})  # on the retry: DuplicateKeyError, it is already saved
        raise AutoReconnect("connection reset after the write")

    def over_quota():
        raise OperationFailure("you are over your space quota")

    store._submit(lands_but_ack_is_lost)
    store._submit(over_quota)
    store._submit(lambda: collection.insert_one({"_id": "r2", "n": 2}))
    store.flush()
    assert len(attempts) == 2  # first attempt failed, the retry hit the duplicate and counted as done
    assert sorted(d["_id"] for d in collection.find()) == ["r1", "r2"]  # the queue went on past the rejection
    store.close()


def test_open_store_falls_back_to_memory():
    assert open_store("").kind == "memory"
    assert open_store("postgresql://kumbh@localhost/kumbhmela").kind == "memory"  # old DATABASE_URL pasted in
    assert open_store("mongodb://127.0.0.1:9/", timeout_ms=200).kind == "memory"  # nothing listening


@pytest.mark.skipif(not os.getenv("TEST_MONGODB_URI"),
                    reason="set TEST_MONGODB_URI to a throwaway MongoDB (e.g. a separate Atlas database) to run")
def test_mongo_store_against_a_real_server():
    from pymongo import MongoClient

    client = MongoClient(os.environ["TEST_MONGODB_URI"], tz_aware=True, serverSelectionTimeoutMS=10000)
    client.drop_database("kumbh_pytest")
    store = MongoStore("", "kumbh_pytest", client=client)
    try:
        exercise(store)
    finally:
        store.flush()
        client.drop_database("kumbh_pytest")
        store.close()
