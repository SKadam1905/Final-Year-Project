"""Move readings, incidents and alerts made with demo or provenance-less weights out of the live
MongoDB collections into <collection>_demo_backup. Nothing is deleted outright: a document is
removed from the live collection only once it is in the backup.

    cd backend
    ..\\venv\\Scripts\\python.exe -m tools.quarantine_demo_data                                          # dry run
    ..\\venv\\Scripts\\python.exe -m tools.quarantine_demo_data --legacy-before 2026-10-09T00:00:00Z         # dry run
    ..\\venv\\Scripts\\python.exe -m tools.quarantine_demo_data --legacy-before 2026-10-09T00:00:00Z --apply

Demo data is data whose `model_kind` is "demo" or "untrained" (stored since 2026-10-09). Older
data has no tag: --legacy-before also takes untagged data from before that UTC time, e.g. before
the first real weights went live. The dry run prints counts and date ranges, plus how many of the
untagged readings it would move do NOT show the demo model's constant count (700.05 people):
that should be 0, or the cut-off is wrong.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from app.config import get_settings
from app.store import DEMO_KINDS

TIME_FIELDS = {"zone_readings": "ts", "incidents": "started_at", "alerts": "triggered_at"}
DEMO_COUNT = (699.5, 700.5)  # scripts/make_demo.py's model counts 700.05 people on every frame


def demo_query(time_field: str, legacy_before: datetime | None) -> dict:
    tagged = {"model_kind": {"$in": list(DEMO_KINDS)}}
    if legacy_before is None:
        return tagged
    return {"$or": [tagged, {"model_kind": {"$exists": False}, time_field: {"$lt": legacy_before}}]}


def summarise(db, legacy_before: datetime | None) -> dict[str, dict]:
    """Per collection: how many documents would move, and their first and last times."""
    out = {}
    for name, field in TIME_FIELDS.items():
        query = demo_query(field, legacy_before)
        first = list(db[name].find(query, {field: 1}).sort(field, 1).limit(1))
        last = list(db[name].find(query, {field: 1}).sort(field, -1).limit(1))
        out[name] = {"count": db[name].count_documents(query),
                     "first": first[0][field] if first else None, "last": last[0][field] if last else None,
                     "kept": db[name].count_documents({}) - db[name].count_documents(query)}
    if legacy_before is not None:
        untagged = {"model_kind": {"$exists": False}, "ts": {"$lt": legacy_before}}
        out["zone_readings"]["untagged_not_demo_count"] = db.zone_readings.count_documents(
            {**untagged, "count": {"$not": {"$gte": DEMO_COUNT[0], "$lte": DEMO_COUNT[1]}}})
    return out


def move(db, name: str, query: dict, batch: int = 1000) -> int:
    """Copy the matching documents to <name>_demo_backup, then delete exactly those copied."""
    from pymongo.errors import BulkWriteError

    backup = db[f"{name}_demo_backup"]
    moved = 0
    while docs := list(db[name].find(query).limit(batch)):
        try:
            backup.insert_many(docs, ordered=False)
        except BulkWriteError as exc:  # already in the backup from an interrupted run: fine
            if any(e.get("code") != 11000 for e in exc.details.get("writeErrors", [])):
                raise
        ids = [d["_id"] for d in docs]
        saved = [d["_id"] for d in backup.find({"_id": {"$in": ids}}, {"_id": 1})]
        if len(saved) != len(ids):
            raise RuntimeError(f"{name}: only {len(saved)} of {len(ids)} documents reached the backup; stopped")
        db[name].delete_many({"_id": {"$in": saved}})
        moved += len(saved)
    return moved


def main(argv=None) -> None:
    from pymongo import MongoClient

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--legacy-before", help="UTC time, e.g. 2026-10-09T00:00:00Z: untagged data before it is demo data")
    p.add_argument("--apply", action="store_true", help="move the data (default: only show what would move)")
    args = p.parse_args(argv)
    legacy_before = (datetime.fromisoformat(args.legacy_before.replace("Z", "+00:00")).astimezone(timezone.utc)
                     if args.legacy_before else None)

    s = get_settings()
    if not s.mongodb_uri:
        raise SystemExit("MONGODB_URI is not set in .env: nothing to clean")
    db = MongoClient(s.mongodb_uri, tz_aware=True, serverSelectionTimeoutMS=10_000)[s.mongodb_db]
    for name, info in summarise(db, legacy_before).items():
        span = f"{info['first']:%Y-%m-%d %H:%M:%S} .. {info['last']:%Y-%m-%d %H:%M:%S} UTC" if info["count"] else "-"
        extra = (f"; untagged ones not counting 700.05: {info['untagged_not_demo_count']}"
                 if "untagged_not_demo_count" in info else "")
        print(f"{name}: {info['count']} to move ({span}), {info['kept']} kept{extra}")
    if not args.apply:
        print("dry run: nothing moved (add --apply)")
        return
    for name, field in TIME_FIELDS.items():
        print(f"{name}: moved {move(db, name, demo_query(field, legacy_before))} to {name}_demo_backup")


if __name__ == "__main__":
    main()
