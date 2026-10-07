"""Initial schema: PostGIS, zones, zone_readings, incidents, alert_logs.

Written by hand to match app/models.py (autogenerate needs a live database);
tests/test_migrations.py checks that every model table and column is created.

Revision ID: 001_initial
Revises:
Create Date: 2026-10-08
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from geoalchemy2 import Geometry
from sqlalchemy.dialects import postgresql

revision: str = "001_initial"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Created once below; columns refer to it without re-creating it.
risk_level = postgresql.ENUM("GREEN", "YELLOW", "ORANGE", "RED", name="risk_level", create_type=False)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    op.execute("CREATE TYPE risk_level AS ENUM ('GREEN', 'YELLOW', 'ORANGE', 'RED')")

    op.create_table(
        "zones",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("camera_source", sa.String(500)),
        sa.Column("boundary", Geometry("POLYGON", srid=4326, spatial_index=False)),
        sa.Column("area_m2", sa.Float()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("idx_zones_boundary", "zones", ["boundary"], postgresql_using="gist")

    op.create_table(
        "zone_readings",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("zone_id", sa.String(64), sa.ForeignKey("zones.id"), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("count", sa.Float(), nullable=False),
        sa.Column("density", sa.Float(), nullable=False),
        sa.Column("turbulence", sa.Float(), nullable=False),
        sa.Column("pressure", sa.Float(), nullable=False),
        sa.Column("smoothed_pressure", sa.Float(), nullable=False),
        sa.Column("level", risk_level, nullable=False),
    )
    op.create_index("ix_zone_readings_zone_ts", "zone_readings", ["zone_id", "ts"])

    op.create_table(
        "incidents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("zone_id", sa.String(64), sa.ForeignKey("zones.id"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
        sa.Column("peak_level", risk_level, nullable=False),
        sa.Column("peak_pressure", sa.Float(), nullable=False),
        sa.Column("peak_density", sa.Float(), nullable=False),
        sa.Column("peak_count", sa.Float(), nullable=False),
        sa.Column("notes", sa.Text()),
    )
    op.create_index("ix_incidents_zone_id", "incidents", ["zone_id"])

    op.create_table(
        "alert_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("zone_id", sa.String(64), sa.ForeignKey("zones.id"), nullable=False),
        sa.Column("incident_id", sa.Integer(), sa.ForeignKey("incidents.id")),
        sa.Column("level", risk_level, nullable=False),
        sa.Column("channel", sa.String(32), nullable=False),
        sa.Column("recipient", sa.String(100), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("triggered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True)),
        sa.Column("latency_ms", sa.Float()),
        sa.Column("provider_id", sa.String(64)),
        sa.Column("error", sa.Text()),
    )
    op.create_index("ix_alert_logs_zone_id", "alert_logs", ["zone_id"])


def downgrade() -> None:
    op.drop_table("alert_logs")
    op.drop_table("incidents")
    op.drop_table("zone_readings")
    op.drop_index("idx_zones_boundary", table_name="zones")
    op.drop_table("zones")
    op.execute("DROP TYPE risk_level")
