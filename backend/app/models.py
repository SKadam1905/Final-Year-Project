"""Database schema (PostgreSQL 16 + PostGIS 3.4): zones, readings, incidents, alert log.

Readings are kept for the thesis evaluation (crowd-history plots, latency analysis).
`AlertLog.triggered_at` vs `sent_at` gives the alert-delivery latency of plan §8.
"""

from __future__ import annotations

from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import BigInteger, DateTime, Enum, Float, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.risk_engine import RiskLevel

RiskLevelType = Enum(RiskLevel, name="risk_level")


class Base(DeclarativeBase):
    pass


class Zone(Base):
    __tablename__ = "zones"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    camera_source: Mapped[str | None] = mapped_column(String(500))
    # Zone outline on the site map (WGS84 lon/lat), used by the dashboard and ST_Contains.
    boundary = mapped_column(Geometry("POLYGON", srid=4326), nullable=True)
    area_m2: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    incidents: Mapped[list[Incident]] = relationship(back_populates="zone")


class ZoneReading(Base):
    __tablename__ = "zone_readings"
    __table_args__ = (Index("ix_zone_readings_zone_ts", "zone_id", "ts"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    zone_id: Mapped[str] = mapped_column(ForeignKey("zones.id"))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    count: Mapped[float] = mapped_column(Float)
    density: Mapped[float] = mapped_column(Float)  # persons/m²
    turbulence: Mapped[float] = mapped_column(Float)
    pressure: Mapped[float] = mapped_column(Float)
    smoothed_pressure: Mapped[float] = mapped_column(Float)
    level: Mapped[RiskLevel] = mapped_column(RiskLevelType)


class Incident(Base):
    """A continuous period during which a zone was ORANGE or RED."""

    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(primary_key=True)
    zone_id: Mapped[str] = mapped_column(ForeignKey("zones.id"), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    peak_level: Mapped[RiskLevel] = mapped_column(RiskLevelType)
    peak_pressure: Mapped[float] = mapped_column(Float)
    peak_density: Mapped[float] = mapped_column(Float)
    peak_count: Mapped[float] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)

    zone: Mapped[Zone] = relationship(back_populates="incidents")
    alerts: Mapped[list[AlertLog]] = relationship(back_populates="incident")


class AlertLog(Base):
    __tablename__ = "alert_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    zone_id: Mapped[str] = mapped_column(ForeignKey("zones.id"), index=True)
    incident_id: Mapped[int | None] = mapped_column(ForeignKey("incidents.id"))
    level: Mapped[RiskLevel] = mapped_column(RiskLevelType)
    channel: Mapped[str] = mapped_column(String(32))  # sms | telegram | console
    recipient: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(16))  # sent | failed
    message: Mapped[str] = mapped_column(Text)
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))  # threshold crossing
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    latency_ms: Mapped[float | None] = mapped_column(Float)
    provider_id: Mapped[str | None] = mapped_column(String(64))  # Twilio SID / Telegram message id
    error: Mapped[str | None] = mapped_column(Text)

    incident: Mapped[Incident | None] = relationship(back_populates="alerts")
