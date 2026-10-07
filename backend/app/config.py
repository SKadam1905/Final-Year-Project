"""Settings from environment variables (loaded from the repo-root .env) and zone config."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
WEIGHTS_DIR = REPO_ROOT / "backend" / "ml" / "weights"

load_dotenv(REPO_ROOT / ".env")

# Alternative names accepted for some variables (other guides for this project use them).
ALIASES = {
    "CSRNET_WEIGHTS": ("MODEL_PATH",),
    "ALERT_SMS_TO": ("SECURITY_PHONE_NUMBERS",),
    "ALERT_COOLDOWN_S": ("ALERT_COOLDOWN_SECONDS",),
    "ROLLING_MAXLEN": ("ROLLING_WINDOW",),
    "RISK_YELLOW": ("THRESHOLD_YELLOW",),
    "RISK_ORANGE": ("THRESHOLD_ORANGE",),
    "RISK_RED": ("THRESHOLD_RED",),
}


def _env(name: str, default: str) -> str:
    for key in (name, *ALIASES.get(name, ())):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return default


def _env_list(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(item.strip() for item in _env(name, default).split(",") if item.strip())


def _env_bool(name: str, default: bool = False) -> bool:
    return _env(name, str(default)).lower() in ("1", "true", "yes", "on")


def repo_path(value: str | Path) -> Path:
    """Resolve a path from config relative to the repository root."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass(frozen=True)
class Settings:
    redis_url: str
    database_url: str
    zones_config: Path
    csrnet_weights: Path
    csrnet_allow_pickle: bool
    device: str
    infer_max_width: int
    frame_source: str  # "redis" (pushed by app.camera_manager) or "local" (read in-process)
    burst_size: int
    burst_interval_s: float
    rolling_window_s: float
    rolling_maxlen: int
    pressure_k: float
    risk_yellow: float
    risk_orange: float
    risk_red: float
    alert_channels: tuple[str, ...]
    alert_cooldown_s: float
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_from_number: str
    alert_sms_to: tuple[str, ...]
    telegram_bot_token: str
    telegram_chat_id: str
    cors_origins: tuple[str, ...]


@lru_cache
def get_settings() -> Settings:
    return Settings(
        redis_url=_env("REDIS_URL", "redis://localhost:6379/0"),
        database_url=_env(
            "DATABASE_URL", "postgresql+psycopg2://kumbh:kumbh_dev_password@localhost:5432/kumbhmela"
        ),
        zones_config=repo_path(_env("ZONES_CONFIG", "config/zones.json")),
        csrnet_weights=repo_path(_env("CSRNET_WEIGHTS", "backend/ml/weights/csrnet_best.pth")),
        csrnet_allow_pickle=_env_bool("CSRNET_ALLOW_PICKLE"),
        device=_env("DEVICE", "auto"),
        infer_max_width=int(_env("INFER_MAX_WIDTH", "1280")),
        frame_source=_env("FRAME_SOURCE", "redis").lower(),
        burst_size=int(_env("BURST_SIZE", "5")),
        burst_interval_s=float(_env("BURST_INTERVAL_S", "2")),
        rolling_window_s=float(_env("ROLLING_WINDOW_S", "30")),
        rolling_maxlen=int(_env("ROLLING_MAXLEN", "15")),
        pressure_k=float(_env("PRESSURE_K", "0.5")),
        risk_yellow=float(_env("RISK_YELLOW", "2.0")),
        risk_orange=float(_env("RISK_ORANGE", "4.0")),
        risk_red=float(_env("RISK_RED", "6.0")),
        alert_channels=tuple(c.lower() for c in _env_list("ALERT_CHANNEL", "console")),
        alert_cooldown_s=float(_env("ALERT_COOLDOWN_S", "600")),
        twilio_account_sid=_env("TWILIO_ACCOUNT_SID", ""),
        twilio_auth_token=_env("TWILIO_AUTH_TOKEN", ""),
        twilio_from_number=_env("TWILIO_FROM_NUMBER", ""),
        alert_sms_to=_env_list("ALERT_SMS_TO"),
        telegram_bot_token=_env("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=_env("TELEGRAM_CHAT_ID", ""),
        cors_origins=_env_list("CORS_ORIGINS", "*"),
    )


Point = tuple[float, float]


@dataclass(frozen=True)
class Calibration:
    """Floor markers: pixel positions and their measured ground positions in metres."""

    image_points: tuple[Point, ...]
    ground_points: tuple[Point, ...]


@dataclass(frozen=True)
class ZoneConfig:
    """One monitored zone = one camera view (see config/zones.json)."""

    id: str
    name: str
    source: str  # rtsp:// URL, or a video file path for offline testing
    roi: tuple[Point, ...] | None = None  # walkable-area polygon in camera pixels
    area_m2: float | None = None  # ground area of the ROI when there is no calibration
    calibration: Calibration | None = None
    graph_node: str | None = None  # node in the exit-routing graph (default: the zone id)


def _points(raw) -> tuple[Point, ...] | None:
    if raw is None:
        return None
    return tuple((float(x), float(y)) for x, y in raw)


def load_zones(path: str | Path) -> list[ZoneConfig]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    zones = []
    for raw in data["zones"]:
        cal = raw.get("calibration")
        source = raw["source"]
        if "://" not in source:
            source = str(repo_path(source))
        zone = ZoneConfig(
            id=raw["id"],
            name=raw.get("name", raw["id"]),
            source=source,
            roi=_points(raw.get("roi")),
            area_m2=raw.get("area_m2"),
            calibration=Calibration(_points(cal["image_points"]), _points(cal["ground_points"]))
            if cal
            else None,
            graph_node=raw.get("graph_node"),
        )
        if zone.area_m2 is None and zone.calibration is None:
            raise ValueError(
                f"zone {zone.id!r}: set 'area_m2' or 'calibration' so counts can be "
                "converted to persons/m²"
            )
        zones.append(zone)
    if len({z.id for z in zones}) != len(zones):
        raise ValueError(f"duplicate zone ids in {path}")
    return zones
