"""Settings from environment variables (loaded from the repo-root .env) and zone config."""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
WEIGHTS_DIR = REPO_ROOT / "backend" / "ml" / "weights"

load_dotenv(REPO_ROOT / ".env")

# Alternative names accepted for some variables (other guides for this project use them).
ALIASES = {
    "CSRNET_WEIGHTS": ("MODEL_PATH",),
    "MONGODB_URI": ("MONGO_URI", "MONGODB_URL"),
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
    mongodb_uri: str  # MongoDB Atlas connection string; empty = keep data in memory
    mongodb_db: str
    zones_config: Path
    csrnet_weights: Path
    csrnet_allow_pickle: bool
    demo_mode: bool  # DEMO_MODE=1: also load demo / provenance-less weights (their counts are not real)
    device: str
    infer_max_width: int
    flow_max_width: int  # optical flow runs at this width (results/flow_speed_vs_accuracy.md)
    turbulence_weighting: str  # "frame" (every moving pixel) or "density" (weighted by the CSRNet density map)
    frame_source: str  # "local" (read in-process, no Docker) or "redis" (pushed by app.camera_manager)
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
    telegram_bot_token: str
    telegram_chat_id: str
    cors_origins: tuple[str, ...]
    upload_dir: Path  # uploaded videos and their analysis results (git-ignored)
    video_max_mb: int
    yolo_weights: Path | None = None  # YOLOv8 person detector; None = off (CSRNet counts only)
    yolo_imgsz: int = 640
    yolo_conf: float = 0.25
    hybrid_threshold: float = 60.0  # below this many people the detector's count is used
    hybrid_rule: str = "agree"  # "agree": both models below the threshold; "detector": YOLO alone decides
    grid_rows: int = 3  # zone-wise segmentation of each camera view
    grid_cols: int = 3
    secret_key: str = ""  # signs session tokens (app.auth); from .env SECRET_KEY
    secret_key_ephemeral: bool = False  # True: SECRET_KEY was missing, a random one is used until restart
    users_config: Path = REPO_ROOT / "config" / "users.json"
    site_map: Path = REPO_ROOT / "config" / "site_map.json"  # schematic node positions for the map
    session_hours: float = 8.0
    # Speed-surge trigger (app.surge; results/umn_speed_surge.md). Off by default: the ratio is still
    # computed and shown, but nothing is escalated unless surge_trigger is on.
    surge_trigger: bool = False
    surge_ratio: float = 1.10  # speed / median speed of the previous surge_history_s seconds
    surge_consecutive: int = 2  # readings in a row that must reach surge_ratio
    surge_history_s: float = 10.0
    surge_min_level: str = "YELLOW"  # density gate: only zones already at this level or above are escalated


@lru_cache
def get_settings() -> Settings:
    return Settings(
        redis_url=_env("REDIS_URL", "redis://localhost:6379/0"),
        mongodb_uri=_env("MONGODB_URI", ""),
        mongodb_db=_env("MONGODB_DB", "kumbhmela"),
        zones_config=repo_path(_env("ZONES_CONFIG", "config/zones.json")),
        csrnet_weights=repo_path(_env("CSRNET_WEIGHTS", "backend/ml/weights/csrnet_best.pth")),
        csrnet_allow_pickle=_env_bool("CSRNET_ALLOW_PICKLE"),
        demo_mode=_env_bool("DEMO_MODE"),
        device=_env("DEVICE", "auto"),
        infer_max_width=int(_env("INFER_MAX_WIDTH", "1280")),
        flow_max_width=int(_env("FLOW_MAX_WIDTH", "320")),
        turbulence_weighting=_env("TURBULENCE_WEIGHTING", "frame").lower(),
        frame_source=_env("FRAME_SOURCE", "local").lower(),  # redis needs Docker (Redis + camera manager)
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
        telegram_bot_token=_env("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=_env("TELEGRAM_CHAT_ID", ""),
        cors_origins=_env_list("CORS_ORIGINS", "*"),
        upload_dir=repo_path(_env("UPLOAD_DIR", "data/uploads")),
        video_max_mb=int(_env("VIDEO_MAX_MB", "500")),
        yolo_weights=_optional_path(_env("YOLO_WEIGHTS", "backend/ml/weights/yolo/yolov8s.pt")),
        yolo_imgsz=int(_env("YOLO_IMGSZ", "640")),
        yolo_conf=float(_env("YOLO_CONF", "0.25")),
        hybrid_threshold=float(_env("HYBRID_THRESHOLD", "60")),
        hybrid_rule=_env("HYBRID_RULE", "agree").lower(),
        grid_rows=int(_env("GRID_ROWS", "3")),
        grid_cols=int(_env("GRID_COLS", "3")),
        secret_key=_env("SECRET_KEY", "") or secrets.token_urlsafe(32),
        secret_key_ephemeral=not _env("SECRET_KEY", ""),
        users_config=repo_path(_env("USERS_CONFIG", "config/users.json")),
        session_hours=float(_env("SESSION_HOURS", "8")),
        site_map=repo_path(_env("SITE_MAP", "config/site_map.json")),
        surge_trigger=_env_bool("SURGE_TRIGGER"),
        surge_ratio=float(_env("SURGE_RATIO", "1.10")),
        surge_consecutive=int(_env("SURGE_CONSECUTIVE", "2")),
        surge_history_s=float(_env("SURGE_HISTORY_S", "10")),
        surge_min_level=_env("SURGE_MIN_LEVEL", "YELLOW").upper(),
    )


def _optional_path(value: str) -> Path | None:
    return None if value.lower() in ("off", "none", "0", "false") else repo_path(value)


def redact(source: str) -> str:
    """Hide credentials in camera URLs such as rtsp://user:pass@host/stream."""
    parts = urlsplit(source)
    if parts.username is None and parts.password is None:
        return source
    host = parts.hostname or ""
    if parts.port:
        host += f":{parts.port}"
    return urlunsplit((parts.scheme, f"***@{host}", parts.path, parts.query, parts.fragment))


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
    fixed_camera: bool = False  # true: never remove "camera motion" (a pole-mounted CCTV camera)


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
            fixed_camera=bool(raw.get("fixed_camera", False)),
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
