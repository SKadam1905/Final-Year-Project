"""Alert policy and delivery channels (plan Layer 3, §4.6).

Policy: alert when a zone's smoothed level reaches ORANGE or RED, then stay quiet for that
zone for the cooldown (10 min), except that an escalation (ORANGE -> RED) always fires. A level
raised by the speed-surge trigger (status.escalation) goes through the same policy, and its reason
is named in the message.
A zone that simply stays RED is not re-alerted every 2 s; it gets a reminder once the
cooldown has passed. Failed deliveries are reported in the alert (the dashboard shows them)
rather than retried every reading, which would flood the dashboard while Telegram is down.

Channels, chosen with ALERT_CHANNEL in .env (comma-separated): console, telegram, or all.
Messages stay plain ASCII (any console can print them) and short enough to read on a lock screen.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence

import httpx

from app.config import Settings
from app.gate_actions import place_name
from app.risk_engine import RiskLevel, ZoneStatus

log = logging.getLogger(__name__)

ACTIONS = {
    RiskLevel.ORANGE: "Hold inflow, open alternate exits.",
    RiskLevel.RED: "STOP inflow now, open all exits.",
}


@dataclass(frozen=True)
class Delivery:
    channel: str
    recipient: str
    ok: bool
    latency_ms: float
    provider_id: str | None = None
    error: str | None = None


class Channel(Protocol):
    name: str

    def send(self, text: str) -> list[Delivery]: ...


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1e3


class ConsoleChannel:
    name = "console"

    def send(self, text: str) -> list[Delivery]:
        print(f"[ALERT] {text}", flush=True)
        return [Delivery(self.name, "stdout", True, 0.0)]


class TelegramChannel:
    name = "telegram"

    def __init__(self, bot_token: str, chat_id: str, client: httpx.Client | None = None, timeout_s: float = 5.0,
                 connect_timeout_s: float = 3.0):
        self._token = bot_token
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self.chat_id = chat_id
        # Sending runs inside the inference cycle: on Wi-Fi without internet, give up quickly rather
        # than hold the live dashboard for long (Telegram itself answers in well under a second).
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout_s, connect=connect_timeout_s))

    def send(self, text: str) -> list[Delivery]:
        started = time.perf_counter()
        try:
            response = self._client.post(self._url, json={"chat_id": self.chat_id, "text": text})
            data = response.json()
            if response.status_code == 200 and data.get("ok"):
                message_id = str(data["result"]["message_id"])
                return [Delivery(self.name, self.chat_id, True, _ms_since(started), provider_id=message_id)]
            error = data.get("description") or f"HTTP {response.status_code}"
        except (httpx.HTTPError, ValueError) as exc:
            error = str(exc).replace(self._token, "***")  # the token is part of the URL
        log.warning("Telegram alert failed: %s", error)
        return [Delivery(self.name, self.chat_id, False, _ms_since(started), error=error)]


CHANNEL_NAMES = {"console", "telegram"}


def build_channels(settings: Settings, names: Sequence[str] | None = None) -> list[Channel]:
    """Channels named in ALERT_CHANNEL (or `names`). Ones missing credentials are skipped with a warning."""
    wanted = {n.lower() for n in (names or settings.alert_channels)}
    if "all" in wanted:
        wanted = set(CHANNEL_NAMES)
    if unknown := wanted - CHANNEL_NAMES:
        raise ValueError(f"unknown alert channel(s) {sorted(unknown)}: use console, telegram or all "
                         "(SMS through Twilio was removed)")
    s = settings
    channels: list[Channel] = []
    if "console" in wanted:
        channels.append(ConsoleChannel())
    if "telegram" in wanted:
        if s.telegram_bot_token and s.telegram_chat_id:
            channels.append(TelegramChannel(s.telegram_bot_token, s.telegram_chat_id))
        else:
            log.warning("Telegram alerts skipped: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
    return channels


class AlertPolicy:
    def __init__(self, cooldown_s: float = 600.0, min_level: RiskLevel = RiskLevel.ORANGE) -> None:
        self.cooldown_s = cooldown_s
        self.min_level = min_level
        self._last: dict[str, tuple[float, RiskLevel]] = {}

    def should_alert(self, zone_id: str, level: RiskLevel, now: float) -> bool:
        if level < self.min_level:
            return False
        last = self._last.get(zone_id)
        if last is None:
            return True
        last_time, last_level = last
        return level > last_level or now - last_time >= self.cooldown_s

    def record(self, zone_id: str, level: RiskLevel, now: float) -> None:
        self._last[zone_id] = (now, level)


def format_alert(status: ZoneStatus, zone_name: str, route: Sequence[str] | None = None,
                 actions: Sequence[str] | None = None) -> str:
    """Alert text; `route` is the recommended exit path starting at the zone, `actions` the gate
    actions (app.gate_actions), each on its own line (the generic advice when there are none)."""
    ts = status.timestamp
    when = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts > 1e9 else f"t={ts:.0f}s"
    exit_via = f" Exit via {' > '.join(place_name(n) for n in route[1:])}." if route and len(route) > 1 else ""
    spot = status.hotspot
    hotspot = (f" Hotspot: {spot['label']} of the view (~{spot['count']:.0f} people, {spot['density']:.1f} ppl/m2)."
               if spot else "")
    surge = (f" SPEED SURGE: people moving {status.surge_ratio:.1f}x faster than in the last few seconds."
             if status.escalation and status.surge_ratio else
             f" Escalated: {status.escalation}." if status.escalation else "")
    text = (f"{status.level}: {zone_name} -{surge} pressure {status.smoothed_pressure:.1f}, "
            f"{status.density:.1f} ppl/m2, turbulence {status.turbulence:.2f}, ~{status.count:.0f} people, "
            f"{when}.{hotspot}{exit_via}")
    if actions:
        return text + "\nGate actions:\n" + "\n".join(f"- {a}" for a in actions)
    return f"{text} {ACTIONS.get(status.level, '')}".strip()


@dataclass(frozen=True)
class AlertEvent:
    zone_id: str
    level: RiskLevel
    text: str
    triggered_at: float
    deliveries: tuple[Delivery, ...]
    route: tuple[str, ...] = ()
    hotspot: dict | None = None  # the zone's most crowded grid cell when the alert fired
    actions: tuple[str, ...] = ()  # the gate actions sent with it (app.gate_actions)
    reason: str | None = None  # why the level was raised above the pressure's own, e.g. "speed surge"


class AlertDispatcher:
    """`prefix` marks every message, e.g. "[VIDEO TEST] " for uploaded test videos."""

    def __init__(self, channels: Sequence[Channel], policy: AlertPolicy | None = None, prefix: str = "") -> None:
        self.channels = list(channels)
        self.policy = policy or AlertPolicy()
        self.prefix = prefix

    def handle(self, status: ZoneStatus, zone_name: str, route: Sequence[str] | None = None,
               actions: Sequence[str] | None = None) -> AlertEvent | None:
        """Send an alert for this reading if the policy allows; None when nothing was sent."""
        if not self.policy.should_alert(status.zone_id, status.level, status.timestamp):
            return None
        text = self.prefix + format_alert(status, zone_name, route, actions)
        deliveries = tuple(d for channel in self.channels for d in channel.send(text))
        self.policy.record(status.zone_id, status.level, status.timestamp)
        return AlertEvent(status.zone_id, status.level, text, status.timestamp, deliveries, tuple(route or ()),
                          status.hotspot, tuple(actions or ()), status.escalation)
