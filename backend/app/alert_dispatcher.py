"""Alert policy and delivery channels (plan Layer 3, §4.6).

Policy: alert when a zone's smoothed level reaches ORANGE or RED, then stay quiet for that
zone for the cooldown (10 min), except that an escalation (ORANGE -> RED) always fires.
A zone that simply stays RED is not re-alerted every 2 s; it gets a reminder once the
cooldown has passed. If every channel fails, nothing is recorded, so the next reading retries.

Channels, chosen with ALERT_CHANNEL in .env (comma-separated): console, sms (alias twilio),
telegram (no DLT registration needed, so it is the demo fallback), or all. Messages stay
plain ASCII (emoji would switch SMS to 70-character segments) and about one SMS long.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence

import httpx

from app.config import Settings
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


class TwilioSmsChannel:
    name = "sms"

    def __init__(self, account_sid: str, auth_token: str, from_number: str, to_numbers: Sequence[str], client=None):
        if client is None:
            from twilio.rest import Client

            client = Client(account_sid, auth_token)
        self._client = client
        self.from_number = from_number
        self.to_numbers = tuple(to_numbers)

    def send(self, text: str) -> list[Delivery]:
        results = []
        for to in self.to_numbers:
            started = time.perf_counter()
            try:
                message = self._client.messages.create(body=text, from_=self.from_number, to=to)
                results.append(Delivery(self.name, to, True, _ms_since(started), provider_id=message.sid))
            except Exception as exc:  # network failure or Twilio API error (e.g. unverified trial number)
                log.warning("SMS to %s failed: %s", to, exc)
                results.append(Delivery(self.name, to, False, _ms_since(started), error=str(exc)))
        return results


class TelegramChannel:
    name = "telegram"

    def __init__(self, bot_token: str, chat_id: str, client: httpx.Client | None = None, timeout_s: float = 10.0):
        self._token = bot_token
        self._url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self.chat_id = chat_id
        self._client = client or httpx.Client(timeout=timeout_s)

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


CHANNEL_NAMES = {"console", "sms", "telegram"}
CHANNEL_ALIASES = {"twilio": "sms"}


def build_channels(settings: Settings, names: Sequence[str] | None = None) -> list[Channel]:
    """Channels named in ALERT_CHANNEL (or `names`). Ones missing credentials are skipped with a warning."""
    wanted = {CHANNEL_ALIASES.get(n.lower(), n.lower()) for n in (names or settings.alert_channels)}
    if "all" in wanted:
        wanted = set(CHANNEL_NAMES)
    if unknown := wanted - CHANNEL_NAMES:
        raise ValueError(f"unknown alert channel(s) {sorted(unknown)}: use console, sms (or twilio), telegram, all")
    s = settings
    channels: list[Channel] = []
    if "console" in wanted:
        channels.append(ConsoleChannel())
    if "sms" in wanted:
        if s.twilio_account_sid and s.twilio_auth_token and s.twilio_from_number and s.alert_sms_to:
            channels.append(TwilioSmsChannel(s.twilio_account_sid, s.twilio_auth_token, s.twilio_from_number,
                                             s.alert_sms_to))
        else:
            log.warning("SMS alerts skipped: set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, ALERT_SMS_TO")
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


def format_alert(status: ZoneStatus, zone_name: str, route: Sequence[str] | None = None) -> str:
    """One-line alert text; `route` is the recommended exit path starting at the zone."""
    ts = status.timestamp
    when = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts > 1e9 else f"t={ts:.0f}s"
    exit_via = f" Exit via {'>'.join(route[1:])}." if route and len(route) > 1 else ""
    return (f"{status.level}: {zone_name} - pressure {status.smoothed_pressure:.1f}, "
            f"{status.density:.1f} ppl/m2, turbulence {status.turbulence:.2f}, ~{status.count:.0f} people, "
            f"{when}.{exit_via} {ACTIONS.get(status.level, '')}").strip()


@dataclass(frozen=True)
class AlertEvent:
    zone_id: str
    level: RiskLevel
    text: str
    triggered_at: float
    deliveries: tuple[Delivery, ...]
    route: tuple[str, ...] = ()


class AlertDispatcher:
    def __init__(self, channels: Sequence[Channel], policy: AlertPolicy | None = None) -> None:
        self.channels = list(channels)
        self.policy = policy or AlertPolicy()

    def handle(self, status: ZoneStatus, zone_name: str, route: Sequence[str] | None = None) -> AlertEvent | None:
        """Send an alert for this reading if the policy allows; None when nothing was sent."""
        if not self.policy.should_alert(status.zone_id, status.level, status.timestamp):
            return None
        text = format_alert(status, zone_name, route)
        deliveries = tuple(d for channel in self.channels for d in channel.send(text))
        if any(d.ok for d in deliveries):
            self.policy.record(status.zone_id, status.level, status.timestamp)
        return AlertEvent(status.zone_id, status.level, text, status.timestamp, deliveries, tuple(route or ()))
