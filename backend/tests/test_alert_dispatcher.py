from dataclasses import replace

import httpx
import pytest

from app.alert_dispatcher import (
    AlertDispatcher,
    AlertPolicy,
    Delivery,
    TelegramChannel,
    build_channels,
    format_alert,
)
from app.config import get_settings
from app.risk_engine import RiskLevel, ZoneStatus

GREEN, YELLOW, ORANGE, RED = RiskLevel


def status(level, t=0.0, zone="z1"):
    return ZoneStatus(zone_id=zone, timestamp=t, count=1234.0, density=5.1, turbulence=0.52,
                      pressure=6.4, smoothed_pressure=6.4, level=level, samples=15)


def test_policy_cooldown_and_escalation():
    policy = AlertPolicy(cooldown_s=600)
    assert not policy.should_alert("z1", YELLOW, 0)
    assert policy.should_alert("z1", ORANGE, 0)
    policy.record("z1", ORANGE, 0)
    assert not policy.should_alert("z1", ORANGE, 300)  # within cooldown
    assert policy.should_alert("z1", RED, 300)  # escalation bypasses cooldown
    policy.record("z1", RED, 300)
    assert not policy.should_alert("z1", ORANGE, 400)
    assert not policy.should_alert("z1", RED, 899)
    assert policy.should_alert("z1", ORANGE, 900)  # cooldown over
    assert policy.should_alert("z2", ORANGE, 400)  # zones are independent


class FakeChannel:
    name = "fake"

    def __init__(self, ok=True):
        self.ok = ok
        self.sent = []

    def send(self, text):
        self.sent.append(text)
        return [Delivery(self.name, "x", self.ok, 1.0, error=None if self.ok else "down")]


def test_dispatcher_sends_once_per_cooldown():
    channel = FakeChannel()
    dispatcher = AlertDispatcher([channel], AlertPolicy(cooldown_s=600))
    assert dispatcher.handle(status(GREEN, 0), "Ghat") is None
    event = dispatcher.handle(status(ORANGE, 2), "Ghat")
    assert event.level is ORANGE and event.deliveries[0].ok
    assert dispatcher.handle(status(ORANGE, 4), "Ghat") is None
    assert dispatcher.handle(status(RED, 6), "Ghat").level is RED
    assert len(channel.sent) == 2


def test_failed_delivery_is_reported_not_repeated_every_reading():
    channel = FakeChannel(ok=False)
    dispatcher = AlertDispatcher([channel])
    event = dispatcher.handle(status(ORANGE, 0), "Ghat")
    assert event.deliveries[0].error == "down"
    assert dispatcher.handle(status(ORANGE, 2), "Ghat") is None  # cooldown applies anyway
    assert dispatcher.handle(status(RED, 4), "Ghat").level is RED  # escalation still fires


def test_alert_text_is_short_plain_ascii():
    text = format_alert(status(RED, 1_791_000_000.0), "Ramkund main ghat steps")
    assert text.startswith("RED: Ramkund main ghat steps")
    assert text.isascii() and len(text) <= 160


def test_message_contains_zone_risk_and_exit_route():
    route = ["RAMKUND_NORTH", "GHAT_1", "EXIT_GATE_C"]
    text = format_alert(status(ORANGE, 1_791_000_000.0), "Ramkund north steps", route)
    assert "Ramkund north steps" in text and text.startswith("ORANGE")
    assert "Exit via Ghat 1 > Gate C." in text
    assert text.isascii() and len(text) <= 160


def test_dispatcher_puts_route_in_event():
    event = AlertDispatcher([FakeChannel()]).handle(status(RED, 2), "Ghat", ["Z", "EXIT_GATE_A"])
    assert event.route == ("Z", "EXIT_GATE_A") and "Exit via Gate A." in event.text


def channel_settings(channels, telegram=False):
    return replace(get_settings(), alert_channels=channels, telegram_bot_token="123:abc" if telegram else "",
                   telegram_chat_id="42" if telegram else "")


def test_build_channels_follows_alert_channel():
    assert [c.name for c in build_channels(channel_settings(("console",)))] == ["console"]
    assert [c.name for c in build_channels(channel_settings(("TELEGRAM",), telegram=True))] == ["telegram"]
    assert [c.name for c in build_channels(channel_settings(("all",), telegram=True))] == ["console", "telegram"]
    assert [c.name for c in build_channels(channel_settings(("console",)), ["telegram"])] == []  # no credentials


def test_build_channels_rejects_unknown_names():
    with pytest.raises(ValueError, match="pager"):
        build_channels(channel_settings(("pager",)))
    with pytest.raises(ValueError, match="Twilio was removed"):
        build_channels(channel_settings(("sms",)))


def test_telegram_success_and_token_redaction():
    token = "123456:SECRET"

    def handler(request):
        assert request.url.path == f"/bot{token}/sendMessage"
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 99}})

    ok = TelegramChannel(token, "42", client=httpx.Client(transport=httpx.MockTransport(handler))).send("hi")
    assert ok[0].ok and ok[0].provider_id == "99"

    def failing(request):
        raise httpx.ConnectError(f"cannot reach {request.url}")

    failed = TelegramChannel(token, "42", client=httpx.Client(transport=httpx.MockTransport(failing))).send("hi")
    assert not failed[0].ok and "SECRET" not in failed[0].error

    def rejected(request):
        return httpx.Response(400, json={"ok": False, "description": "Bad Request: chat not found"})

    bad = TelegramChannel(token, "42", client=httpx.Client(transport=httpx.MockTransport(rejected))).send("hi")
    assert bad[0].error == "Bad Request: chat not found"


def test_telegram_gives_up_quickly_without_internet():
    """Alerts are sent inside the inference cycle, so a dead network must not hold it for long."""
    timeout = TelegramChannel("123:abc", "42")._client.timeout
    assert (timeout.connect, timeout.read, timeout.write) == (3.0, 5.0, 5.0)
