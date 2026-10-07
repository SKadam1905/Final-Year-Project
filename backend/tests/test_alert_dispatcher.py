from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from app.alert_dispatcher import (
    AlertDispatcher,
    AlertPolicy,
    Delivery,
    TelegramChannel,
    TwilioSmsChannel,
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


def test_failed_delivery_is_retried_next_reading():
    channel = FakeChannel(ok=False)
    dispatcher = AlertDispatcher([channel])
    assert not dispatcher.handle(status(ORANGE, 0), "Ghat").deliveries[0].ok
    channel.ok = True
    assert dispatcher.handle(status(ORANGE, 2), "Ghat").deliveries[0].ok
    assert dispatcher.handle(status(ORANGE, 4), "Ghat") is None


def test_alert_text_fits_one_sms_segment():
    text = format_alert(status(RED, 1_791_000_000.0), "Ramkund main ghat steps")
    assert text.startswith("RED: Ramkund main ghat steps")
    assert text.isascii() and len(text) <= 160


def test_message_contains_zone_risk_and_exit_route():
    route = ["RAMKUND_NORTH", "GHAT_1", "EXIT_GATE_C"]
    text = format_alert(status(ORANGE, 1_791_000_000.0), "Ramkund north steps", route)
    assert "Ramkund north steps" in text and text.startswith("ORANGE")
    assert "Exit via GHAT_1>EXIT_GATE_C." in text
    assert text.isascii() and len(text) <= 160


def test_dispatcher_puts_route_in_event():
    event = AlertDispatcher([FakeChannel()]).handle(status(RED, 2), "Ghat", ["Z", "EXIT_GATE_A"])
    assert event.route == ("Z", "EXIT_GATE_A") and "Exit via EXIT_GATE_A." in event.text


def channel_settings(channels, twilio=False, telegram=False):
    return replace(
        get_settings(), alert_channels=channels,
        twilio_account_sid="AC123" if twilio else "", twilio_auth_token="token" if twilio else "",
        twilio_from_number="+15550001111" if twilio else "", alert_sms_to=("+919999999999",) if twilio else (),
        telegram_bot_token="123:abc" if telegram else "", telegram_chat_id="42" if telegram else "",
    )


def test_build_channels_follows_alert_channel():
    assert [c.name for c in build_channels(channel_settings(("console",)))] == ["console"]
    assert [c.name for c in build_channels(channel_settings(("twilio",), twilio=True))] == ["sms"]
    assert [c.name for c in build_channels(channel_settings(("all",), twilio=True, telegram=True))] == \
        ["console", "sms", "telegram"]
    assert [c.name for c in build_channels(channel_settings(("console",)), ["telegram"])] == []  # no credentials


def test_build_channels_rejects_unknown_names():
    with pytest.raises(ValueError, match="pager"):
        build_channels(channel_settings(("pager",)))


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


def test_twilio_channel_sends_to_each_number():
    calls = []

    def create(body, from_, to):
        if to == "+910000000000":
            raise RuntimeError("unverified number")
        calls.append((body, from_, to))
        return SimpleNamespace(sid="SM123")

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    channel = TwilioSmsChannel("sid", "token", "+15550001111", ["+919999999999", "+910000000000"], client=client)
    first, second = channel.send("alert")
    assert calls == [("alert", "+15550001111", "+919999999999")]
    assert first.ok and first.provider_id == "SM123"
    assert not second.ok and "unverified" in second.error
