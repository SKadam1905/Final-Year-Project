"""Week-1 check that Telegram alerts actually arrive (plan Phase 1).

    python -m tools.send_test_alert                    # the channels in ALERT_CHANNEL (.env)
    python -m tools.send_test_alert --channel telegram # or console / all

Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the repo-root .env first (see .env.example),
and send /start to the bot once from that chat.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from app.alert_dispatcher import build_channels
from app.config import get_settings


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channel", choices=["console", "telegram", "all"],
                   help="default: ALERT_CHANNEL from .env")
    args = p.parse_args(argv)

    channels = build_channels(get_settings(), [args.channel] if args.channel else None)
    if not channels:
        raise SystemExit("no channel configured; set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
    text = f"TEST: Kumbh crowd-safety prototype alert check at {datetime.now():%H:%M:%S}. No action needed."
    failed = 0
    for channel in channels:
        for d in channel.send(text):
            status = "sent" if d.ok else f"FAILED: {d.error}"
            print(f"{d.channel:9s} -> {d.recipient}: {status} ({d.latency_ms:.0f} ms)")
            failed += not d.ok
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
