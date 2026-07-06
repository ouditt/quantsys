"""notify.py — out-of-band push notifications (best-effort, non-blocking).

Sends short messages to whatever channel is configured in the environment, so
you get filled / kill-switch / arb-signal / alert pings when you're away from
the terminal. Zero required deps — every channel is a plain HTTP POST.

Configure any subset (first-configured wins per call, or set QTSYS_NOTIFY to
force one of: ntfy | telegram | slack | none):
  ntfy      QTSYS_NTFY_TOPIC   (free, no account — https://ntfy.sh/<topic>)
  telegram  QTSYS_TG_TOKEN + QTSYS_TG_CHAT
  slack     QTSYS_SLACK_WEBHOOK   (also works for Discord webhooks)

Every send is fire-and-forget on a thread; a failure is swallowed (a down
notification channel must never affect trading). Priorities map to ntfy
tags/telegram silent so routine pings don't buzz like a kill switch does.
"""
from __future__ import annotations

import json
import os
import threading
import urllib.request

_TIMEOUT = 6
PRIORITY = ("low", "normal", "high", "urgent")


def channel() -> str:
    forced = os.environ.get("QTSYS_NOTIFY", "").strip().lower()
    if forced:
        return forced
    if os.environ.get("QTSYS_NTFY_TOPIC"):
        return "ntfy"
    if os.environ.get("QTSYS_TG_TOKEN") and os.environ.get("QTSYS_TG_CHAT"):
        return "telegram"
    if os.environ.get("QTSYS_SLACK_WEBHOOK"):
        return "slack"
    return "none"


def _post(url: str, data: bytes, headers: dict):
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        r.read()


def _send_sync(title: str, body: str, priority: str):
    ch = channel()
    try:
        if ch == "ntfy":
            topic = os.environ["QTSYS_NTFY_TOPIC"]
            base = os.environ.get("QTSYS_NTFY_URL", "https://ntfy.sh")
            tags = {"low": "information_source", "normal": "chart",
                    "high": "warning", "urgent": "rotating_light"}.get(priority, "chart")
            _post(f"{base.rstrip('/')}/{topic}", body.encode("utf-8"),
                  {"Title": title, "Priority":
                   {"low": "2", "normal": "3", "high": "4", "urgent": "5"}.get(priority, "3"),
                   "Tags": tags})
        elif ch == "telegram":
            tok, chat = os.environ["QTSYS_TG_TOKEN"], os.environ["QTSYS_TG_CHAT"]
            payload = json.dumps({"chat_id": chat, "text": f"*{title}*\n{body}",
                                  "parse_mode": "Markdown",
                                  "disable_notification": priority in ("low", "normal")})
            _post(f"https://api.telegram.org/bot{tok}/sendMessage",
                  payload.encode(), {"Content-Type": "application/json"})
        elif ch == "slack":
            _post(os.environ["QTSYS_SLACK_WEBHOOK"],
                  json.dumps({"text": f"*{title}*\n{body}"}).encode(),
                  {"Content-Type": "application/json"})
    except Exception:
        pass


def send(title: str, body: str = "", priority: str = "normal") -> bool:
    """Fire-and-forget notification. Returns False only if no channel is
    configured (so callers can note it), True once dispatched."""
    if channel() == "none":
        return False
    threading.Thread(target=_send_sync, args=(title, body, priority),
                     daemon=True).start()
    return True


if __name__ == "__main__":
    print("configured channel:", channel())
    ok = send("QTSYS test", "notifications are wired ✓", "normal")
    print("dispatched:" if ok else "no channel configured —", ok)
