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


_TG_MAX = 4096   # Telegram message length limit


def _telegram_send_chunked(tok: str, chat_id: str, text: str, **extra):
    """Send a message, splitting at line boundaries if it exceeds 4096 chars."""
    chunks = []
    buf = ""
    for line in text.split("\n"):
        candidate = (buf + "\n" + line) if buf else line
        if len(candidate) > _TG_MAX:
            if buf:
                chunks.append(buf)
            # if a single line itself exceeds the limit, hard-split it
            while len(line) > _TG_MAX:
                chunks.append(line[:_TG_MAX])
                line = line[_TG_MAX:]
            buf = line
        else:
            buf = candidate
    if buf:
        chunks.append(buf)
    for chunk in (chunks or [text]):
        payload = json.dumps({"chat_id": chat_id, "text": chunk, **extra})
        _post(f"https://api.telegram.org/bot{tok}/sendMessage",
              payload.encode(), {"Content-Type": "application/json"})


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
            _telegram_send_chunked(
                tok, chat, f"*{title}*\n{body}",
                parse_mode="Markdown",
                disable_notification=priority in ("low", "normal"))
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


def send_action_request(pid: str, code: str, desc: str) -> bool:
    """Push a remote-confirm request to the phone. Telegram gets tappable
    Confirm/Reject inline buttons (the real remote-confirm path); other
    channels get a notification carrying the 4-digit code to type on screen.
    Returns True if Telegram interactive buttons were sent."""
    body = f"Confirm to proceed · code {code}\n{desc}"
    tg_tok = os.environ.get("QTSYS_TG_TOKEN")
    tg_chat = os.environ.get("QTSYS_TG_CHAT")
    if tg_tok and tg_chat:
        try:
            payload = json.dumps({
                "chat_id": tg_chat, "text": f"*QTSYS action*\n{desc}",
                "parse_mode": "Markdown",
                "reply_markup": {"inline_keyboard": [[
                    {"text": "✅ Confirm", "callback_data": f"ok:{pid}"},
                    {"text": "✖ Reject", "callback_data": f"no:{pid}"}]]}})
            _post(f"https://api.telegram.org/bot{tg_tok}/sendMessage",
                  payload.encode(), {"Content-Type": "application/json"})
            return True
        except Exception:
            pass
    send("QTSYS · confirm needed", body, "high")     # ntfy/slack: code to type
    return False


def telegram_get_updates(offset: int) -> tuple[list, list, int]:
    """Long-poll Telegram for button taps AND text messages.

    Returns (callbacks, messages, next_offset) where:
      callbacks — [(update_id, pid, approve, callback_id), ...]
      messages  — [(chat_id, text), ...]  (only from QTSYS_TG_CHAT)
    Empty lists and unchanged offset when Telegram isn't configured."""
    tok = os.environ.get("QTSYS_TG_TOKEN")
    if not tok:
        return [], [], offset
    try:
        url = (f"https://api.telegram.org/bot{tok}/getUpdates?timeout=25"
               f"&allowed_updates=[\"callback_query\",\"message\"]"
               + (f"&offset={offset}" if offset else ""))
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read())
    except Exception:
        return [], [], offset
    cbs = []
    msgs = []
    nxt = offset
    allowed_chat = os.environ.get("QTSYS_TG_CHAT", "")
    for u in data.get("result", []):
        nxt = u["update_id"] + 1
        cq = u.get("callback_query")
        if cq:
            d = cq.get("data", "")
            if ":" in d:
                act, pid = d.split(":", 1)
                cbs.append((u["update_id"], pid, act == "ok", cq.get("id")))
            continue
        msg = u.get("message") or {}
        txt = (msg.get("text") or "").strip()
        chat_id = str(msg.get("chat", {}).get("id", ""))
        if txt and chat_id and chat_id == allowed_chat:
            msgs.append((chat_id, txt))
    return cbs, msgs, nxt


def telegram_ack(callback_id: str, text: str):
    """Clear the button spinner + toast the result in Telegram."""
    tok = os.environ.get("QTSYS_TG_TOKEN")
    if not tok or not callback_id:
        return
    try:
        _post(f"https://api.telegram.org/bot{tok}/answerCallbackQuery",
              json.dumps({"callback_query_id": callback_id, "text": text}).encode(),
              {"Content-Type": "application/json"})
    except Exception:
        pass


def telegram_reply(chat_id: str, text: str):
    """Send a plain-text reply back to a Telegram chat, auto-splitting long
    messages so nothing is truncated."""
    tok = os.environ.get("QTSYS_TG_TOKEN")
    if not tok or not chat_id:
        return
    try:
        _telegram_send_chunked(tok, chat_id, text)
    except Exception:
        pass


if __name__ == "__main__":
    print("configured channel:", channel())
    ok = send("QTSYS test", "notifications are wired ✓", "normal")
    print("dispatched:" if ok else "no channel configured —", ok)
