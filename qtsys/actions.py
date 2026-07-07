"""actions.py — voice/text-STAGED desk actions with a confirm gate.

The Copilot answers questions; THIS turns a spoken/typed command into a STAGED
action that must be CONFIRMED before it runs — on-screen (type the word) or
remotely from the phone (Telegram inline buttons / ntfy). Nothing here bypasses
the ExecutionGateway or the auto-trader's guardrails; confirm just authorises
the same code path the UI buttons already call.

Intent parsing is DELIBERATELY deterministic (regex over a small verb set) —
we never let a local model's misread trigger a trade. Unrecognised commands
fall through to the Copilot as a question.

  parse_intent(text)  -> {kind, desc} | None
  PendingStore        -> stage/confirm/reject with a short single-use code + TTL
"""
from __future__ import annotations

import re
import secrets
import threading
import time

# each intent: (compiled regex, kind, human description for the confirm prompt)
_INTENTS = [
    (re.compile(r"\b(dis-?arm|disable|stop|turn\s*off|pause)\b.*\b(auto|trad|engine|bot)", re.I),
     "disarm", "DISARM the auto-trader (stop autonomous trading)"),
    (re.compile(r"\b(arm|enable|activate|turn\s*on|start)\b.*\b(auto|trad|engine|bot)", re.I),
     "arm", "ARM the auto-trader (allow autonomous trading within guardrails)"),
    (re.compile(r"\b(kill|flatten|emergency|panic)\b", re.I),
     "kill", "KILL SWITCH — flatten the book and halt all trading"),
    (re.compile(r"\b(resume|un-?halt|re-?start\s*trading|lift\s*the\s*halt)\b", re.I),
     "resume", "RESUME trading after a halt"),
    (re.compile(r"\b(re-?build|build|draft|make|create|prepare|generate)\b.*\bplan\b", re.I),
     "build_plan", "BUILD today's trade plan (draft + desk deliberation)"),
    (re.compile(r"\b(execute|run|place|enter|action)\b.*\bplan\b", re.I),
     "execute_plan", "EXECUTE the adopted plan (auto-trader enters the verified ideas)"),
]


def parse_intent(text: str) -> dict | None:
    t = (text or "").strip()
    if not t:
        return None
    for rx, kind, desc in _INTENTS:
        if rx.search(t):
            return {"kind": kind, "desc": desc}
    return None


class PendingStore:
    """Short-lived, single-use staged actions awaiting confirmation."""

    def __init__(self, ttl: float = 300.0):
        self.ttl = ttl
        self._lock = threading.Lock()
        self._d: dict[str, dict] = {}

    def stage(self, kind: str, desc: str, source: str = "") -> dict:
        self._sweep()
        pid = secrets.token_hex(4)
        code = f"{secrets.randbelow(9000) + 1000}"      # 4-digit confirm code
        with self._lock:
            self._d[pid] = {"id": pid, "code": code, "kind": kind, "desc": desc,
                            "source": source, "ts": time.time(),
                            "status": "pending"}
        return dict(self._d[pid])

    def get(self, pid: str) -> dict | None:
        self._sweep()
        with self._lock:
            d = self._d.get(pid)
            return dict(d) if d else None

    def resolve(self, pid: str, code: str | None, approve: bool) -> dict | None:
        """Confirm/reject. Code must match on approve (None skips the check for
        the on-screen path which is already token-authenticated)."""
        self._sweep()
        with self._lock:
            d = self._d.get(pid)
            if not d or d["status"] != "pending":
                return None
            if approve and code is not None and code != d["code"]:
                return {"error": "bad code"}
            d["status"] = "confirmed" if approve else "rejected"
            return dict(d)

    def find_by_code(self, code: str) -> dict | None:
        self._sweep()
        with self._lock:
            for d in self._d.values():
                if d["status"] == "pending" and d["code"] == code:
                    return dict(d)
        return None

    def _sweep(self):
        now = time.time()
        with self._lock:
            for k in [k for k, v in self._d.items() if now - v["ts"] > self.ttl]:
                self._d.pop(k, None)


def _selftest():
    assert parse_intent("arm the auto trader")["kind"] == "arm"
    assert parse_intent("please disarm the trading bot")["kind"] == "disarm"
    assert parse_intent("build today's plan")["kind"] == "build_plan"
    assert parse_intent("execute the plan now")["kind"] == "execute_plan"
    assert parse_intent("kill everything")["kind"] == "kill"
    assert parse_intent("resume trading")["kind"] == "resume"
    assert parse_intent("what is my pnl") is None, "questions are NOT actions"
    assert parse_intent("how did BNO do") is None
    st = PendingStore(ttl=100)
    p = st.stage("arm", "arm it", "voice")
    assert st.resolve(p["id"], "0000", True) == {"error": "bad code"}, "wrong code"
    assert st.resolve(p["id"], p["code"], True)["status"] == "confirmed"
    assert st.resolve(p["id"], p["code"], True) is None, "single-use"
    p2 = st.stage("kill", "kill", "voice")
    assert st.find_by_code(p2["code"])["id"] == p2["id"]        # remote lookup
    assert st.resolve(p2["id"], None, True)["status"] == "confirmed"  # on-screen path
    st2 = PendingStore(ttl=-1)                                  # instant expiry
    e = st2.stage("arm", "x")
    assert st2.get(e["id"]) is None, "expired"
    print("actions self-test ✓  intent parse (6 verbs, questions excluded), "
          "stage/confirm/reject, single-use code, remote lookup, TTL")


if __name__ == "__main__":
    _selftest()
