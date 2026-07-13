"""errlog.py — make swallowed errors VISIBLE without spamming the log.

The hot loops (quote tick, WebSocket stream, autotrader, alerts, live-weights,
broker fills, intel fetches) must never crash the server on a transient error,
so they swallow exceptions. Silent `except: pass` hides real, recurring failures
though. errlog keeps a bounded per-key counter: it logs the FIRST occurrence and
then every Nth, and exposes `stats()` for `/api/health` and `GET /api/errors` so
an operator can SEE that "alpaca_fills" has failed 4,000 times even if the UI
still renders. Behaviour is unchanged — the error is still swallowed — but it is
now counted and surfaced.

Run:  python -m qtsys.errlog
"""
from __future__ import annotations

import threading
import time

_LOCK = threading.Lock()
_ERRORS: dict[str, dict] = {}


def report(key: str, exc: BaseException | str, log=None, every: int = 20) -> int:
    """Count one failure under `key`. Logs (via the optional agent `log`
    callable) on the first occurrence and then every `every`-th, so a persistent
    failure is loud once and then sampled. Returns the running count for `key`.
    Never raises — it is called from except-blocks."""
    msg = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    etype = "str" if isinstance(exc, str) else type(exc).__name__
    with _LOCK:
        e = _ERRORS.setdefault(key, {"count": 0, "last_ts": 0.0, "last": "",
                                     "type": ""})
        e["count"] += 1
        e["last_ts"] = time.time()
        e["last"] = msg
        e["type"] = etype
        n = e["count"]
    if log and (n == 1 or (every > 0 and n % every == 0)):
        try:
            log("errlog", f"{key}: {msg} (x{n})", "error")
        except Exception:
            pass
    return n


def stats() -> dict:
    """Snapshot of every error key: count, last message/type, last timestamp,
    plus a total. Safe to serialise straight into an API response."""
    with _LOCK:
        keys = {k: dict(v) for k, v in _ERRORS.items()}
    total = sum(v["count"] for v in keys.values())
    return {"total": total, "keys": keys}


def reset() -> None:
    """Clear all counters (used by tests)."""
    with _LOCK:
        _ERRORS.clear()


def _selftest():
    reset()
    logged: list[tuple] = []
    def log(agent, msg, level="info"):
        logged.append((agent, msg, level))

    # first occurrence logs; then every 20th
    for i in range(25):
        n = report("quotes", ValueError("boom"), log=log, every=20)
    assert n == 25, n
    s = stats()
    assert s["keys"]["quotes"]["count"] == 25
    assert s["keys"]["quotes"]["type"] == "ValueError"
    assert s["keys"]["quotes"]["last"] == "ValueError: boom"
    assert s["total"] == 25
    # logged on the 1st and the 20th only (2 lines), all at error level
    assert len(logged) == 2, [m for _, m, _ in logged]
    assert all(lv == "error" for _, _, lv in logged)

    # a string reason is accepted too, and keys are independent
    report("intel", "timeout fetching fundamentals", log=log)
    s2 = stats()
    assert s2["keys"]["intel"]["count"] == 1
    assert s2["total"] == 26
    # never raises even with a bad log callable
    report("x", RuntimeError("z"), log=lambda *a, **k: (_ for _ in ()).throw(IOError()))
    reset()
    assert stats()["total"] == 0
    print("errlog self-test ✓  per-key counting, first+every-Nth logging, "
          "string/exception reasons, stats() total+keys, reset, never raises")


if __name__ == "__main__":
    _selftest()
