"""autotrader.py — guarded autonomous execution of the adopted daily plan.

Executes the adopted trade plan and manages its own TP/SL exits — active
trading, but fenced by hard guardrails and the ExecutionGateway. NEW intraday
opportunities are NOT auto-traded here (they go to the proposal inbox for a
human); this engine only runs the plan the desk already adopted and reacts to
its own positions hitting target or stop.

SAFETY (belt and braces):
  - MASTER OFF by default (QTSYS_AUTOTRADE=1 or the /api/autotrader/toggle
    endpoint to arm).
  - PAPER unless QTSYS_AUTOTRADE_LIVE=1 — if the broker is live and that flag
    is unset, every action refuses and logs. One env var is the only thing
    between paper and live, by the operator's explicit choice.
  - Every order still passes ExecutionGateway.pretrade_check (notional /
    position / leverage / price-band / halt) — this never bypasses it.
  - Per-day order cap, max concurrent managed positions, and a daily-loss
    circuit breaker that disarms the engine and pushes a notification.
  - Respects the kill switch: while halted, no entries; the gateway's halt
    flattens the book and the monitor reconciles managed rows to closed.

State in autotrader.db (gitignored). Orders route through the injected
gateway.submit; quotes/positions through the injected broker.
"""
from __future__ import annotations

import datetime
import os
import sqlite3
import time


def _env_flag(name: str, default=False) -> bool:
    v = os.environ.get(name)
    return default if v is None else v not in ("0", "", "false", "False")


class AutoTrader:
    def __init__(self, gateway, broker, vmap=None, log=None, notify=None,
                 db_path=None):
        self.gw = gateway
        self.broker = broker
        self.vmap = vmap or {}
        self.rmap = {v: k for k, v in self.vmap.items()}
        self.log = log or (lambda *a, **k: None)
        self.notify = notify or (lambda *a, **k: None)
        self.db = sqlite3.connect(
            db_path or os.path.join(os.path.dirname(__file__), "autotrader.db"),
            check_same_thread=False)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS managed(
          id INTEGER PRIMARY KEY, plan_date TEXT, symbol TEXT, side TEXT,
          qty REAL, entry REAL, stop REAL, target REAL, order_id TEXT,
          status TEXT, opened_ts REAL, closed_ts REAL, exit_reason TEXT,
          realized REAL);
        CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);""")
        self.db.commit()
        # armed state persists across restarts; default OFF unless env forces on
        if _env_flag("QTSYS_AUTOTRADE"):
            self._set("enabled", "1")
        # guardrails (overridable via env)
        self.max_orders_day = int(os.environ.get("QTSYS_AT_MAX_ORDERS", "20"))
        self.max_concurrent = int(os.environ.get("QTSYS_AT_MAX_CONCURRENT", "8"))
        self.max_daily_loss = float(os.environ.get("QTSYS_AT_MAX_DAILY_LOSS", "0.04"))

    # ---------------------------------------------------------------- kv/state
    def _get(self, k, d=None):
        r = self.db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return r[0] if r else d

    def _set(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (k, str(v)))
        self.db.commit()

    @property
    def enabled(self) -> bool:
        return self._get("enabled") == "1"

    def set_enabled(self, on: bool):
        self._set("enabled", "1" if on else "0")
        self.log("AutoTrader", f"engine {'ARMED' if on else 'DISARMED'}",
                 "warn" if on else "info")

    def live_ok(self) -> bool:
        """True only if it's safe to send: paper always ok; live needs the flag."""
        paper = getattr(self.broker, "paper", True)
        return paper or _env_flag("QTSYS_AUTOTRADE_LIVE")

    def status(self) -> dict:
        return {"enabled": self.enabled, "live_ok": self.live_ok(),
                "paper": getattr(self.broker, "paper", True),
                "orders_today": self._orders_today(),
                "max_orders_day": self.max_orders_day,
                "open": len(self.open_positions()),
                "max_concurrent": self.max_concurrent,
                "max_daily_loss": self.max_daily_loss,
                "realized_today": round(self._realized_today(), 2),
                "positions": self.open_positions()}

    # ---------------------------------------------------------- bookkeeping
    def _today(self):
        return str(datetime.date.today())

    def _orders_today(self) -> int:
        return int(self._get("orders:" + self._today(), "0"))

    def _bump_orders(self):
        self._set("orders:" + self._today(), self._orders_today() + 1)

    def _realized_today(self) -> float:
        r = self.db.execute("SELECT COALESCE(SUM(realized),0) FROM managed WHERE "
                            "status='closed' AND plan_date=?",
                            (self._today(),)).fetchone()
        return float(r[0] or 0.0)

    def open_positions(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT id,symbol,side,qty,entry,stop,target,opened_ts FROM managed "
            "WHERE status='open' ORDER BY opened_ts DESC").fetchall()
        return [{"id": r[0], "symbol": r[1], "side": r[2], "qty": r[3],
                 "entry": r[4], "stop": r[5], "target": r[6], "opened_ts": r[7]}
                for r in rows]

    # ------------------------------------------------------- guardrail check
    def _blocked(self) -> str | None:
        if not self.enabled:
            return "engine disarmed"
        if not self.live_ok():
            return "live keys but QTSYS_AUTOTRADE_LIVE not set — refusing"
        if getattr(self.gw, "halted", False):
            return "kill switch active"
        eq = 0.0
        try:
            eq = float(self.broker.get_account().get("equity") or 0)
        except Exception:
            pass
        if eq and self._realized_today() / eq <= -self.max_daily_loss:
            self.set_enabled(False)
            self.notify("QTSYS · auto-trade halted",
                        f"daily loss circuit breaker hit ({self.max_daily_loss:.0%})"
                        " — engine disarmed", "urgent")
            return "daily-loss circuit breaker tripped"
        return None

    def _venue(self, sym: str) -> str:
        return self.vmap.get(sym, sym)

    # ------------------------------------------------------------ execution
    def execute_plan(self, plan: dict) -> dict:
        """Enter the adopted plan's ideas, guardrail- and gateway-checked."""
        from .brokers import Order
        blk = self._blocked()
        if blk:
            return {"executed": 0, "blocked": blk}
        done, skipped = 0, []
        existing = {p["symbol"] for p in self.open_positions()}
        for idea in plan.get("ideas", []):
            sym = idea["symbol"]
            if sym in existing:
                skipped.append((sym, "already managed"))
                continue
            if self._orders_today() >= self.max_orders_day:
                skipped.append((sym, "daily order cap")); continue
            if len(self.open_positions()) >= self.max_concurrent:
                skipped.append((sym, "max concurrent")); continue
            if not idea.get("qty") or not idea.get("stop") or not idea.get("target"):
                skipped.append((sym, "unsized")); continue
            side = "buy" if idea["side"] in ("LONG", "buy") else "sell"
            o = Order(self._venue(sym), side, float(idea["qty"]), "market", None)
            res = self.gw.submit(o)
            self._bump_orders()
            if res.status == "rejected":
                skipped.append((sym, f"gateway: {res.reason}")); continue
            self.db.execute(
                "INSERT INTO managed(plan_date,symbol,side,qty,entry,stop,target,"
                "order_id,status,opened_ts,realized) VALUES (?,?,?,?,?,?,?,?, "
                "'open', ?, 0)",
                (plan.get("date", self._today()), sym, side, float(idea["qty"]),
                 idea["entry"], idea["stop"], idea["target"], res.id or "",
                 time.time()))
            self.db.commit()
            done += 1
            self.log("AutoTrader",
                     f"ENTERED {side.upper()} {idea['qty']} {sym} @~{idea['entry']} "
                     f"(stop {idea['stop']} / target {idea['target']})", "warn")
        if done:
            self.notify("QTSYS · plan executed",
                        f"{done} positions entered from the {plan.get('date')} plan",
                        "high")
        return {"executed": done, "skipped": skipped}

    def monitor(self) -> dict:
        """Close managed positions that hit TP or SL; reconcile on halt."""
        closed = 0
        halted = getattr(self.gw, "halted", False)
        for p in self.open_positions():
            if halted:                             # kill switch flattened the book
                self._mark_closed(p["id"], "halt", None)
                continue
            try:
                px = self.broker.get_quote(self._venue(p["symbol"]))
            except Exception:
                continue
            if not px or px != px:
                continue
            long = p["side"] == "buy"
            hit = ("target" if (px >= p["target"] if long else px <= p["target"])
                   else "stop" if (px <= p["stop"] if long else px >= p["stop"])
                   else None)
            if hit:
                if self._close(p, px, hit):
                    closed += 1
        return {"closed": closed, "open": len(self.open_positions())}

    def _close(self, p: dict, px: float, reason: str) -> bool:
        from .brokers import Order
        side = "sell" if p["side"] == "buy" else "buy"
        o = Order(self._venue(p["symbol"]), side, float(p["qty"]), "market", None)
        res = self.gw.submit(o)
        self._bump_orders()
        if res.status == "rejected":
            self.log("AutoTrader", f"close REJECTED {p['symbol']}: {res.reason}",
                     "error")
            return False
        realized = ((px - p["entry"]) if p["side"] == "buy"
                    else (p["entry"] - px)) * p["qty"]
        self._mark_closed(p["id"], reason, realized)
        emoji = "🎯" if reason == "target" else "🛑"
        self.log("AutoTrader", f"{emoji} {reason.upper()} {p['symbol']} @ {px:g} "
                 f"— realized {realized:+.2f}", "warn")
        self.notify(f"QTSYS · {reason} hit · {p['symbol']}",
                    f"closed {p['qty']} {p['symbol']} @ {px:g}, P&L {realized:+.2f}",
                    "high")
        return True

    def _mark_closed(self, mid: int, reason: str, realized):
        self.db.execute("UPDATE managed SET status='closed', closed_ts=?, "
                        "exit_reason=?, realized=? WHERE id=?",
                        (time.time(), reason, realized, mid))
        self.db.commit()


# ------------------------------------------------------------------ self-test
class _FakeOrder:
    def __init__(self, sym, side, qty):
        self.symbol, self.side, self.qty = sym, side, qty
        self.status, self.reason, self.id = "accepted", "", "oid"


class _FakeGW:
    halted = False
    def submit(self, o):
        o.status = "accepted"; o.id = "oid"; return o


class _FakeBroker:
    paper = True
    def __init__(self): self.px = {"AAPL": 200.0}
    def get_quote(self, s): return self.px.get(s, 100.0)
    def get_account(self): return {"equity": 100000.0}


def _selftest():
    import tempfile
    from .brokers import Order as _RealOrder      # ensure Order import path works
    at = AutoTrader(_FakeGW(), _FakeBroker(), db_path=tempfile.mktemp(suffix=".db"))
    assert not at.enabled, "disarmed by default"
    assert at.execute_plan({"ideas": [{}]})["blocked"] == "engine disarmed"
    at.set_enabled(True)
    plan = {"date": at._today(), "ideas": [
        {"symbol": "AAPL", "side": "LONG", "qty": 10, "entry": 200.0,
         "stop": 194.0, "target": 212.0}]}
    r = at.execute_plan(plan)
    assert r["executed"] == 1 and len(at.open_positions()) == 1, r
    # price below entry, above stop -> no exit
    at.broker.px["AAPL"] = 205.0
    assert at.monitor()["closed"] == 0
    # hit target -> close with positive realized
    at.broker.px["AAPL"] = 213.0
    assert at.monitor()["closed"] == 1 and not at.open_positions()
    assert at._realized_today() > 0, "target close booked a profit"
    # re-enter, then hit stop -> close with loss
    at.execute_plan(plan)
    at.broker.px["AAPL"] = 193.0
    at.monitor()
    assert at._realized_today() < 130, "stop close booked the loss"      # net of the win
    # live-safety gate: a live broker without the flag refuses
    at.broker.paper = False
    os.environ.pop("QTSYS_AUTOTRADE_LIVE", None)
    assert at.execute_plan(plan)["blocked"].startswith("live keys"), "live gate"
    print("autotrader self-test ✓  disarmed default, enter->TP profit->SL loss, "
          "gateway routed, live-without-flag refused")


if __name__ == "__main__":
    _selftest()
