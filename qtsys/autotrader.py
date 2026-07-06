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
          realized REAL, mode TEXT DEFAULT 'paper');
        CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);""")
        for mig in ("ALTER TABLE managed ADD COLUMN mode TEXT DEFAULT 'paper'",
                    "ALTER TABLE managed ADD COLUMN kind TEXT DEFAULT 'equity'",
                    "ALTER TABLE managed ADD COLUMN legs TEXT",
                    "ALTER TABLE managed ADD COLUMN expiration TEXT"):
            try:                                   # migrate older databases
                self.db.execute(mig)
            except Exception:
                pass
        self.db.commit()
        # armed state persists across restarts; default OFF unless env forces on
        if _env_flag("QTSYS_AUTOTRADE"):
            self._set("enabled", "1")
        # guardrails (overridable via env)
        self.max_orders_day = int(os.environ.get("QTSYS_AT_MAX_ORDERS", "20"))
        self.max_concurrent = int(os.environ.get("QTSYS_AT_MAX_CONCURRENT", "8"))
        self.max_daily_loss = float(os.environ.get("QTSYS_AT_MAX_DAILY_LOSS", "0.04"))
        # per-symbol exposure cap, as a fraction of equity (default 10%)
        self.max_symbol_pct = float(os.environ.get("QTSYS_AT_MAX_SYMBOL_PCT", "0.10"))
        # live unlock: distinct PAPER trading days required before the engine
        # will honor QTSYS_AUTOTRADE_LIVE (0 disables the requirement)
        self.paper_days_req = int(os.environ.get("QTSYS_AT_PAPER_DAYS", "60"))
        # verified edge -> machine may trade it; unverified -> human approves.
        # Both knobs are operator-settable at runtime (PLAN page) and persist
        # in the kv store; env vars only seed the first boot.
        if self._get("require_dsr") is None:
            self._set("require_dsr",
                      "1" if _env_flag("QTSYS_AT_REQUIRE_DSR", True) else "0")
        if self._get("dsr_threshold") is None:
            self._set("dsr_threshold",
                      os.environ.get("QTSYS_AT_DSR_THRESHOLD", "0.95"))
        # options auto-trading (defined-risk verticals ONLY) — off by default
        self.options_on = _env_flag("QTSYS_AT_OPTIONS", False)

    @property
    def require_dsr(self) -> bool:
        return self._get("require_dsr") == "1"

    @property
    def dsr_threshold(self) -> float:
        try:
            return float(self._get("dsr_threshold", "0.95"))
        except Exception:
            return 0.95

    def set_dsr_gate(self, require: bool | None = None,
                     threshold: float | None = None):
        if require is not None:
            self._set("require_dsr", "1" if require else "0")
            self.log("AutoTrader", "DSR gate "
                     + ("ON — only verified edge auto-trades" if require else
                        "OFF — the engine may trade UNVERIFIED ideas (operator "
                        "override)"), "warn")
        if threshold is not None:
            t = min(max(float(threshold), 0.0), 1.0)
            self._set("dsr_threshold", str(t))
            self.log("AutoTrader", f"DSR threshold set to {t:g} "
                     + ("(below the 0.95 'likely real' bar — expect more "
                        "noise trades)" if t < 0.95 else ""), "warn")

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

    def paper_days(self) -> int:
        """Distinct calendar days on which the engine closed PAPER trades —
        the live-unlock track record. Days are marked at close time with the
        account mode, so later live days never inflate the paper count."""
        r = self.db.execute(
            "SELECT COUNT(DISTINCT date(closed_ts,'unixepoch')) FROM managed "
            "WHERE status='closed' AND COALESCE(mode,'paper')='paper'").fetchone()
        return int(r[0] or 0)

    def live_ok(self) -> bool:
        """True only if it's safe to send. Paper: always. Live: needs BOTH the
        explicit env flag AND a proven paper track record (default 60 distinct
        paper trading days) — one env var alone is not enough to go live."""
        if getattr(self.broker, "paper", True):
            return True
        if not _env_flag("QTSYS_AUTOTRADE_LIVE"):
            return False
        return self.paper_days_req <= 0 or self.paper_days() >= self.paper_days_req

    def status(self) -> dict:
        return {"enabled": self.enabled, "live_ok": self.live_ok(),
                "paper": getattr(self.broker, "paper", True),
                "orders_today": self._orders_today(),
                "max_orders_day": self.max_orders_day,
                "open": len(self.open_positions()),
                "max_concurrent": self.max_concurrent,
                "max_daily_loss": self.max_daily_loss,
                "max_symbol_pct": self.max_symbol_pct,
                "paper_days": self.paper_days(),
                "paper_days_req": self.paper_days_req,
                "require_dsr": self.require_dsr,
                "dsr_threshold": self.dsr_threshold,
                "options_on": self.options_on,
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
        import json
        rows = self.db.execute(
            "SELECT id,symbol,side,qty,entry,stop,target,opened_ts,"
            "COALESCE(kind,'equity'),legs,expiration FROM managed "
            "WHERE status='open' ORDER BY opened_ts DESC").fetchall()
        return [{"id": r[0], "symbol": r[1], "side": r[2], "qty": r[3],
                 "entry": r[4], "stop": r[5], "target": r[6], "opened_ts": r[7],
                 "kind": r[8], "legs": json.loads(r[9]) if r[9] else None,
                 "expiration": r[10]}
                for r in rows]

    # ------------------------------------------------------- guardrail check
    def _blocked(self) -> str | None:
        if not self.enabled:
            return "engine disarmed"
        if not self.live_ok():
            if not _env_flag("QTSYS_AUTOTRADE_LIVE"):
                return "live keys but QTSYS_AUTOTRADE_LIVE not set — refusing"
            return (f"live locked: {self.paper_days()}/{self.paper_days_req} "
                    "paper trading days proven — keep running on paper")
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
        try:
            equity = float(self.broker.get_account().get("equity") or 0)
        except Exception:
            equity = 0.0
        sym_cap = equity * self.max_symbol_pct if equity else None
        exposure = {}                               # managed notional per symbol
        for p in self.open_positions():
            exposure[p["symbol"]] = (exposure.get(p["symbol"], 0.0)
                                     + abs(p["qty"] * p["entry"]))
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
            if self.require_dsr:
                d = idea.get("dsr")
                ok = ((d is not None and d >= self.dsr_threshold)
                      or (d is None and idea.get("verified")))
                if not ok:
                    skipped.append((sym, f"DSR {d if d is not None else '—'} < "
                                    f"{self.dsr_threshold:g} gate — needs INBOX "
                                    "approval"))
                    continue
            want = abs(float(idea.get("notional") or 0))
            if sym_cap and exposure.get(sym, 0.0) + want > sym_cap:
                skipped.append((sym, f"per-symbol cap: {exposure.get(sym, 0.0) + want:,.0f}"
                                f" > {self.max_symbol_pct:.0%} of equity"))
                continue
            # options alternative: defined-risk vertical instead of shares,
            # only when the operator has switched options trading on
            if self.options_on and idea.get("options_alt"):
                ok, why = self._enter_spread(idea, plan.get("date", self._today()))
                if ok:
                    done += 1
                    existing.add(sym)
                else:
                    skipped.append((sym, why))
                continue
            side = "buy" if idea["side"] in ("LONG", "buy") else "sell"
            o = Order(self._venue(sym), side, float(idea["qty"]), "market", None)
            res = self.gw.submit(o)
            self._bump_orders()
            if res.status == "rejected":
                skipped.append((sym, f"gateway: {res.reason}")); continue
            self.db.execute(
                "INSERT INTO managed(plan_date,symbol,side,qty,entry,stop,target,"
                "order_id,status,opened_ts,realized,mode) VALUES (?,?,?,?,?,?,?,?, "
                "'open', ?, 0, ?)",
                (plan.get("date", self._today()), sym, side, float(idea["qty"]),
                 idea["entry"], idea["stop"], idea["target"], res.id or "",
                 time.time(),
                 "paper" if getattr(self.broker, "paper", True) else "live"))
            self.db.commit()
            exposure[sym] = exposure.get(sym, 0.0) + want
            done += 1
            self.log("AutoTrader",
                     f"ENTERED {side.upper()} {idea['qty']} {sym} @~{idea['entry']} "
                     f"(stop {idea['stop']} / target {idea['target']})", "warn")
        if done:
            self.notify("QTSYS · plan executed",
                        f"{done} positions entered from the {plan.get('date')} plan",
                        "high")
        return {"executed": done, "skipped": skipped}

    def _enter_spread(self, idea: dict, plan_date: str) -> tuple[bool, str]:
        """Enter a defined-risk vertical (from optexec.pick_spread). Max loss
        is the prepaid debit — sized inside the idea's risk budget already."""
        import json
        sp = idea["options_alt"]
        if not hasattr(self.broker, "option_spread_order"):
            return False, "venue has no multi-leg options support"
        # marketable limit: debit per share + 2% buffer, so all legs fill
        # together or the order rests — never legged in
        lim = round(sp["debit_per"] / 100.0 * 1.02, 2)
        res = self.broker.option_spread_order(sp["legs"], sp["contracts"], lim)
        self._bump_orders()
        if res.get("status") == "rejected":
            return False, f"spread: {res.get('reason', 'rejected')}"
        self.db.execute(
            "INSERT INTO managed(plan_date,symbol,side,qty,entry,stop,target,"
            "order_id,status,opened_ts,realized,mode,kind,legs,expiration) "
            "VALUES (?,?,?,?,?,?,?,?, 'open', ?, 0, ?, 'ospread', ?, ?)",
            (plan_date, idea["symbol"], idea["side"], sp["contracts"],
             sp["debit_per"], sp["exit"]["stop_value"],
             sp["exit"]["target_value"], res.get("id", ""), time.time(),
             "paper" if getattr(self.broker, "paper", True) else "live",
             json.dumps(sp["legs"]), sp.get("expiration", "")))
        self.db.commit()
        self.log("AutoTrader",
                 f"ENTERED {sp['preset']} {sp['contracts']}x {idea['symbol']} "
                 f"debit ${sp['debit_per']}/contract (max loss "
                 f"${abs(sp['total_max_loss']):,.0f} prepaid, target "
                 f"${sp['exit']['target_value']})", "warn")
        return True, ""

    def monitor(self) -> dict:
        """Close managed positions that hit TP or SL; reconcile on halt."""
        closed = 0
        halted = getattr(self.gw, "halted", False)
        for p in self.open_positions():
            if halted:                             # kill switch flattened the book
                self._mark_closed(p["id"], "halt", None)
                continue
            if p.get("kind") == "ospread":         # defined-risk vertical
                if self._monitor_spread(p):
                    closed += 1
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

    def _monitor_spread(self, p: dict) -> bool:
        """TP/SL/time exit for a managed vertical: value the legs, apply the
        optexec exit rules, close with an MLEG order when one fires."""
        from . import optexec
        val = optexec.spread_value(p["legs"] or [], self.broker.get_quote)
        spread = {"exit": {"target_value": p["target"], "stop_value": p["stop"],
                           "time_exit_days": optexec.TIME_EXIT_DAYS},
                  "expiration": p.get("expiration", "")}
        reason = optexec.exit_check(val, spread)
        if not reason:
            return False
        lim = round(max((val if val is not None else p["entry"]), 0.01)
                    / 100.0 * 0.98, 2)             # marketable close limit
        res = self.broker.option_spread_order(p["legs"], int(p["qty"]), lim,
                                              close=True)
        self._bump_orders()
        if res.get("status") == "rejected":
            self.log("AutoTrader", f"spread close REJECTED {p['symbol']}: "
                     f"{res.get('reason')}", "error")
            return False
        realized = ((val - p["entry"]) * p["qty"]) if val is not None else None
        self._mark_closed(p["id"], reason, realized)
        self.log("AutoTrader",
                 f"{'🎯' if reason == 'target' else '🛑' if reason == 'stop' else '⏳'} "
                 f"spread {reason.upper()} {p['symbol']} @ ${val} — realized "
                 f"{realized:+.2f}" if realized is not None else
                 f"spread {reason.upper()} {p['symbol']} (unpriced)", "warn")
        self.notify(f"QTSYS · spread {reason} · {p['symbol']}",
                    f"closed {p['qty']}x vertical, P&L "
                    f"{realized:+.2f}" if realized is not None else
                    f"closed {p['qty']}x vertical", "high")
        return True

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
    def option_spread_order(self, legs, contracts, limit_price, close=False):
        return {"status": "accepted", "id": "mleg1"}


def _selftest():
    import tempfile
    from .brokers import Order as _RealOrder      # ensure Order import path works
    at = AutoTrader(_FakeGW(), _FakeBroker(), db_path=tempfile.mktemp(suffix=".db"))
    assert not at.enabled, "disarmed by default"
    assert at.execute_plan({"ideas": [{}]})["blocked"] == "engine disarmed"
    at.set_enabled(True)
    plan = {"date": at._today(), "ideas": [
        {"symbol": "AAPL", "side": "LONG", "qty": 10, "entry": 200.0,
         "stop": 194.0, "target": 212.0, "verified": True}]}
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
    # per-symbol exposure cap: 10% of 100k equity = 10k; a 20k idea is blocked
    big = {"date": at._today(), "ideas": [
        {"symbol": "MSFT", "side": "LONG", "qty": 100, "entry": 200.0,
         "stop": 194.0, "target": 212.0, "notional": 20000.0,
         "verified": True}]}
    r = at.execute_plan(big)
    assert r["executed"] == 0 and "per-symbol cap" in r["skipped"][0][1], r
    # DSR gate: an unverified idea is skipped for INBOX approval
    unv = {"date": at._today(), "ideas": [
        {"symbol": "NFLX", "side": "LONG", "qty": 1, "entry": 500.0,
         "stop": 490.0, "target": 520.0, "notional": 500.0}]}
    r = at.execute_plan(unv)
    assert r["executed"] == 0 and "DSR" in r["skipped"][0][1], r
    # live-safety gate 1: a live broker without the flag refuses
    at.broker.paper = False
    os.environ.pop("QTSYS_AUTOTRADE_LIVE", None)
    assert at.execute_plan(plan)["blocked"].startswith("live keys"), "live gate"
    # live-safety gate 2: flag set but paper record short -> still locked
    os.environ["QTSYS_AUTOTRADE_LIVE"] = "1"
    at.paper_days_req = 3
    assert "live locked" in at.execute_plan(plan)["blocked"], "paper-days lock"
    # accrue 3 distinct PAPER days (plus a LIVE day that must NOT count)
    for d in range(4):
        at.db.execute("INSERT INTO managed(plan_date,symbol,side,qty,entry,stop,"
                      "target,order_id,status,opened_ts,closed_ts,exit_reason,"
                      "realized,mode) VALUES ('x','T','buy',1,1,1,1,'','closed',"
                      "?,?,'target',1,?)",
                      (0, 86400 * (d + 1) + 60, "live" if d == 3 else "paper"))
    at.db.commit()
    # 3 synthetic paper days + today's own TP/SL closes = 4; the LIVE row
    # would make 5 if it (wrongly) counted
    assert at.paper_days() == 4, f"live day must not count: {at.paper_days()}"
    assert at.live_ok(), "paper days >= req + flag -> live unlocked"
    at.paper_days_req = 60
    assert not at.live_ok(), "60-day requirement re-locks"
    os.environ.pop("QTSYS_AUTOTRADE_LIVE", None)
    # ---- options spread lifecycle (defined-risk vertical) ----
    at.broker.paper = True
    at.options_on = True
    import datetime as _dt
    exp = str(_dt.date.today() + _dt.timedelta(days=10))
    sp = {"kind": "ospread", "preset": "bull_call", "side": "LONG",
          "expiration": exp, "contracts": 2,
          "legs": [{"symbol": "OC300", "qty": 1, "right": "call",
                    "strike": 300.0, "mid": 6.0},
                   {"symbol": "OC310", "qty": -1, "right": "call",
                    "strike": 310.0, "mid": 2.0}],
          "debit_per": 400.0, "max_loss_per": -400.0, "max_profit_per": 600.0,
          "breakevens": [304.0], "total_debit": 800.0, "total_max_loss": -800.0,
          "exit": {"target_value": 760.0, "stop_value": 200.0,
                   "time_exit_days": 1}}
    oplan = {"date": at._today(), "ideas": [
        {"symbol": "MSFT2", "side": "LONG", "qty": 5, "entry": 200.0,
         "stop": 194.0, "target": 212.0, "notional": 1000.0, "verified": True,
         "options_alt": sp}]}
    r = at.execute_plan(oplan)
    assert r["executed"] == 1, r
    pos = [p for p in at.open_positions() if p["kind"] == "ospread"]
    assert pos and pos[0]["legs"][0]["symbol"] == "OC300", "spread managed"
    # legs at entry mids -> value 400 = debit -> hold
    at.broker.px.update({"OC300": 6.0, "OC310": 2.0})
    assert at.monitor()["closed"] == 0
    # rally: long leg 9.8, short 1.2 -> value 860 >= 760 target -> close
    at.broker.px.update({"OC300": 9.8, "OC310": 1.2})
    assert at.monitor()["closed"] == 1
    assert not [p for p in at.open_positions() if p["kind"] == "ospread"]
    row = at.db.execute("SELECT exit_reason, realized FROM managed WHERE "
                        "kind='ospread' AND status='closed'").fetchone()
    assert row[0] == "target" and abs(row[1] - (860 - 400) * 2) < 1e-6, row
    # options off -> options_alt idea falls through to the SHARES path
    at.options_on = False
    r2 = at.execute_plan({"date": at._today(), "ideas": [
        {"symbol": "MSFT3", "side": "LONG", "qty": 2, "entry": 200.0,
         "stop": 194.0, "target": 212.0, "notional": 400.0, "verified": True,
         "options_alt": sp}]})
    assert r2["executed"] == 1
    assert [p for p in at.open_positions() if p["symbol"] == "MSFT3"][0]["kind"] == "equity"
    print("autotrader self-test ✓  disarmed default, enter->TP profit->SL loss, "
          "gateway routed, per-symbol cap, DSR gate, live-without-flag refused, "
          "60-paper-day lock, spread enter->target close (+920 realized), "
          "options-off falls back to shares")


if __name__ == "__main__":
    _selftest()
