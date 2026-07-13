"""riskgov.py — the adaptive risk governor: "high CALCULATED risk".

A single book-level size multiplier in [0.25, 1.5] that scales risk UP only when
the live edge is proven and the book is calm, and cuts it on drawdown, losing
streaks, or elevated volatility. It never overrides a hard cap — the multiplier
feeds `tradeplan.draft`'s risk_pct BEFORE the small-account floor and stays
inside the 8% blow-up guard and every gateway check.

The bright line (asserted in the self-test): the edge BONUS can only ever fire
when the drawdown throttle AND the streak multiplier are BOTH exactly 1.0. So a
drawdown or a losing streak makes it structurally impossible to size up — the
governor can raise risk only from a position of proven strength, never to chase
losses.

State in riskgov.db (gitignored, WAL). All maths deterministic — no LLM.

Run:  python -m qtsys.riskgov        (synthetic drawdown/weekly-loss/bounds)
"""
from __future__ import annotations

import datetime
import json
import os
import sqlite3
import time

from . import sizing

HERE = os.path.dirname(__file__)

MULT_LO, MULT_HI = 0.25, 1.5
DD_HALT = -0.12            # drawdown from persistent peak that halts the engine
WEEK_GATE = -0.06         # weekly loss that gates new entries until next ISO week
PRUNE_DAYS = 90           # equity-curve retention


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class RiskGovernor:
    def __init__(self, db_path: str | None = None, journal_db: str | None = None,
                 log=None, notify=None, vix_fn=None):
        self.db = sqlite3.connect(db_path or os.path.join(HERE, "riskgov.db"),
                                  check_same_thread=False)
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS equity_curve(ts REAL PRIMARY KEY, equity REAL);
        CREATE TABLE IF NOT EXISTS gov_log(ts REAL, multiplier REAL,
          components TEXT, reasons TEXT);
        CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);""")
        self.db.commit()
        self.journal_db = journal_db or os.path.join(HERE, "journal.db")
        self.log = log or (lambda *a, **k: None)
        self.notify = notify or (lambda *a, **k: None)
        self.vix_fn = vix_fn                       # best-effort VIX reader, optional

    # ---------------------------------------------------------------- kv
    def _get(self, k, d=None):
        r = self.db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return r[0] if r else d

    def _set(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (k, str(v)))
        self.db.commit()

    # ---------------------------------------------------------- equity curve
    def record_equity(self, equity: float, min_interval_s: float = 60) -> bool:
        """Append the current equity, throttled to at most one point per
        min_interval_s, then prune anything older than PRUNE_DAYS. Returns True
        if a point was written."""
        if not equity or equity <= 0:
            return False
        now = time.time()
        last = self.db.execute("SELECT MAX(ts) FROM equity_curve").fetchone()[0]
        if last is not None and now - last < min_interval_s:
            return False
        self.db.execute("INSERT OR REPLACE INTO equity_curve VALUES (?,?)",
                        (now, float(equity)))
        self.db.execute("DELETE FROM equity_curve WHERE ts < ?",
                        (now - PRUNE_DAYS * 86400,))
        self.db.commit()
        return True

    def _curve(self) -> list[tuple]:
        return self.db.execute("SELECT ts, equity FROM equity_curve "
                               "ORDER BY ts").fetchall()

    def _daily_series(self) -> list[tuple]:
        """Last equity per calendar day, chronological — the basis for the
        drawdown, weekly change, Sharpe and realised-vol reads."""
        byday: dict[str, tuple] = {}
        for ts, eq in self._curve():
            d = datetime.date.fromtimestamp(ts).isoformat()
            byday[d] = (ts, eq)               # last write of the day wins
        return [(d, v[1]) for d, v in sorted(byday.items())]

    # ---------------------------------------------------------- journal reads
    def _journal_returns(self, limit: int | None = None) -> list[float]:
        if not os.path.exists(self.journal_db):
            return []
        try:
            from .journal import Journal
            import pandas as pd
            df = Journal(self.journal_db).frame()
            r = pd.to_numeric(df.get("net_ret"), errors="coerce").dropna().tolist()
            return r[-limit:] if limit else r
        except Exception:
            return []

    def _backtest_ref(self) -> float:
        """Book-level backtest expectancy reference: mean certified test_exp of
        the DSR-verified survivors (the edge the book is supposed to have)."""
        path = os.path.join(HERE, "registry_summary.csv")
        if not os.path.exists(path):
            return 0.0
        try:
            import pandas as pd
            d = pd.read_csv(path)
            dsr = pd.to_numeric(d.get("dsr"), errors="coerce")
            exp = pd.to_numeric(d.get("test_exp"), errors="coerce")
            surv = exp[(dsr >= 0.95) & exp.notna()]
            return float(surv.mean()) if len(surv) else 0.0
        except Exception:
            return 0.0

    # ---------------------------------------------------------------- reads
    def snapshot(self) -> dict:
        daily = self._daily_series()
        equity = daily[-1][1] if daily else 0.0
        curve_eq = [e for _, e in daily]
        peak = max(curve_eq) if curve_eq else equity
        drawdown = (equity / peak - 1.0) if peak else 0.0
        # weekly change: equity now vs the last point >= ~7 days ago
        week_change = 0.0
        if len(daily) >= 2:
            cutoff = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
            ref = next((e for d, e in daily if d <= cutoff), daily[0][1])
            week_change = (equity / ref - 1.0) if ref else 0.0
        rets = [curve_eq[i] / curve_eq[i - 1] - 1.0 for i in range(1, len(curve_eq))]
        r30 = rets[-30:]
        sharpe = None
        if len(r30) >= 5:
            import statistics
            sd = statistics.pstdev(r30)
            sharpe = (statistics.fmean(r30) / sd) if sd > 0 else None
        jr = self._journal_returns(30)
        trade_exp = (sum(jr) / len(jr)) if jr else None
        streak = sizing.streak_mult([x > 0 for x in self._journal_returns(10)])
        vol_scalar, vol_regime = self._vol_scalar(rets)
        return {"equity": equity, "peak": peak, "drawdown": round(drawdown, 4),
                "week_change": round(week_change, 4),
                "daily_sharpe_30d": round(sharpe, 3) if sharpe is not None else None,
                "trade_expectancy_30": round(trade_exp, 4) if trade_exp is not None else None,
                "streak_mult": streak, "vol_regime": vol_regime,
                "n_curve": len(curve_eq)}

    def _vol_scalar(self, rets: list[float]) -> tuple[float, str]:
        """Volatility de-risking: cut size when the VIX is high OR the book's own
        realised vol is running hot vs its 60-day median."""
        vix = None
        if self.vix_fn:
            try:
                vix = float(self.vix_fn())
            except Exception:
                vix = None
        book_hot = False
        if len(rets) >= 40:
            import statistics
            win = 10
            roll = [statistics.pstdev(rets[i - win:i])
                    for i in range(win, len(rets) + 1)]
            if roll:
                med = statistics.median(roll)
                book_hot = med > 0 and roll[-1] > 1.5 * med
        if vix is not None and vix > 32:
            return 0.6, "stressed"
        if (vix is not None and vix > 25) or book_hot:
            return 0.75, "elevated"
        return 1.0, "normal"

    def multiplier(self, vix: float | None = None) -> dict:
        """The book-level size multiplier, clamped to [MULT_LO, MULT_HI].

            value = clamp(dd_throttle × streak × vol_scalar × edge_bonus)

        edge_bonus > 1.0 requires ALL of: >=30 live trades, live expectancy >=
        the certified backtest reference, dd_throttle == 1.0 AND streak == 1.0.
        Thus a drawdown or losing streak makes sizing UP impossible."""
        daily = self._daily_series()
        curve_eq = [e for _, e in daily]
        equity = curve_eq[-1] if curve_eq else 0.0
        peak = max(curve_eq) if curve_eq else equity
        drawdown = (equity / peak - 1.0) if peak else 0.0
        dd_throttle = sizing.throttle_mult(drawdown)
        streak = sizing.streak_mult([x > 0 for x in self._journal_returns(10)])
        rets = [curve_eq[i] / curve_eq[i - 1] - 1.0 for i in range(1, len(curve_eq))]
        if vix is None and self.vix_fn:
            try:
                vix = float(self.vix_fn())
            except Exception:
                vix = None
        vol_scalar, vol_regime = self._vol_scalar(rets)
        if vix is not None:                       # explicit override wins
            if vix > 32:
                vol_scalar, vol_regime = 0.6, "stressed"
            elif vix > 25:
                vol_scalar, vol_regime = min(vol_scalar, 0.75), "elevated"

        edge_bonus, edge_reason = 1.0, "edge bonus off (preconditions unmet)"
        jr = self._journal_returns()
        # the bonus can ONLY be considered from a position of strength
        if dd_throttle == 1.0 and streak == 1.0 and len(jr) >= 30:
            import statistics
            exp = statistics.fmean(jr)
            ref = self._backtest_ref()
            sd = statistics.pstdev(jr)
            se = sd / (len(jr) ** 0.5) if sd > 0 else 0.0
            if exp >= ref and se > 0:
                z = (exp - ref) / se
                edge_bonus = _clamp(1.0 + 0.25 * z, 1.0, MULT_HI)
                edge_reason = (f"proven edge: live exp {exp:+.4f} >= backtest "
                               f"{ref:+.4f} (z {z:.1f}) — bonus x{edge_bonus:.2f}")
            elif exp >= ref:
                edge_reason = "edge >= backtest but variance too high to size up"

        value = _clamp(dd_throttle * streak * vol_scalar * edge_bonus, MULT_LO, MULT_HI)
        components = {"dd_throttle": dd_throttle, "streak": streak,
                      "vol_scalar": vol_scalar, "edge_bonus": round(edge_bonus, 3)}
        reasons = []
        if dd_throttle < 1.0:
            reasons.append(f"drawdown {drawdown:.1%} -> throttle x{dd_throttle:g}")
        if streak < 1.0:
            reasons.append("loss streak -> half risk until a win")
        if vol_scalar < 1.0:
            reasons.append(f"{vol_regime} vol -> x{vol_scalar:g}")
        if edge_bonus > 1.0:
            reasons.append(edge_reason)
        if not reasons:
            reasons.append("calm book, no proven size-up edge -> neutral x1.0")
        out = {"value": round(value, 3), "components": components,
               "reasons": reasons, "vol_regime": vol_regime}
        self._maybe_log(out)
        return out

    def _maybe_log(self, out: dict) -> None:
        """Append to gov_log + daemon log + notify only when the value moves by
        more than 0.05 — so a polled endpoint does not spam the log."""
        try:
            prev = float(self._get("last_mult", "1.0"))
        except Exception:
            prev = 1.0
        if abs(out["value"] - prev) <= 0.05:
            return
        self.db.execute("INSERT INTO gov_log VALUES (?,?,?,?)",
                        (time.time(), out["value"], json.dumps(out["components"]),
                         json.dumps(out["reasons"])))
        self.db.commit()
        self._set("last_mult", out["value"])
        direction = "UP" if out["value"] > prev else "DOWN"
        msg = f"risk multiplier {direction} {prev:.2f} -> {out['value']:.2f}: " \
              + "; ".join(out["reasons"])
        self.log("RiskGovernor", msg, "warn")
        self.notify("QTSYS · risk multiplier", msg,
                    "high" if direction == "DOWN" else "normal")

    def gov_log(self, limit: int = 50) -> list[dict]:
        rows = self.db.execute("SELECT ts, multiplier, components, reasons FROM "
                               "gov_log ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [{"ts": r[0], "multiplier": r[1],
                 "components": json.loads(r[2]) if r[2] else {},
                 "reasons": json.loads(r[3]) if r[3] else []} for r in rows]

    # ------------------------------------------------------------- enforcement
    def _iso_week(self) -> str:
        y, w, _ = datetime.date.today().isocalendar()
        return f"{y}-W{w:02d}"

    def entry_blocked(self) -> str | None:
        """Reason string while the weekly-loss gate is active (this ISO week),
        else None. Consumed by autotrader._blocked()."""
        if self._get("weekly_gate_week") == self._iso_week():
            return self._get("weekly_gate_reason",
                             "weekly loss gate active — new entries blocked until "
                             "next week")
        return None

    def enforce(self, gw) -> dict:
        """Book-level circuit breakers, evaluated each tick:
          * weekly loss <= WEEK_GATE -> gate new entries until the next ISO week
            (floors the multiplier via streak/throttle already; gate blocks entry).
          * drawdown <= DD_HALT from the persistent peak -> halt the gateway and
            urgent-notify; operator resume required.
        """
        snap = self.snapshot()
        actions = {"weekly_gate": False, "halted": False, "reasons": []}
        if snap["week_change"] <= WEEK_GATE and self.entry_blocked() is None:
            reason = (f"weekly loss {snap['week_change']:.1%} <= {WEEK_GATE:.0%} "
                      "— new entries gated until next week")
            self._set("weekly_gate_week", self._iso_week())
            self._set("weekly_gate_reason", reason)
            actions["weekly_gate"] = True
            actions["reasons"].append(reason)
            self.log("RiskGovernor", reason, "warn")
            self.notify("QTSYS · weekly loss gate", reason, "high")
        if snap["drawdown"] <= DD_HALT and not getattr(gw, "halted", False):
            reason = (f"drawdown {snap['drawdown']:.1%} <= {DD_HALT:.0%} from peak "
                      f"{snap['peak']:,.0f} — max-drawdown halt, operator resume "
                      "required")
            gw.halt(reason, kind="max_drawdown")
            actions["halted"] = True
            actions["reasons"].append(reason)
            self.log("RiskGovernor", reason, "error")
            self.notify("QTSYS · MAX DRAWDOWN HALT", reason, "urgent")
        return actions


# ------------------------------------------------------------------ self-test
class _FakeGW:
    def __init__(self):
        self.halted, self.halt_reason, self.halt_kind = False, "", ""
    def halt(self, reason, kind=""):
        self.halted, self.halt_reason, self.halt_kind = True, reason, kind
    def resume(self):
        self.halted, self.halt_reason, self.halt_kind = False, "", ""


def _selftest():
    import tempfile
    tmp = tempfile.mkdtemp()
    rg = RiskGovernor(os.path.join(tmp, "riskgov.db"),
                      journal_db=os.path.join(tmp, "journal.db"))

    def seed_curve(points):
        """points = list of (days_ago, equity)."""
        rg.db.execute("DELETE FROM equity_curve")
        now = time.time()
        for days_ago, eq in points:
            rg.db.execute("INSERT OR REPLACE INTO equity_curve VALUES (?,?)",
                          (now - days_ago * 86400, float(eq)))
        rg.db.commit()

    # 5) EMPTY curve -> multiplier defaults to 1.0
    assert rg.multiplier()["value"] == 1.0, "empty curve -> 1.0"
    assert rg.snapshot()["drawdown"] == 0.0

    # 1a) synthetic 13% drawdown -> fake gateway halted (max_drawdown)
    seed_curve([(20, 100_000), (10, 100_000), (1, 87_000)])
    snap = rg.snapshot()
    assert abs(snap["drawdown"] - (-0.13)) < 1e-6, snap["drawdown"]
    gw = _FakeGW()
    act = rg.enforce(gw)
    assert gw.halted and gw.halt_kind == "max_drawdown", (gw.halted, gw.halt_kind)
    assert act["halted"]
    # multiplier during a drawdown is throttled and NEVER sized up
    m = rg.multiplier()
    assert m["components"]["edge_bonus"] == 1.0, "no edge bonus in drawdown"
    assert m["value"] <= 1.0, m
    assert MULT_LO <= m["value"] <= MULT_HI, "bounded"

    # 1b) weekly -6% loss sets the gate; entries blocked this ISO week.
    # (clear any gate the 13%-drawdown week above already set, to test in isolation)
    rg.db.execute("DELETE FROM kv WHERE k LIKE 'weekly_gate%'"); rg.db.commit()
    seed_curve([(6, 100_000), (5, 100_000), (0, 93_000)])   # -7% on the week
    gw2 = _FakeGW()
    act2 = rg.enforce(gw2)
    assert act2["weekly_gate"], act2
    assert rg.entry_blocked(), "weekly gate blocks entries"

    # 1c) ANTI-MARTINGALE: edge bonus is impossible while streak<1 or throttle<1.
    # 30 trades but the last 3 are losses (streak_mult -> 0.5) -> bonus stays off.
    from .journal import Journal
    j = Journal(os.path.join(tmp, "journal.db"))
    for _ in range(27):
        j.log(setup_id="x", asset="A", side="buy", regime_trend="AUTO", net_ret=0.05)
    for _ in range(3):
        j.log(setup_id="x", asset="A", side="buy", regime_trend="AUTO", net_ret=-0.08)
    seed_curve([(30, 100_000), (10, 108_000), (0, 112_000)])   # no drawdown
    m2 = rg.multiplier()
    assert m2["components"]["streak"] == 0.5, m2["components"]
    assert m2["components"]["edge_bonus"] == 1.0, "streak<1 forbids the bonus"
    assert m2["value"] <= 1.0, m2

    # 1d) bonus CAN fire only from strength: all wins, no drawdown, calm vol
    j2path = os.path.join(tmp, "journal2.db")
    rg.journal_db = j2path
    j2 = Journal(j2path)
    for i in range(40):                            # strongly +ve, with variance
        j2.log(setup_id="x", asset="A", side="buy", regime_trend="AUTO",
               net_ret=0.12 if i % 2 else 0.18)    # mean 0.15 >> backtest ref
    seed_curve([(30, 100_000), (10, 108_000), (0, 118_000)])   # rising, at highs
    m3 = rg.multiplier()
    assert m3["components"]["dd_throttle"] == 1.0 and m3["components"]["streak"] == 1.0
    assert m3["components"]["edge_bonus"] > 1.0, "proven edge sizes up"
    assert m3["value"] <= MULT_HI, "still bounded by 1.5"

    # 1e) VIX stress cuts size regardless of edge
    assert rg.multiplier(vix=35)["components"]["vol_scalar"] == 0.6, "VIX>32 -> 0.6"
    assert rg.multiplier(vix=28)["components"]["vol_scalar"] <= 0.75, "VIX>25 -> <=0.75"

    # record_equity throttle: two immediate writes -> only one point added
    rg.db.execute("DELETE FROM equity_curve"); rg.db.commit()
    assert rg.record_equity(100_000) is True
    assert rg.record_equity(100_100) is False, "throttled within min_interval"

    import shutil
    rg.db.close()
    shutil.rmtree(tmp, ignore_errors=True)
    print("riskgov self-test ✓  empty->1.0, 13% drawdown->max_drawdown halt, "
          "weekly -7%->entry gate, anti-martingale (streak/drawdown forbid the "
          "edge bonus), proven-edge size-up bounded 1.5, VIX stress cut, "
          "throttled equity recording")


if __name__ == "__main__":
    _selftest()
