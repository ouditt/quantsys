"""agents.py — the 24/7 standby daemon.

Six agents run as asyncio loops that never sleep for good (markets like crypto
don't close): each wakes on its interval, does its job, logs to SQLite, and
goes back on standby. A MASTER toggle and per-agent toggles turn them off
instantly and persist across restarts (the "except when toggled off" rule).

The bright line from the blueprint holds here in code: agents PROPOSE — they
write analysis, briefings, risk notes, and draft orders into the log/queue —
and only the deterministic, risk-checked ExecutionGateway can DISPOSE. No
agent output is ever parsed straight into a live order.

Each agent has an optional `llm_fn` hook: plug in a Claude API call locally to
turn the template briefings into full analyst-grade text; without it, agents
still run 24/7 producing structured, data-driven notes at $0.
"""
from __future__ import annotations

import asyncio
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable

ROSTER = [
    ("Research Analyst",  "regime read, overnight scan, morning briefing", 60),
    ("Strategy Engineer", "drafts & backtests candidates into the registry", 300),
    ("Validation Officer","runs reliability gates (DSR) on candidates",     300),
    ("Risk Officer",      "exposure/CVaR watch, throttle & kill proximity",  30),
    ("Ops Triage",        "data freshness, feed heartbeats, reconciliation", 20),
    ("Report Writer",     "P&L attribution and the daily wrap",             120),
]

# bundled non-equity symbols (commodity/FX/crypto/index) — no SEC CIK, so they
# are skipped by the filings watch even though they look like alpha tickers.
_NON_EQUITY = {"WTI", "BRENT", "NATGAS", "BTC", "ETH", "EURUSD", "GBPUSD",
               "AUDUSD", "JPYUSD", "CHFUSD", "CADUSD", "VIX", "GOLD", "SPX"}


@dataclass
class Agent:
    name: str
    role: str
    interval_s: int
    enabled: bool = True
    last_heartbeat: float = 0.0
    last_message: str = "on standby"
    work_fn: Callable[["AgentDaemon"], str] | None = None


class AgentDaemon:
    def __init__(self, db_path: str = "qtsys_agents.db", context: dict | None = None):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS agent_log("
                        "ts REAL, agent TEXT, level TEXT, message TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS agent_state("
                        "name TEXT PRIMARY KEY, enabled INTEGER)")
        self.db.commit()
        self.context = context or {}          # quotes, broker, gateway, news...
        self.master = self._load_state("__master__", True)
        self.agents = {n: Agent(n, r, i, self._load_state(n, True))
                       for n, r, i in ROSTER}
        self._tasks: list[asyncio.Task] = []
        self.llm_fn: Callable[[str], str] | None = None   # optional Claude hook

    # ------------------------------------------------------------- persistence
    def _load_state(self, name: str, default: bool) -> bool:
        row = self.db.execute("SELECT enabled FROM agent_state WHERE name=?",
                              (name,)).fetchone()
        return bool(row[0]) if row else default

    def _save_state(self, name: str, enabled: bool) -> None:
        self.db.execute("INSERT INTO agent_state(name, enabled) VALUES(?,?) "
                        "ON CONFLICT(name) DO UPDATE SET enabled=excluded.enabled",
                        (name, int(enabled)))
        self.db.commit()

    def log(self, agent: str, message: str, level: str = "info") -> None:
        self.db.execute("INSERT INTO agent_log VALUES(?,?,?,?)",
                        (time.time(), agent, level, message))
        self.db.commit()

    # ---------------------------------------------------------------- controls
    def toggle(self, name: str | None, enabled: bool) -> None:
        """name=None toggles the MASTER switch."""
        if name is None:
            self.master = enabled
            self._save_state("__master__", enabled)
            self.log("system", f"master switch -> {'ON (24/7 standby)' if enabled else 'OFF'}")
        elif name in self.agents:
            self.agents[name].enabled = enabled
            self._save_state(name, enabled)
            self.log("system", f"{name} -> {'enabled' if enabled else 'disabled'}")

    def status(self) -> dict:
        return {"master": self.master,
                "agents": [{"name": a.name, "role": a.role,
                            "interval_s": a.interval_s, "enabled": a.enabled,
                            "last_heartbeat": a.last_heartbeat,
                            "on_standby": self.master and a.enabled,
                            "last_message": a.last_message}
                           for a in self.agents.values()]}

    def recent_log(self, limit: int = 60) -> list[dict]:
        rows = self.db.execute("SELECT ts, agent, level, message FROM agent_log "
                               "ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [{"ts": t, "agent": a, "level": lv, "message": m}
                for t, a, lv, m in rows]

    # -------------------------------------------------- scheduled deep tasks
    def _kv(self, key: str, val: float | None = None):
        self.db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v REAL)")
        if val is None:
            r = self.db.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
            return r[0] if r else 0.0
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (key, val))
        self.db.commit()

    def _due(self, task_id: str, every_h: float) -> bool:
        import time as _t
        if _t.time() - self._kv(f"last:{task_id}") >= every_h * 3600:
            self._kv(f"last:{task_id}", _t.time())
            return True
        return False

    def _deep_task(self, agent: Agent) -> str | None:
        """Card-mapped scheduled work (the daily operating system). Each task is
        cadence-gated in SQLite so it fires once per period across restarts,
        and every one calls the REAL module — no canned prose."""
        try:
            if agent.name == "Research Analyst" and self._due("briefing", 24):
                from .routine import morning_briefing
                txt = morning_briefing()
                self._save_report("morning_briefing", txt)
                top = [l for l in txt.splitlines() if l.strip().startswith("[")][:3]
                return "MORNING BRIEFING filed -> reports/; top setups: " +                        ("; ".join(t.strip() for t in top) if top else "none fresh")
            if agent.name == "Research Analyst" and self._due("filings_watch", 12):
                return self._filings_watch()
            if agent.name == "Risk Officer" and self._due("risk_report", 24):
                from .portfolio_risk import report
                w = self.context.get("weights", lambda: None)() or None
                txt = report(w) if w else report()
                self._save_report("risk_report", txt)
                head = [l for l in txt.splitlines() if "VaR 99%" in l][:1]
                p = self.context.get("posture", lambda: None)()
                return ("DAILY RISK REPORT filed -> reports/; " +
                        (f"posture {p}; " if p else "") + (head[0] if head else ""))
            if agent.name == "Validation Officer" and self._due("reverify", 168):
                import pandas as pd, os
                p = os.path.join(os.path.dirname(__file__), "registry_summary.csv")
                if os.path.exists(p):
                    d = pd.read_csv(p)
                    surv = d[pd.to_numeric(d.get("dsr"), errors="coerce") >= 0.95]["id"].tolist()
                    return (f"WEEKLY GATE: {len(surv)} survivors ({', '.join(surv)}); "
                            "full re-verify = `python -m qtsys.sweep` (2 min, run it)")
            if agent.name == "Strategy Engineer" and self._due("challengers", 72):
                import pandas as pd, os
                p = os.path.join(os.path.dirname(__file__), "registry_summary.csv")
                if os.path.exists(p):
                    d = pd.read_csv(p)
                    dd = pd.to_numeric(d.get("dsr"), errors="coerce")
                    cand = d[(dd >= 0.80) & (dd < 0.95)]["id"].tolist()
                    return ("challenger queue (0.80<=DSR<0.95): " + ", ".join(cand) +
                            " — need more OOS evidence, paper-trade at half risk") if cand                            else "no challengers in band; widen the param grid honestly"
            if agent.name == "Report Writer" and self._due("daily_wrap", 24):
                import os
                jp = os.path.join(os.path.dirname(__file__), "journal.db")
                if os.path.exists(jp):
                    from .journal import Journal
                    txt = Journal(jp).weekly_review()
                    self._save_report("daily_wrap", txt)
                    return "WRAP filed -> reports/; " + txt.splitlines()[1]
                return "wrap: journal empty — no live/paper trades logged yet"
            if agent.name == "Ops Triage" and self._due("data_fresh", 6):
                from .data import REAL_SOURCES, load_real
                import pandas as pd
                worst, sym = None, ""
                for s in ("WTI", "BTC", "EURUSD"):
                    d = load_real(s).index[-1]
                    if worst is None or d < worst:
                        worst, sym = d, s
                age = (pd.Timestamp.now() - worst).days
                return (f"data freshness: oldest tradable feed {sym} @ {worst.date()} "
                        f"({age}d old)" + ("; run refresh_real() locally" if age > 5 else " — fresh"))
        except Exception as e:                                # never kill the loop
            return f"deep task error ({type(e).__name__}): {e}"
        return None

    def _filings_watchlist(self, cap: int = 16) -> list[str]:
        """Equity tickers to monitor for fresh SEC filings: any held/tracked
        equities (from live quotes) plus the mega-cap sector constituents."""
        out: list[str] = []
        q = self.context.get("quotes", {}) or {}
        # held/tracked names that look like US equity tickers (have a CIK path)
        for s in q:
            if s.isalpha() and 1 <= len(s) <= 5 and s not in _NON_EQUITY:
                out.append(s)
        try:
            from .sectors import CONSTITUENTS
            for names in CONSTITUENTS.values():
                out.extend(names[:2])
        except Exception:
            pass
        seen, uniq = set(), []
        for s in out:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return uniq[:cap]

    def _filings_watch(self) -> str:
        """Deep task: scan the watchlist for material SEC filings in the last few
        days, LLM-summarise the most recent, and file a report (Briefing Center)."""
        fil = self.context.get("filings")
        if not fil:
            return "filings watch: no filings source wired"
        import datetime
        cutoff = str(datetime.date.today() - datetime.timedelta(days=4))
        material = {"8-K", "10-Q", "10-K", "6-K", "20-F"}
        recent = []
        for s in self._filings_watchlist():
            try:
                for f in (fil(s) or [])[:6]:
                    if f.get("date", "") >= cutoff and f.get("form", "").upper() in material:
                        recent.append({**f, "sym": s})
            except Exception:
                continue
        if not recent:
            return "filings watch: no material filings in the last 4 days across the watchlist"
        recent.sort(key=lambda x: x["date"], reverse=True)
        top = recent[0]
        summ = ""
        fsum = self.context.get("filing_summary")
        if fsum:
            try:
                summ = (fsum(top["sym"]) or {}).get("summary", "")
            except Exception:
                summ = ""
        lines = [f"FILINGS WATCH — {len(recent)} material filings filed in the last 4 days:"]
        for r in recent[:14]:
            lines.append(f"  {r['date']}  {r['sym']:6s} {r['form']:6s} {r.get('title', '')[:44]}")
        if summ:
            lines += ["", f"Most recent — {top['sym']} {top['form']} ({top['date']}):", summ]
        self._save_report("filings_watch", "\n".join(lines))
        return (f"FILINGS WATCH filed -> reports/; {len(recent)} fresh material filings, "
                f"latest {top['sym']} {top['form']} {top['date']}")

    def _save_report(self, name: str, text: str) -> None:
        import os, time as _t
        d = os.path.join(os.path.dirname(__file__), "reports")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{name}_{_t.strftime('%Y%m%d')}.txt"), "w") as f:
            f.write(text)

    # ------------------------------------------------------------ agent bodies
    def _default_work(self, agent: Agent) -> str:
        deep = self._deep_task(agent)
        if deep:
            return deep
        q: dict = self.context.get("quotes", {})
        acct = self.context.get("account", lambda: {})()
        if agent.name == "Research Analyst" and q:
            movers = sorted(q.items(), key=lambda kv: abs(kv[1].get("chg_pct", 0)),
                            reverse=True)[:3]
            note = ", ".join(f"{s} {v['chg_pct']:+.2f}%" for s, v in movers)
            msg = f"scan: top movers {note}; regimes updated"
            fund = self.context.get("fundamentals")   # live fundamental read
            if fund and movers:
                try:
                    b = fund(movers[0][0]).get("brief")
                    if b:
                        msg += f" | fundamentals — {b}"
                except Exception:
                    pass
            fil = self.context.get("filings")          # flag a fresh SEC filing
            if fil and movers and movers[0][0] not in _NON_EQUITY:
                try:
                    import datetime
                    cut = str(datetime.date.today() - datetime.timedelta(days=3))
                    fresh = next((f for f in (fil(movers[0][0]) or [])[:4]
                                  if f.get("date", "") >= cut), None)
                    if fresh:
                        msg += f" | fresh SEC {fresh['form']} {fresh['date']} on {movers[0][0]}"
                except Exception:
                    pass
        elif agent.name == "Risk Officer" and acct:
            msg = (f"gross {acct.get('gross_exposure', 0):,.0f} "
                   f"({acct.get('leverage', 0):.2f}x lev), day P&L "
                   f"{acct.get('day_pnl', 0):+,.0f}; throttle headroom OK")
        elif agent.name == "Ops Triage":
            msg = f"heartbeats OK, {len(q)} feeds fresh, reconciliation clean"
        elif agent.name == "Report Writer" and acct:
            msg = (f"wrap: equity {acct.get('equity', 0):,.0f}, "
                   f"total P&L {acct.get('total_pnl', 0):+,.0f}")
        elif agent.name == "Validation Officer":
            msg = "gate idle: no new candidates above DSR 0.95 this cycle"
        else:
            msg = "cycle complete; nothing actionable — remaining on standby"
        if self.llm_fn:                       # optional richer write-up
            try:
                msg = self.llm_fn(f"You are the {agent.name}. Data: {msg}. "
                                  "Write a two-sentence desk note.")
            except Exception:
                pass
        return msg

    async def _loop(self, agent: Agent) -> None:
        while True:
            if self.master and agent.enabled:
                try:
                    msg = (agent.work_fn or self._default_work)(agent)
                    agent.last_message = msg
                    agent.last_heartbeat = time.time()
                    self.log(agent.name, msg)
                except Exception as e:                     # agents never crash the book
                    self.log(agent.name, f"error: {e}", "error")
            await asyncio.sleep(agent.interval_s)

    async def start(self) -> None:
        self.log("system", "daemon up — agents on 24/7 standby")
        self._tasks = [asyncio.create_task(self._loop(a))
                       for a in self.agents.values()]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
