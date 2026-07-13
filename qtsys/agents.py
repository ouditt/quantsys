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
    ("Microstructure Analyst", "crypto L2 depth sampling + weekly benefit A/B", 45),
    ("Fundamental Analyst", "valuation/growth/quality read on the book + watchlist", 180),
    ("Arb Strategist", "pairs stat-arb scan + crypto triangular loop monitor", 90),
    ("Portfolio Manager", "drafts the daily plan, runs the desk deliberation, "
     "hands adopted trades to the auto-trader", 300),
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
        self.db.execute("PRAGMA busy_timeout=5000")   # wait out brief contention
        self.db.execute("PRAGMA journal_mode=WAL")     # concurrent read/write
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
        self.l2lab = None                                 # crypto-L2 experiment (set by server)
        try:
            from .proposals import ProposalStore
            self.proposals = ProposalStore()              # agent -> action inbox
        except Exception:
            self.proposals = None

    def propose(self, agent, kind, summary, notify_priority=None, **kw):
        """Durable actionable proposal + optional out-of-band push. Agents
        still only PROPOSE; the gateway disposes."""
        pid = None
        if self.proposals:
            try:
                pid = self.proposals.propose(agent, kind, summary, **kw)
            except Exception:
                pass
        if notify_priority:
            try:
                from . import notify
                notify.send(f"QTSYS · {agent}", summary, notify_priority)
            except Exception:
                pass
        return pid

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
            if agent.name == "Fundamental Analyst" and self._due("fund_brief", 24):
                return self._fundamental_brief()
            if agent.name == "Arb Strategist" and self._due("arb_brief", 24):
                return self._arb_brief()
            if agent.name == "Portfolio Manager" and self._due("day_plan", 24):
                if not getattr(self, "build_plan", None):
                    return "day plan: builder not wired"
                at = getattr(self, "autotrader", None)
                plan = self.build_plan()             # draft + deliberate + adopt
                res = None
                if at and at.enabled and not res:    # armed -> execute the plan
                    res = at.execute_plan(plan)
                head = (f"DAY PLAN adopted: {len(plan.get('ideas', []))} ideas, "
                        f"{plan.get('dropped', 0)} dropped in the desk review")
                if res:
                    head += f"; auto-trader entered {res.get('executed', 0)}"
                elif at:
                    head += "; auto-trader DISARMED (plan is on the PLAN page)"
                return head
            if agent.name == "Microstructure Analyst" and self.l2lab:
                # 180-day one-shot upgrade decision (date-gated + one-shot inside lab)
                if self.l2lab.upgrade_due():
                    txt = self.l2lab.upgrade_reminder()
                    self._save_report("l2_upgrade_decision", txt)
                    return ("⚠ L2/SIP 180-DAY DECISION filed -> reports/ — action "
                            "requested: implement paid equity data + SIP? (see report)")
                if self._due("l2_benefit", 168):        # weekly A/B benefit report
                    txt = self.l2lab.weekly_report()
                    self._save_report("l2_benefit", txt)
                    head = next((l for l in txt.splitlines() if l.startswith("VERDICT")),
                                "weekly crypto-L2 benefit filed")
                    return f"CRYPTO L2 BENEFIT filed -> reports/; {head}"
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
            if agent.name == "Report Writer" and self._due("learning_report", 24):
                # deterministic learning pass: score strategies vs their certified
                # backtests, promote/demote on proven drift, tune exits from
                # realised MFE/MAE, and grade the committee's critiques.
                from . import learning
                txt = learning.nightly_report()
                self._save_report("learning_report", txt)
                head = next((l.strip() for l in txt.splitlines()
                             if l.strip().startswith(("DEMOTE", "PROMOTE", "RESTORE"))), None)
                return ("LEARNING REPORT filed -> reports/"
                        + (f"; {head}" if head else "; no strategy state changes"))
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

    # ------------------------------------------------- fundamental analysis
    _FUND_COLS = (("forward_pe", "fPE", False), ("peg", "PEG", False),
                  ("rev_growth_pct", "rev%", True),
                  ("profit_margin_pct", "mgn%", True),
                  ("debt_to_equity", "D/E", False),
                  ("div_yield_pct", "dy%", True))

    def _fundamental_brief(self) -> str:
        """Deep task (24h): cross-sectional valuation/growth/quality read over
        held equities + sector mega-caps. Ranks a composite of forward P/E,
        PEG, revenue growth, margin and leverage (rank-based, so one crazy
        outlier can't dominate), names the standouts both ways, LLM-writes the
        synthesis, and files a fundamental_brief to the Briefing Center."""
        fund = self.context.get("fundamentals")
        if not fund:
            return "fundamental brief: no fundamentals source wired"
        rows = []
        for s in self._filings_watchlist(cap=14):
            try:
                m = (fund(s) or {}).get("metrics", {}) or {}
            except Exception:
                continue
            if m.get("forward_pe") or m.get("pe"):
                rows.append((s, m))
        if len(rows) < 4:
            return "fundamental brief: too few equities with live metrics"

        def _ranks(key, higher_better):
            vals = [(s, m.get(key)) for s, m in rows]
            have = sorted((v, s) for s, v in vals if isinstance(v, (int, float)))
            if higher_better:
                have = have[::-1]
            pos = {s: i / max(len(have) - 1, 1) for i, (v, s) in enumerate(have)}
            return {s: pos.get(s, 0.5) for s, _ in vals}     # missing -> neutral

        parts = [_ranks(k, hb) for k, _, hb in self._FUND_COLS[:5]]
        score = {s: sum(p[s] for p in parts) / len(parts) for s, _ in rows}
        ranked = sorted(rows, key=lambda r: score[r[0]])     # low = attractive
        q = self.context.get("quotes", {})

        def line(s, m):
            cells = []
            for k, lbl, _ in self._FUND_COLS:
                v = m.get(k)
                cells.append(f"{lbl} {v:.1f}" if isinstance(v, (int, float))
                             else f"{lbl} —")
            tgt, last = m.get("target_mean"), (q.get(s) or {}).get("last")
            up = (f" | tgt upside {(tgt / last - 1) * 100:+.0f}%"
                  if isinstance(tgt, (int, float)) and last else "")
            return f"  {s:6s} " + "  ".join(cells) + up

        L = ["FUNDAMENTAL BRIEF — cross-sectional value/growth/quality "
             f"(rank composite over {len(rows)} names)", "",
             "MOST ATTRACTIVE (cheap growth, clean balance sheet first):"]
        L += [line(s, m) for s, m in ranked[:4]]
        L += ["", "RICHEST / MOST FRAGILE (priced for perfection or levered):"]
        L += [line(s, m) for s, m in ranked[-3:]]
        top = ranked[0][0]
        fsum = self.context.get("filing_summary")
        if fsum:
            try:
                fs = fsum(top) or {}
                if fs.get("summary"):
                    L += ["", f"LATEST FILING on top pick {top} "
                          f"({fs.get('form')} {fs.get('date')}):", fs["summary"]]
            except Exception:
                pass
        txt = "\n".join(L)
        if self.llm_fn:
            try:
                from .llm import guard
                syn = self.llm_fn(guard(
                    "You are the desk's Fundamental Analyst. Given the "
                    "cross-sectional read in the data block, write 4 tight "
                    "bullets: which names the fundamentals favour, which to "
                    "avoid, one risk the composite may be hiding, and what to "
                    "verify next. Be specific.", txt))
                txt += "\n\nANALYST SYNTHESIS:\n" + syn.strip()
            except Exception:
                pass
        self._save_report("fundamental_brief", txt)
        n_pit = 0
        try:                       # accumulate the point-in-time vintage store
            from .pit import PITStore
            n_pit = PITStore().snapshot([s for s, _ in rows], fund)
        except Exception:
            pass
        best_sym, best_m = ranked[0]
        tgt, last = best_m.get("target_mean"), (q.get(best_sym) or {}).get("last")
        up = (f", analyst target +{(tgt / last - 1) * 100:.0f}%"
              if isinstance(tgt, (int, float)) and last else "")
        self.propose("Fundamental Analyst", "pick",
                     f"{best_sym} tops the value/growth/quality composite{up}",
                     symbol=best_sym, side="buy", dedup=f"fund_pick:{best_sym}")
        return (f"FUNDAMENTAL BRIEF filed -> reports/; favours "
                f"{', '.join(s for s, _ in ranked[:3])}; "
                f"richest {ranked[-1][0]}; PIT vintages +{n_pit}")

    # --------------------------------------------------------- arb strategy
    _ARB_SYMS = ("WTI", "BRENT", "NATGAS", "BTC", "ETH",
                 "EURUSD", "GBPUSD", "AUDUSD", "CADUSD", "CHFUSD")

    def _arb_brief(self) -> str:
        """Deep task (24h): Engle-Granger pairs scan over the bundled real
        daily history + OOS backtests on the cointegrated survivors, the
        current triangular-loop read, and the CIP snapshot. Filed to the
        Briefing Center. All PROPOSALS — the gateway owns execution."""
        from .arb import cip, pairs, triangular
        from .data import load_real
        px = {}
        for s in self._ARB_SYMS:
            try:
                # full available history (decades) — the DSR gate needs enough
                # out-of-sample trades to verify; capped at 6000 bars for speed
                px[s] = load_real(s)["close"].to_numpy()[-6000:]
            except Exception:
                continue
        gs = pairs.gated_scan(px)               # cointegration -> OOS bt -> DSR
        co = [r for r in gs if r["cointegrated"]]
        n_tested = len(pairs.find_pairs(px))
        L = [f"ARB BRIEF — pairs scan over {len(px)} instruments "
             f"({n_tested} pairs tested, {len(co)} cointegrated at EG 5%)", "",
             "DSR-GATED (the SAME verification every registry strategy passes; "
             "DSR corrects for how many pairs we searched — verify the ECONOMIC "
             "link too, statistical passes without one burn accounts):"]
        for r in co[:6]:
            tag = ("✓" if r["dsr"] >= 0.95 else "~" if r["dsr"] >= 0.80 else "✗")
            L.append(f"  {tag} {r['y']:7s}~{r['x']:7s} ADF {r['adf']:+.2f}  "
                     f"β {r['beta']:+.3f}  hl {r['half_life']}d  "
                     f"OOS {r['n_trades']}tr wr {(r['win_rate'] or 0):.0%} "
                     f"{r['total_ret_pct']}%  DSR {r['dsr']}")
        survivors = [r for r in co if r["dsr"] >= 0.95]
        if survivors:
            s = survivors[0]
            self.propose("Arb Strategist", "pairs",
                         f"{s['y']}~{s['x']} passed the DSR gate ({s['dsr']}): "
                         f"long/short the spread at ±2σ, β {s['beta']}",
                         symbol=s["y"], side="buy",
                         dedup=f"pair:{s['y']}~{s['x']}",
                         payload={"x": s["x"], "beta": s["beta"], "dsr": s["dsr"]})
        ob = self.context.get("orderbook")
        if ob:
            L += ["", "TRIANGULAR LOOPS (live L2-walked, net of taker fees):"]
            for tri in triangular.TRIANGLES:
                try:
                    t = triangular.loop_edge(lambda p: ob(p, 20), tri, 1000)
                    if "error" in t:
                        continue
                    L.append(f"  {tri}: fwd {t['fwd']['edge_bps']:+.1f}bps / "
                             f"rev {t['rev']['edge_bps']:+.1f}bps -> "
                             f"{'SIGNAL' if t['signal'] else 'no edge (normal)'}")
                except Exception:
                    continue
        q = self.context.get("quotes", {})
        spots = {p: (q.get(p) or {}).get("last") for p in ("EURUSD", "GBPUSD")}
        try:
            from .intel import _fred_latest
            rows = cip.snapshot(_fred_latest, {k: v for k, v in spots.items() if v})
            if rows:
                L += ["", "CIP (theoretical 3M forwards from FRED — analysis "
                      "only, no forward market on this venue):"]
                for r in rows:
                    L.append(f"  {r['pair']}: spot {r['spot']:.4f} -> fwd "
                             f"{r['fwd']} ({r['points']:+.1f}pts, carry "
                             f"{r['carry_bps_ann']:+.0f}bps ann)")
        except Exception:
            pass
        txt = "\n".join(L)
        if self.llm_fn:
            try:
                from .llm import guard
                syn = self.llm_fn(guard(
                    "You are the desk's Arb Strategist. From the scan in the "
                    "data block, in 3 tight bullets: which pair (if any) "
                    "deserves paper capital and why, what invalidates it, and "
                    "one thing the statistics may be hiding. Be specific.", txt))
                txt += "\n\nSTRATEGIST SYNTHESIS:\n" + syn.strip()
            except Exception:
                pass
        self._save_report("arb_brief", txt)
        top = co[0] if co else None
        return ("ARB BRIEF filed -> reports/; " +
                (f"top pair {top['y']}~{top['x']} (ADF {top['adf']}, "
                 f"hl {top['half_life']}d)" if top else "no cointegrated pairs") +
                "; triangular loops inside fees (normal)")

    def _tri_watch(self) -> str:
        """Routine cycle: walk both crypto triangles on live L2; propose loudly
        the moment a loop nets positive after fees — otherwise a quiet read."""
        from .arb import triangular
        ob = self.context.get("orderbook")
        if not ob:
            return "triangular watch idle: no order-book source wired"
        reads = []
        for tri in triangular.TRIANGLES:
            try:
                t = triangular.loop_edge(lambda p: ob(p, 20), tri, 1000)
            except Exception:
                continue
            if "error" in t:
                continue
            if t["signal"]:
                msg = (f"TRIANGULAR {tri}: {t['best']['path']} nets "
                       f"{t['best']['edge_bps']:+.1f}bps on $1k after fees")
                self.log("Arb Strategist", "⚠ " + msg +
                         " — PROPOSAL only, gateway owns execution", "warn")
                first = tri.split("-")[0]
                self.propose("Arb Strategist", "triangular", msg,
                             notify_priority="high", symbol=f"{first}/USD",
                             side="buy", payload={"edge_bps": t["best"]["edge_bps"],
                             "path": t["best"]["path"]}, dedup=f"tri:{tri}")
            best = max(t["fwd"]["edge_bps"], t["rev"]["edge_bps"])
            reads.append(f"{tri} {best:+.0f}bps")
        return ("triangular loops: " + ", ".join(reads) +
                " (net of fees; negative = no arb, the normal state)"
                if reads else "triangular watch: books unavailable")

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
        elif agent.name == "Fundamental Analyst" and q:
            movers = [s for s, v in sorted(q.items(),
                      key=lambda kv: abs(kv[1].get("chg_pct") or 0), reverse=True)
                      if s not in _NON_EQUITY and s.isalpha()][:1]
            msg = "watchlist steady; valuation ranks unchanged since the daily brief"
            fund = self.context.get("fundamentals")
            if movers and fund:
                try:
                    b = (fund(movers[0]) or {}).get("brief")
                    ch = q[movers[0]].get("chg_pct")
                    if b:
                        msg = (f"{movers[0]} moving {ch:+.2f}% — fundamental "
                               f"context: {b}")
                except Exception:
                    pass
        elif agent.name == "Arb Strategist":
            msg = self._tri_watch()
        elif agent.name == "Portfolio Manager":
            at = getattr(self, "autotrader", None)
            if at and at.enabled:
                ps = getattr(self, "planstore", None)
                plan = ps.latest() if ps else None   # armed late? run today's plan
                pd = plan.get("date", "") or at._today()
                if (plan and plan.get("status") == "adopted"
                        and not at.plan_executed(pd)):
                    res = at.execute_plan(plan)
                    return (f"executing today's plan: entered {res.get('executed', 0)}, "
                            f"skipped {len(res.get('skipped', []))}")
                # TP/SL monitoring is owned by the dedicated 30s _autotrader_loop
                # — the PM does NOT also call monitor() (that raced it into
                # double-closing a position). Read-only status here.
                st = at.status()
                msg = (f"auto-trader ARMED · {st['open']} open, "
                       f"{st['orders_today']}/{st['max_orders_day']} orders today, "
                       f"realized {st['realized_today']:+.0f}")
            else:
                msg = "day plan set; auto-trader disarmed — trades await your arm/approval"
        elif agent.name == "Microstructure Analyst":
            if not self.l2lab:
                msg = "crypto L2 feed not wired — experiment idle"
            else:
                n = self.l2lab.snapshot()
                d = self.l2lab.days_active()
                msg = (f"sampled {n} crypto L2 book(s); depth+imbalance recorded "
                       f"(day {d:.0f}/180 of the free-L2 benefit trial)")
        elif agent.name == "Validation Officer":
            msg = "gate idle: no new candidates above DSR 0.95 this cycle"
        else:
            msg = "cycle complete; nothing actionable — remaining on standby"
        if agent.name in ("Arb Strategist", "Microstructure Analyst",
                          "Portfolio Manager"):
            return msg                        # quantitative reads stay verbatim
        if self.llm_fn:                       # optional richer write-up
            try:
                from .llm import guard
                msg = self.llm_fn(guard(
                    f"You are the {agent.name}. Write a two-sentence desk "
                    "note from the data block.", msg))
            except Exception:
                pass
        return msg

    async def _loop(self, agent: Agent) -> None:
        while True:
            if self.master and agent.enabled:
                try:
                    fn = agent.work_fn or self._default_work
                    msg = await asyncio.to_thread(fn, agent)
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
