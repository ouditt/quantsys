"""tradeplan.py — the daily trading plan: draft, one-round deliberation, adopt.

The Portfolio Manager drafts a day plan from the morning scans (universe
setups + DSR-verified stat-arb + fundamental picks). Each specialist agent
then contributes ONE scoped critique — Risk (exposure/correlation/leverage),
Validation (was it verified?), Fundamental (earnings/valuation landmines),
Microstructure (liquidity) — and the PM synthesises a single revision that
drops or trims the flagged ideas. One bounded round, then it's adopted.

Every idea carries entry / stop / target / size derived from ATR and the
active posture's risk-per-trade, so the auto-trader (execution owner) has
concrete, gateway-checkable orders — and a human can read the whole plan and
the debate transcript on the PLAN page.

All sizing is PROPOSAL math; the ExecutionGateway still checks every order.
SQLite (tradeplan.db, gitignored).
"""
from __future__ import annotations

import datetime
import json
import os
import sqlite3
import time

RISK_PCT = {"SURVIVAL": 0.0075, "BALANCED": 0.015, "AGGRESSIVE": 0.025}
ATR_STOP = 1.5          # stop = entry ∓ ATR_STOP × ATR
R_MULTIPLE = 2.0        # target = R_MULTIPLE × the stop distance
MAX_IDEAS = 8


# ---------------------------------------------------------------- idea sizing
def _size_idea(idea: dict, equity: float, risk_pct: float,
               max_notional: float) -> dict:
    entry, atr = idea.get("entry"), idea.get("atr")
    long = idea["side"] in ("LONG", "buy")
    if not entry or not atr or atr <= 0 or not equity:
        idea.update(stop=None, target=None, qty=0.0, notional=0.0)
        return idea
    stop = entry - ATR_STOP * atr if long else entry + ATR_STOP * atr
    dist = abs(entry - stop)
    target = entry + R_MULTIPLE * dist if long else entry - R_MULTIPLE * dist
    qty = (equity * risk_pct) / dist if dist else 0.0
    notional = qty * entry
    if notional > max_notional and entry:            # respect the per-order cap
        qty = max_notional / entry
        notional = max_notional
    idea.update(stop=round(stop, 4), target=round(target, 4),
                qty=round(qty, 6), notional=round(notional, 2),
                risk_amt=round(qty * dist, 2), rr=R_MULTIPLE)
    return idea


# ------------------------------------------------------------------- drafting
def draft(data: dict) -> dict:
    """Assemble candidate ideas from the morning scans. `data` injects the live
    numbers so this stays testable off-server:
      equity, posture, setups[list], quote(sym)->px, atr(sym)->atr,
      arb_survivors[list], fundamental_picks[list]."""
    equity = data.get("equity") or 0.0
    posture = data.get("posture", "BALANCED")
    risk_pct = RISK_PCT.get(posture, 0.015)
    maxN = data.get("max_order_notional", 25_000.0)
    quote, atr = data.get("quote", lambda s: None), data.get("atr", lambda s: None)
    ideas = []
    seen = set()

    verified_set = set(data.get("verified_strategies", []))
    strategy_dsr = data.get("strategy_dsr", {})
    thr = float(data.get("dsr_threshold", 0.95))

    def add(sym, side, strategy, source, rationale, tier="", dsr=None,
            verified=False):
        if sym in seen:
            return
        px = quote(sym)
        if not px:
            return
        seen.add(sym)
        ideas.append(_size_idea(
            {"symbol": sym, "side": side, "strategy": strategy, "source": source,
             "rationale": rationale, "tier": tier, "dsr": dsr,
             "verified": bool(verified),
             "entry": round(px, 4), "atr": atr(sym)},
            equity, risk_pct, maxN))

    for s in data.get("setups", [])[:16]:            # ranked scan setups
        side = "LONG" if s.get("side") in ("LONG", "buy") else "SHORT"
        exp = s.get("hist_exp")
        d = s.get("dsr", strategy_dsr.get(s.get("strategy")))
        via = f" via proxy of {s['proxy_of']}" if s.get("proxy_of") else ""
        add(s["asset"], side, s.get("strategy", "?"), "scan",
            f"fresh {s.get('family','')} signal{via}"
            + (f", hist exp {exp:+.2%}/trade" if isinstance(exp, (int, float)) else ""),
            tier=s.get("tier", ""), dsr=d,
            verified=(d is not None and d >= thr)
            or s.get("strategy") in verified_set)
    for a in data.get("arb_survivors", [])[:3]:       # DSR-passed stat-arb
        add(a["y"], "LONG", f"pairs {a['y']}~{a['x']}", "arb",
            f"cointegrated (DSR {a.get('dsr')}), long the spread",
            dsr=a.get("dsr"), verified=(a.get("dsr") or 0) >= thr)
    for f in data.get("fundamental_picks", [])[:2]:   # value/quality picks
        add(f["symbol"], "LONG", "fundamental", "fundamental",
            f.get("rationale", "top of the value/growth composite"),
            verified=False)                           # no backtest -> human call

    sized = [i for i in ideas if i.get("stop") and i.get("qty")]   # drop unsized
    return {"date": str(datetime.date.today()), "posture": posture,
            "equity": equity, "risk_pct": risk_pct,
            "ideas": sized[:MAX_IDEAS], "critiques": [], "notes": "",
            "status": "draft", "ts": time.time(),
            "unsized_skipped": len(ideas) - len(sized)}


# --------------------------------------------------------- specialist reviews
def _risk_review(plan, data):
    """Exposure, correlation, leverage."""
    ideas = plan["ideas"]
    gross = sum(i["notional"] for i in ideas)
    eq = plan["equity"] or 1
    lev = gross / eq
    rejects, notes = set(), []
    max_lev = data.get("max_gross_leverage", 2.0)
    if lev > max_lev:                                 # trim the smallest-edge tail
        notes.append(f"planned gross {gross:,.0f} = {lev:.2f}x > {max_lev:.1f}x cap "
                     f"— trimming lowest-conviction ideas")
        order = sorted(range(len(ideas)),
                       key=lambda i: (ideas[i].get("dsr") or 0, ideas[i]["notional"]))
        while lev > max_lev and order:
            j = order.pop(0)
            rejects.add(j)
            gross -= ideas[j]["notional"]
            lev = gross / eq
    for grp in data.get("clusters", []):              # correlation clustering
        inp = [i for i, x in enumerate(ideas) if x["symbol"] in grp and i not in rejects]
        if len(inp) > data.get("max_cluster_positions", 2):
            notes.append(f"{'+'.join(sorted(grp))}: {len(inp)} correlated positions "
                         f"> {data.get('max_cluster_positions', 2)} — capping the cluster")
            for j in inp[data.get("max_cluster_positions", 2):]:
                rejects.add(j)
    if not notes:
        notes.append(f"exposure OK: {lev:.2f}x gross, no cluster breach")
    return {"agent": "Risk Officer", "notes": notes, "rejects": rejects}


def _validation_review(plan, data):
    rejects, notes = set(), []
    for i, idea in enumerate(plan["ideas"]):
        if not idea.get("verified"):
            notes.append(f"{idea['symbol']}/{idea['strategy']}: not DSR-verified "
                         "— HALF size, and the auto-trader will NOT touch it "
                         "(routes to the INBOX for human approval)")
            idea["qty"] = round(idea["qty"] * 0.5, 6)
            idea["notional"] = round(idea["notional"] * 0.5, 2)
            idea["half_size"] = True
    if not notes:
        notes.append("all ideas trace to a DSR-verified strategy — machine-tradable")
    return {"agent": "Validation Officer", "notes": notes, "rejects": rejects}


def _fundamental_review(plan, data):
    fund = data.get("fundamentals")
    notes = []
    if not fund:
        return {"agent": "Fundamental Analyst", "notes": ["no fundamental feed"],
                "rejects": set()}
    for idea in plan["ideas"]:
        try:
            m = (fund(idea["symbol"]) or {}).get("metrics", {}) or {}
        except Exception:
            continue
        ed = m.get("next_earnings_days")
        if isinstance(ed, (int, float)) and 0 <= ed <= 3:
            notes.append(f"{idea['symbol']}: earnings in ~{ed:.0f}d — event risk, "
                         "size down or wait")
            idea["earnings_flag"] = True
        pe = m.get("forward_pe")
        if isinstance(pe, (int, float)) and pe > 60 and idea["side"] == "LONG":
            notes.append(f"{idea['symbol']}: fwd P/E {pe:.0f} — priced for perfection")
    if not notes:
        notes.append("no earnings-window or extreme-valuation flags on the book")
    return {"agent": "Fundamental Analyst", "notes": notes, "rejects": set()}


def _microstructure_review(plan, data):
    ob = data.get("orderbook")
    notes = []
    for idea in plan["ideas"]:
        if "/" in idea["symbol"] and ob:              # crypto: real L2 liquidity
            try:
                from .orderbook import metrics
                m = metrics(ob(idea["symbol"]), idea["notional"] or 5000)
                if m.get("depth_exhausted"):
                    notes.append(f"{idea['symbol']}: book too thin for "
                                 f"{idea['notional']:,.0f} — will slip; slice it")
                    idea["liquidity_flag"] = True
            except Exception:
                pass
    if not notes:
        notes.append("liquidity adequate for the planned sizes (or equity — assumed liquid)")
    return {"agent": "Microstructure Analyst", "notes": notes, "rejects": set()}


REVIEWS = (_risk_review, _validation_review, _fundamental_review,
           _microstructure_review)


def deliberate(plan: dict, data: dict, llm_fn=None) -> dict:
    """One bounded round: gather every specialist's scoped critique, drop the
    rejected ideas, then have the PM synthesise the final notes."""
    all_rejects = set()
    for review in REVIEWS:
        try:
            r = review(plan, data)
        except Exception as e:
            r = {"agent": review.__name__, "notes": [f"review error: {e}"],
                 "rejects": set()}
        plan["critiques"].append({"agent": r["agent"], "notes": r["notes"]})
        all_rejects |= r["rejects"]
    kept = [i for j, i in enumerate(plan["ideas"]) if j not in all_rejects]
    plan["ideas"] = kept
    plan["dropped"] = len(all_rejects)
    # PM synthesis
    gross = sum(i["notional"] for i in kept)
    base = (f"Adopted {len(kept)} ideas ({plan['dropped']} dropped in review); "
            f"planned gross {gross:,.0f} ({gross / max(plan['equity'], 1):.2f}x), "
            f"posture {plan['posture']}. Each carries an ATR stop and a "
            f"{R_MULTIPLE:g}R target.")
    if llm_fn:
        try:
            from .llm import guard
            transcript = json.dumps({"ideas": [
                {k: i.get(k) for k in ("symbol", "side", "strategy", "entry",
                 "stop", "target", "notional", "rationale")} for i in kept],
                "critiques": plan["critiques"]}, default=str)
            base = llm_fn(guard(
                "You are the Portfolio Manager. From the day's ideas and the "
                "specialists' critiques in the data block, write a tight 3-4 "
                "sentence trading-plan summary a human can act on: the theme, "
                "the top 2 convictions, the main risk the desk flagged, and "
                "the one thing to watch intraday. No preamble.", transcript)).strip()
        except Exception:
            pass
    plan["notes"] = base
    plan["status"] = "adopted"
    return plan


# --------------------------------------------------------------- persistence
class PlanStore:
    def __init__(self, db_path=None):
        self.db = sqlite3.connect(
            db_path or os.path.join(os.path.dirname(__file__), "tradeplan.db"),
            check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS plan("
                        "date TEXT PRIMARY KEY, ts REAL, status TEXT, doc TEXT)")
        self.db.commit()

    def save(self, plan: dict):
        self.db.execute("INSERT OR REPLACE INTO plan VALUES (?,?,?,?)",
                        (plan["date"], plan.get("ts", time.time()),
                         plan.get("status", "draft"), json.dumps(plan, default=str)))
        self.db.commit()

    def latest(self) -> dict | None:
        r = self.db.execute("SELECT doc FROM plan ORDER BY date DESC "
                            "LIMIT 1").fetchone()
        return json.loads(r[0]) if r else None

    def get(self, date: str) -> dict | None:
        r = self.db.execute("SELECT doc FROM plan WHERE date=?", (date,)).fetchone()
        return json.loads(r[0]) if r else None


# ------------------------------------------------------------------ self-test
def _selftest():
    px = {"AAPL": 200.0, "NVDA": 120.0, "MSFT": 400.0, "XOM": 110.0,
          "BTC/USD": 60000.0, "WTI": 75.0, "BRENT": 78.0}
    atr = {"AAPL": 4.0, "NVDA": 3.0, "MSFT": 6.0, "XOM": 2.0,
           "BTC/USD": 1500.0, "WTI": 1.5, "BRENT": 1.6}
    data = {
        "equity": 100_000.0, "posture": "BALANCED", "max_order_notional": 25_000,
        "max_gross_leverage": 2.0, "quote": px.get, "atr": atr.get,
        "setups": [{"asset": "AAPL", "side": "LONG", "strategy": "roll_high_252",
                    "family": "Momentum", "hist_exp": 0.03},
                   {"asset": "NVDA", "side": "LONG", "strategy": "unverified_x",
                    "family": "MA cross", "hist_exp": 0.02},
                   {"asset": "XOM", "side": "SHORT", "strategy": "meanrev_rsi2",
                    "family": "MeanRev", "hist_exp": 0.01}],
        "arb_survivors": [{"y": "WTI", "x": "BRENT", "dsr": 0.97}],
        "fundamental_picks": [{"symbol": "MSFT", "rationale": "cheap quality"}],
        "verified_strategies": ["roll_high_252", "meanrev_rsi2"],
        "clusters": [{"WTI", "BRENT"}],
    }
    plan = draft(data)
    assert plan["ideas"], "ideas drafted"
    a = next(i for i in plan["ideas"] if i["symbol"] == "AAPL")
    assert a["stop"] < a["entry"] < a["target"], "long: stop<entry<target"
    assert a["risk_amt"] <= 100_000 * 0.015 + 1, "risk never exceeds 1.5%"
    assert a["notional"] <= 25_000 + 1, "per-order notional cap respected"
    assert abs(a["qty"] * a["entry"] - a["notional"]) < 1, "notional = qty×entry"
    x = next(i for i in plan["ideas"] if i["symbol"] == "XOM")
    assert x["stop"] > x["entry"] > x["target"], "short: target<entry<stop"
    plan = deliberate(plan, data)
    assert plan["status"] == "adopted"
    assert len(plan["critiques"]) == 4, "all four specialists reviewed"
    nv = [i for i in plan["ideas"] if i["symbol"] == "NVDA"]
    assert nv and nv[0].get("half_size"), "unverified strategy -> half size, kept"
    # WTI+BRENT are a correlation cluster of 2 — within cap, both allowed here
    store = PlanStore(":memory:")
    store.save(plan)
    assert store.latest()["date"] == plan["date"]
    print(f"tradeplan self-test ✓  drafted {len(plan['ideas'])} ideas w/ ATR "
          f"stops+2R targets, 4-agent deliberation, half-size on unverified, "
          f"persisted")


if __name__ == "__main__":
    _selftest()
