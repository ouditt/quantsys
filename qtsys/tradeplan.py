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
# learning-feedback bounds — clamped here (consumer) as well as in learning.py
# (producer), so a malformed input can never widen sizing beyond these lines.
MULT_LO, MULT_HI = 0.25, 1.5
ATR_STOP_LO, ATR_STOP_HI = 1.0, 2.5
R_MULT_LO, R_MULT_HI = 1.5, 3.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------------------------------------------------------------- idea sizing
def _size_idea(idea: dict, equity: float, risk_pct: float,
               max_notional: float, sym_cap: float | None = None,
               atr_stop: float = ATR_STOP, r_multiple: float = R_MULTIPLE) -> dict:
    entry, atr = idea.get("entry"), idea.get("atr")
    long = idea["side"] in ("LONG", "buy")
    if not entry or not atr or atr <= 0 or not equity:
        idea.update(stop=None, target=None, qty=0.0, notional=0.0)
        return idea
    stop = entry - atr_stop * atr if long else entry + atr_stop * atr
    dist = abs(entry - stop)
    target = entry + r_multiple * dist if long else entry - r_multiple * dist
    qty = (equity * risk_pct) / dist if dist else 0.0
    # clip (never skip) to BOTH caps: per-order notional and the auto-trader's
    # per-symbol exposure cap — otherwise low-vol names structurally size
    # above the cap and every execution gets refused
    cap = min(max_notional, sym_cap) if sym_cap else max_notional
    if qty * entry > cap and entry:
        qty = cap / entry
    if not long and "/" not in idea["symbol"]:
        qty = float(int(qty))          # venue forbids fractional SHORT selling
    notional = qty * entry
    idea.update(stop=round(stop, 4), target=round(target, 4),
                qty=round(qty, 6), notional=round(notional, 2),
                risk_amt=round(qty * dist, 2), rr=r_multiple)
    return idea


# tolerance band around a target before we bother adjusting a held position
_REBALANCE_TOL = 0.20            # ±20% of target notional = "already at target"


def _reconcile_holding(idea: dict, held: dict | None, rank: int,
                       holds: list) -> None:
    """Turn a freshly-sized target into an idempotent action given what we
    already hold. Mutates `idea` in place; a non-executable outcome (hold /
    trim / flip) is appended to `holds` and the idea is emptied (qty->0 so it
    drops out of the executable plan).

    Rules (the operator's spec):
      * not held            -> action "open" (buy the full target once).
      * held, at target     -> HOLD, no order (this kills the duplicate-buy bug).
      * held, want MORE, and this is the single highest-priority idea (rank 0,
        i.e. best expected return the strategist surfaced) and it's a fresh
        signal -> action "increase", sized to the DELTA only.
      * held, want LESS      -> DECREASE surfaced as a trim recommendation
        (risk reduction stays with the monitor/human, not auto-stacked).
      * held opposite side   -> FLIP surfaced for the human (engine never flips).
    """
    if not held or not held.get("qty"):
        return                                   # flat -> normal open
    sym = idea["symbol"]
    tgt_qty = idea.get("qty") or 0.0
    tgt_notional = idea.get("notional") or 0.0
    held_qty = held["qty"]
    held_side = "LONG" if held_qty > 0 else "SHORT"
    held_notional = held.get("notional") or abs(held_qty * (idea.get("entry") or 0))

    def _park(action, msg):
        holds.append({"symbol": sym, "action": action, "held_qty": held_qty,
                      "held_side": held_side, "note": msg,
                      "rank": rank, "rationale": idea.get("rationale", "")})
        idea.update(qty=0.0, notional=0.0, action=action)   # non-executable

    if held_side != idea["side"]:
        _park("flip", f"signal flipped to {idea['side']} but we hold "
              f"{abs(held_qty):g} {held_side} — engine won't flip; human call")
        return
    band = max(_REBALANCE_TOL * max(tgt_notional, 1), 0.01 * (idea.get("entry") or 0))
    if abs(tgt_notional - held_notional) <= band:
        _park("hold", f"already at target ({abs(held_qty):g} sh ~"
              f"{held_notional:,.0f}); no new order")
        return
    if tgt_notional > held_notional:                       # want MORE
        if rank != 0:
            _park("hold", f"holding {abs(held_qty):g} sh; a larger target exists "
                  "but this isn't the top-priority idea — not adding")
            return
        delta = max(tgt_qty - abs(held_qty), 0.0)
        if delta <= 0:
            _park("hold", "at target"); return
        idea.update(qty=round(delta, 6),
                    notional=round(delta * (idea.get("entry") or 0), 2),
                    action="increase", held_qty=held_qty, target_qty=tgt_qty,
                    exposure_change=f"INCREASE +{delta:g} sh "
                    f"({abs(held_qty):g}->{tgt_qty:g}) — fresh top-priority signal")
        return
    # want LESS -> trim recommendation (not auto-executed)
    _park("decrease", f"target below current ({abs(held_qty):g} sh); trim "
          f"~{abs(held_qty) - tgt_qty:g} sh to derisk — via monitor/manual")


# ------------------------------------------------------------------- drafting
def draft(data: dict) -> dict:
    """Assemble candidate ideas from the morning scans. `data` injects the live
    numbers so this stays testable off-server:
      equity, posture, setups[list], quote(sym)->px, atr(sym)->atr,
      arb_survivors[list], fundamental_picks[list]."""
    equity = data.get("equity") or 0.0
    posture = data.get("posture", "BALANCED")
    risk_pct = RISK_PCT.get(posture, 0.015)
    # SMALL-ACCOUNT GROWTH MODE: a percent of tiny equity is dust ($250 x 1.5%
    # = $3.75 risk -> unmovable positions). The server passes a $ risk floor
    # for small books; it lifts risk_pct so each trade risks at least that,
    # hard-capped at 8% so "growth mode" never becomes "blow-up mode".
    floor_amt = float(data.get("risk_floor_amt") or 0)
    if floor_amt and equity:
        risk_pct = min(max(risk_pct, floor_amt / equity), 0.08)
    maxN = data.get("max_order_notional", 25_000.0)
    sym_cap = data.get("max_symbol_notional")        # auto-trader per-symbol cap
    quote, atr = data.get("quote", lambda s: None), data.get("atr", lambda s: None)
    ideas = []
    seen = set()

    verified_set = set(data.get("verified_strategies", []))
    strategy_dsr = data.get("strategy_dsr", {})
    thr = float(data.get("dsr_threshold", 0.95))
    # current book: {engine_symbol: {"qty": signed, "notional": abs$, "side"}}.
    # The planner sizes to a TARGET given what we already hold, so a name is
    # bought ONCE — a held name only re-trades as an explicit increase/decrease.
    holdings = data.get("holdings", {}) or {}
    holds = []                     # non-executable status (hold / trim / flip)
    # LEARNING FEEDBACK (deterministic, from live outcomes): per-strategy size
    # multiplier and tuned exit params, both re-clamped here so a bad input can
    # never breach the bright lines. Empty inputs -> mult 1.0 and the module
    # defaults -> sizing byte-identical to the pre-learning engine.
    strat_mult = data.get("strategy_multiplier", {}) or {}
    exit_params = data.get("exit_params", {}) or {}

    def add(sym, side, strategy, source, rationale, tier="", dsr=None,
            verified=False, rank=99):
        if sym in seen:
            return
        if side == "SHORT" and "/" in sym:
            return          # the venue cannot short crypto — don't plan what
                            # can only die as an order rejection
        px = quote(sym)
        if not px:
            return
        seen.add(sym)
        rp = risk_pct * _clamp(strat_mult.get(strategy, 1.0), MULT_LO, MULT_HI)
        ep = exit_params.get(strategy, {}) or {}
        astop = (_clamp(ep["atr_stop"], ATR_STOP_LO, ATR_STOP_HI)
                 if ep.get("atr_stop") else ATR_STOP)
        rmult = (_clamp(ep["r_multiple"], R_MULT_LO, R_MULT_HI)
                 if ep.get("r_multiple") else R_MULTIPLE)
        idea = _size_idea(
            {"symbol": sym, "side": side, "strategy": strategy, "source": source,
             "rationale": rationale, "tier": tier, "dsr": dsr,
             "verified": bool(verified), "rank": rank, "action": "open",
             # asset class is explicit so the committee transcript and executor
             # show crypto/equity ideas passing the SAME gate; options are added
             # post-deliberation as a defined-risk expression of a passed idea.
             "asset_class": "Crypto" if "/" in sym else "Equity",
             "entry": round(px, 4), "atr": atr(sym)},
            equity, rp, maxN, sym_cap, astop, rmult)
        _reconcile_holding(idea, holdings.get(sym), rank, holds)
        if idea is not None:
            ideas.append(idea)

    setups = data.get("setups", [])[:16]              # ranked scan setups
    for rank, s in enumerate(setups):
        side = "LONG" if s.get("side") in ("LONG", "buy") else "SHORT"
        exp = s.get("hist_exp")
        d = s.get("dsr", strategy_dsr.get(s.get("strategy")))
        via = f" via proxy of {s['proxy_of']}" if s.get("proxy_of") else ""
        add(s["asset"], side, s.get("strategy", "?"), "scan",
            f"fresh {s.get('family','')} signal{via}"
            + (f", hist exp {exp:+.2%}/trade" if isinstance(exp, (int, float)) else ""),
            tier=s.get("tier", ""), dsr=d,
            verified=(d is not None and d >= thr)
            or s.get("strategy") in verified_set, rank=rank)
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
            "holds": holds, "status": "draft", "ts": time.time(),
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
    demoted = data.get("demoted", {}) or {}
    for i, idea in enumerate(plan["ideas"]):
        if idea.get("kind") == "option_structure":
            continue                              # defined-risk structure, not half-sized
        # LEARNING DRIFT: a strategy the learning pass demoted (live edge fell
        # below its certified backtest) is treated EXACTLY like an unverified
        # idea — de-verified and its DSR nulled so the existing auto-trader gate
        # routes it to the INBOX, then half-sized below.
        if idea.get("strategy") in demoted:
            idea["verified"] = False
            idea["dsr"] = None
            idea["demoted"] = True
            notes.append(f"{idea['symbol']}/{idea['strategy']}: DEMOTED by the "
                         f"learning pass ({demoted[idea['strategy']]}) — treated "
                         "as unverified")
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
        if idea.get("kind") == "option_structure":
            # a straddle WANTS a catalyst and a condor already priced the regime
            # — the vol skill owns that call, not the equity earnings screen
            continue
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
        if idea.get("kind") == "option_structure":
            continue                              # options vetted via chain gate_ok/OI
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
        self.db.execute("PRAGMA busy_timeout=5000")
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

    # ---- portfolio-aware idempotency (the duplicate-buy fix) ----
    # AAPL is the rank-0 setup. Hold it AT target -> HOLD, no order.
    aapl_tgt = next(i for i in draft(data)["ideas"] if i["symbol"] == "AAPL")
    at_target = dict(data, holdings={"AAPL": {
        "qty": aapl_tgt["qty"], "notional": aapl_tgt["notional"],
        "side": "LONG", "entry": aapl_tgt["entry"]}})
    p2 = draft(at_target)
    assert not [i for i in p2["ideas"] if i["symbol"] == "AAPL"], "held-at-target buys nothing"
    assert any(h["symbol"] == "AAPL" and h["action"] == "hold" for h in p2["holds"])
    # Hold only HALF the target of the rank-0 name -> INCREASE by the delta.
    half = dict(data, holdings={"AAPL": {
        "qty": aapl_tgt["qty"] / 2, "notional": aapl_tgt["notional"] / 2,
        "side": "LONG", "entry": aapl_tgt["entry"]}})
    p3 = draft(half)
    inc = next(i for i in p3["ideas"] if i["symbol"] == "AAPL")
    assert inc["action"] == "increase", "under-target top-priority -> increase"
    assert abs(inc["qty"] - aapl_tgt["qty"] / 2) < 1e-6, "increase sized to the DELTA only"
    # Same under-target but on a NON-top-priority name (XOM, rank 2) -> HOLD.
    xom_tgt = next(i for i in draft(data)["ideas"] if i["symbol"] == "XOM")
    half_xom = dict(data, holdings={"XOM": {
        "qty": -abs(xom_tgt["qty"]) / 2, "notional": xom_tgt["notional"] / 2,
        "side": "SHORT", "entry": xom_tgt["entry"]}})
    p4 = draft(half_xom)
    assert not [i for i in p4["ideas"] if i["symbol"] == "XOM"], "non-top-priority never adds"
    assert any(h["symbol"] == "XOM" and h["action"] == "hold" for h in p4["holds"])
    # Opposite side held -> FLIP, non-executable.
    flip = dict(data, holdings={"AAPL": {
        "qty": -10, "notional": 2000, "side": "SHORT", "entry": 200.0}})
    p5 = draft(flip)
    assert not [i for i in p5["ideas"] if i["symbol"] == "AAPL"], "won't auto-flip"
    assert any(h["symbol"] == "AAPL" and h["action"] == "flip" for h in p5["holds"])

    # ---- small-account growth mode + crypto rules ----
    # crypto SHORT is never planned (the venue can't short crypto)
    cdata = dict(data, setups=data["setups"] + [
        {"asset": "BTC/USD", "side": "SHORT", "strategy": "meanrev_rsi2",
         "family": "MeanRev", "hist_exp": 0.01}])
    pc = draft(cdata)
    assert not [i for i in pc["ideas"] if i["symbol"] == "BTC/USD"], "no crypto shorts"
    # long crypto on a tiny book: risk floor lifts sizing off the dust level
    tiny = dict(data, equity=250.0, risk_floor_amt=10.0, max_symbol_notional=75.0,
                setups=[{"asset": "BTC/USD", "side": "LONG",
                         "strategy": "roll_high_252", "family": "Momentum",
                         "hist_exp": 0.03}], arb_survivors=[], fundamental_picks=[])
    pt = draft(tiny)
    btc = next(i for i in pt["ideas"] if i["symbol"] == "BTC/USD")
    assert pt["risk_pct"] == 0.04, pt["risk_pct"]          # $10/$250 floor
    assert 0 < btc["qty"] < 1, "fractional crypto qty"
    assert btc["notional"] <= 75.0 + 1, "clipped to the small-account symbol cap"
    assert btc["risk_amt"] >= 1.0, btc["risk_amt"]
    # the floor never exceeds the 8% blow-up guard
    pt2 = draft(dict(tiny, equity=100.0))                  # $10/$100 = 10% -> 8%
    assert pt2["risk_pct"] == 0.08, pt2["risk_pct"]

    # ---- learning feedback: per-strategy multiplier, exit-param override,
    #      and demotion routing (empty inputs -> byte-identical sizing) ----
    ldata = dict(data, max_order_notional=1_000_000)     # unbind the notional cap
    aapl_base = next(i for i in draft(ldata)["ideas"] if i["symbol"] == "AAPL")
    aapl_half = next(i for i in draft(dict(ldata, strategy_multiplier={
        "roll_high_252": 0.5}))["ideas"] if i["symbol"] == "AAPL")
    assert abs(aapl_half["risk_amt"] - aapl_base["risk_amt"] * 0.5) < 0.5, \
        (aapl_half["risk_amt"], aapl_base["risk_amt"])
    # out-of-bounds multiplier is CLAMPED at the consumer (never > 1.5x)
    aapl_huge = next(i for i in draft(dict(ldata, strategy_multiplier={
        "roll_high_252": 9.0}))["ideas"] if i["symbol"] == "AAPL")
    assert aapl_huge["risk_amt"] <= aapl_base["risk_amt"] * 1.5 + 0.5, aapl_huge["risk_amt"]
    # exit-param override reshapes stop/target geometry (clamped to bounds)
    aapl_ep = next(i for i in draft(dict(ldata, exit_params={
        "roll_high_252": {"atr_stop": 2.5, "r_multiple": 3.0}}))["ideas"]
        if i["symbol"] == "AAPL")
    assert aapl_ep["rr"] == 3.0 and abs((aapl_ep["entry"] - aapl_ep["stop"]) - 2.5 * 4.0) < 0.01
    # demotion: a verified strategy the learning pass demoted is de-verified,
    # DSR-nulled, half-sized and INBOX-routed — exactly like an unverified idea.
    # Use the default-cap data so the leverage trim doesn't drop AAPL first.
    dd = dict(data, demoted={"roll_high_252": "live edge drifted below backtest"})
    ad = next(i for i in deliberate(draft(dd), dd)["ideas"] if i["symbol"] == "AAPL")
    assert ad.get("demoted") and ad.get("half_size") and ad["verified"] is False \
        and ad["dsr"] is None, ad

    print(f"tradeplan self-test ✓  drafted {len(plan['ideas'])} ideas w/ ATR "
          f"stops+2R targets, 4-agent deliberation, half-size on unverified, "
          f"persisted; portfolio-aware hold/increase/flip idempotency; "
          f"small-account risk floor + no-crypto-short rule; learning "
          f"multiplier (clamped) + exit-param override + demotion routing")


if __name__ == "__main__":
    _selftest()
