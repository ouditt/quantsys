"""server.py — run:  pip install fastapi uvicorn && ./start.sh  (or
python -m uvicorn qtsys.server:app --port 8001)

Serves terminal.html in LIVE mode against a REAL venue. The simulator has been
removed: QTSYS_BROKER selects alpaca (paper or live) or ibkr; quotes and fills
come from the venue, never from a replay tape. The posture bar's distribution
stats come from the real out-of-sample backtest (sizing.posture_table), and
/api/tracking compares that backtest against the account's actual realised
trades so drift is visible (skill 10's "live-vs-backtest within 1 SE" gate).
"""
from __future__ import annotations

import asyncio
import math
import os
import time

from .agents import AgentDaemon
from .brokers import ExecutionGateway, Order, PaperBroker, RiskLimits
from .data import load_real

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
except ImportError as e:                                   # pragma: no cover
    raise SystemExit("pip install fastapi uvicorn") from e

HERE = os.path.dirname(__file__)

# 100% REAL bundled data — used for history charts and analyse-only symbols.
# Tradable prices come LIVE from the venue; nothing here is simulated.
UNIVERSE = [
    ("WTI",    "WTI Crude Oil (spot, daily 1986->)",      "Commodity", True),
    ("BRENT",  "Brent Crude Oil (spot, daily 1987->)",    "Commodity", True),
    ("NATGAS", "Henry Hub Natural Gas (daily 1997->)",    "Commodity", True),
    ("BTC",    "Bitcoin (Coin Metrics, daily 2010->)",    "Crypto", True),
    ("ETH",    "Ethereum (Coin Metrics, daily 2015->)",   "Crypto", True),
    ("EURUSD", "Euro / US Dollar (daily 1999->)",         "FX", True),
    ("GBPUSD", "British Pound / USD (daily 1971->)",      "FX", True),
    ("AUDUSD", "Australian Dollar / USD (daily 1971->)",  "FX", True),
    ("JPYUSD", "Japanese Yen / USD (daily 1971->)",       "FX", True),
    ("CHFUSD", "Swiss Franc / USD (daily 1971->)",        "FX", True),
    ("CADUSD", "Canadian Dollar / USD (daily 1971->)",    "FX", True),
    ("VIX",    "CBOE Volatility Index (daily 1990->)",    "Index — analyse-only", False),
    ("GOLD",   "Gold (monthly 1833->)",                   "Monthly — page-only", False),
    ("SPX",    "S&P 500 (monthly 1871->, Shiller)",       "Monthly — page-only", False),
]
TRADABLE = {s for s, _, _, ok in UNIVERSE if ok}
MONTHLY = {"GOLD", "SPX"}          # page-only: pinned to their latest real bar
CLS = {s: c for s, _, c, _ in UNIVERSE}

# engine symbol -> venue symbol, per venue. Anything not mapped is NOT sent to
# the venue (engine codes like "WTI" collide with unrelated NYSE tickers).
VENUE_SYMBOLS = {
    "alpaca": {"BTC": "BTC/USD", "ETH": "ETH/USD"},
    "ibkr": None,   # None = venue serves the engine symbols directly
}
POSTURE_SCALE = {"SURVIVAL": 0.5, "BALANCED": 1.0, "AGGRESSIVE": 1.5}

app = FastAPI(title="qtsys terminal API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

state: dict = {}


def _equity(broker) -> float:
    # PaperBroker has equity(); real adapters expose it via get_account()
    if hasattr(broker, "equity"):
        return broker.equity()
    return float(broker.get_account().get("equity", 0.0))


def _cls_of(sym: str) -> str:
    """Asset class for an engine OR venue symbol (BTC and BTC/USD -> Crypto)."""
    if sym in CLS:
        return CLS[sym]
    base = sym.split("/")[0]
    return CLS.get(base, "")


def _fills(broker) -> list[dict]:
    """Normalised fill stream across brokers, for realised-trade tracking."""
    if hasattr(broker, "fills"):
        return list(broker.fills)
    if hasattr(broker, "recent_fills"):
        return broker.recent_fills()
    return []


@app.on_event("startup")
async def boot() -> None:
    venue = os.environ.get("QTSYS_BROKER", "alpaca").lower()
    if venue == "alpaca":
        from .brokers import make_broker
        paper = os.environ.get("ALPACA_PAPER", "1") != "0"
        # live trading uses the separate ALPACA_LIVE_* key pair so the paper
        # keys can never silently authenticate against the live account
        key = os.environ["ALPACA_API_KEY" if paper else "ALPACA_LIVE_API_KEY"]
        sec = os.environ["ALPACA_SECRET_KEY" if paper else "ALPACA_LIVE_SECRET_KEY"]
        broker = make_broker("alpaca", api_key=key, secret=sec, paper=paper)
    elif venue == "ibkr":
        from .brokers import make_broker
        # paper: TWS 7497 / Gateway 4002; live: TWS 7496 / Gateway 4001
        broker = make_broker("ibkr",
                             host=os.environ.get("IBKR_HOST", "127.0.0.1"),
                             port=int(os.environ.get("IBKR_PORT", "7497")),
                             client_id=int(os.environ.get("IBKR_CLIENT_ID", "1")))
    else:
        raise SystemExit(
            f"QTSYS_BROKER={venue!r}: the simulator has been removed from the "
            "live terminal — set QTSYS_BROKER=alpaca or ibkr")

    gw = ExecutionGateway(broker, RiskLimits(max_order_notional=60_000,
                                             max_position_notional=120_000,
                                             max_gross_leverage=2.0))
    hist: dict[str, list[dict]] = {}
    for sym, name, cls, _ok in UNIVERSE:
        df = load_real(sym).tail(2500)
        has_ohlc = {"open", "high", "low"}.issubset(df.columns)
        bars, prev = [], None
        for ix, r in df.iterrows():
            c = float(r["close"])
            if has_ohlc:
                o, h, l = float(r["open"]), float(r["high"]), float(r["low"])
            else:   # close-only real series: bar spans the two REAL closes
                o = prev if prev is not None else c
                h, l = max(o, c), min(o, c)
            bars.append({"t": str(ix.date()), "o": o, "h": h, "l": l, "c": c,
                         "v": None})
            prev = c
        hist[sym] = bars
    # a real venue supplies its own quotes and starts with whatever is actually
    # in the account — no seeded demo book, no replay ticks.
    broker.day_open_equity = _equity(broker)

    daemon = AgentDaemon(
        os.path.join(HERE, "qtsys_agents.db"),
        context={"quotes": {}, "account": broker.get_account})
    from .llm import make_llm_fn
    daemon.llm_fn = make_llm_fn()
    if daemon.llm_fn:
        daemon.log("__system__", f"LLM backends: {daemon.llm_fn.backends}")
    await daemon.start()

    from .sizing import posture_table
    state["posture_stats"] = posture_table()      # from the REAL backtest trades
    names = ("SURVIVAL", "BALANCED", "AGGRESSIVE")
    try:
        row = daemon.db.execute("SELECT v FROM kv WHERE k='posture_name'").fetchone()
        state["posture"] = names[int(row[0])] if row else "BALANCED"
    except Exception:
        state["posture"] = "BALANCED"
    state["base_limits"] = {"order": gw.limits.max_order_notional,
                            "position": gw.limits.max_position_notional}
    scale = POSTURE_SCALE[state["posture"]]
    gw.limits.max_order_notional *= scale
    gw.limits.max_position_notional *= scale
    daemon.context["posture"] = lambda: state["posture"]
    # give every agent live access to fundamentals + news intelligence
    from . import intel as _intel
    daemon.context["fundamentals"] = lambda s: _intel.fundamentals(s, _clsname(s))
    daemon.context["news"] = lambda s: _intel.news(s, _clsname(s))
    state.update(broker=broker, gw=gw, hist=hist, daemon=daemon,
                 venue=venue, vmap=VENUE_SYMBOLS.get(venue),
                 meta={s: (n, c) for s, n, c, _ in UNIVERSE})
    try:                                          # restore last scan across restarts
        from . import universe as _u
        last = _u.load_last_result("1Day")
        if last:
            state["uscan"] = {"running": False, "progress": "done", "result": last}
    except Exception:
        pass
    _load_alerts()
    asyncio.create_task(_tick_loop())
    asyncio.create_task(_daily_scan_loop())      # auto-run the morning scan daily
    asyncio.create_task(_prewarm_screener())     # warm the fundamentals cache
    asyncio.create_task(_alerts_loop())          # evaluate alerts continuously


async def _tick_loop() -> None:
    """LIVE quotes only. Every cycle, poll the venue for a fresh price on each
    tradable symbol; symbols the venue doesn't serve fall back to their latest
    REAL recorded close (marked with its date). No replay, no simulated tape."""
    broker = state["broker"]
    dead: set[str] = set()          # symbols this venue has refused to quote
    while True:
        q = state["daemon"].context["quotes"]
        for sym, bars in state["hist"].items():
            last_close, prev_close = bars[-1]["c"], bars[-2]["c"]
            price, asof = None, bars[-1]["t"]
            vmap = state.get("vmap")
            vsym = sym if vmap is None else vmap.get(sym)
            if sym in TRADABLE and vsym and sym not in dead:
                try:
                    p = await asyncio.to_thread(broker.get_quote, vsym)
                    if p and not math.isnan(p):
                        price, asof = p, "live"
                except Exception:
                    dead.add(sym)            # venue doesn't serve this symbol
            if price is None:
                price = last_close
            q[sym] = {"last": price,
                      "chg_pct": (price / prev_close - 1) * 100, "asof": asof}
        # deterministic circuit breaker (portfolio_risk.LIMITS): -3% on the day
        gw = state["gw"]
        if not gw.halted and broker.day_open_equity:
            day = _equity(broker) / broker.day_open_equity - 1
            if day <= -0.03:
                gw.halt(f"daily loss limit hit ({day:.1%}) — new entries blocked")
        await asyncio.sleep(3.0)     # universe poll cadence (the active symbol
        # is polled every ~1s separately via /api/quote for second-by-second)


def _quote_row(sym: str) -> dict:
    name, cls = state["meta"][sym]
    bars = state["hist"][sym]
    # live quote from the venue (filled by _tick_loop); latest REAL close otherwise
    live = state["daemon"].context["quotes"].get(sym, {})
    last = live.get("last", bars[-1]["c"])
    prev = bars[-2]["c"]
    return {"symbol": sym, "name": name, "cls": cls, "last": last,
            "chg": last - prev, "chg_pct": (last / prev - 1) * 100,
            "asof": live.get("asof", bars[-1]["t"]), "tradable": sym in TRADABLE,
            "spark": [b["c"] for b in bars[-40:]]}


def _tracking() -> dict:
    from .tracking import tracking_report
    return tracking_report(_fills(state["broker"]), _cls_of)


@app.get("/api/health")
def health(): return {"ok": True, "mode": "live",
                      "venue": state.get("venue", "?"),
                      "posture": state.get("posture", "BALANCED"),
                      "note": "live venue quotes and fills; no simulation",
                      "ts": time.time()}


@app.get("/api/quotes")
def quotes(): return [_quote_row(s) for s in state["hist"]]


@app.get("/api/history/{sym}")
def history(sym: str, bars: int = 380, tf: str = "1D"):
    if sym not in state["hist"]:
        raise HTTPException(404, "unknown symbol")
    if tf in ("", "1D", "1Day", "D"):                 # bundled daily history
        return {"symbol": sym, "tf": "1D", "bars": state["hist"][sym][-bars:]}
    # intraday: fetched live from the venue (only for venue-served symbols)
    broker = state["broker"]
    vmap = state.get("vmap")
    vsym = sym if vmap is None else vmap.get(sym)
    frames = {"1Min": "1Min", "5Min": "5Min", "15Min": "15Min", "1H": "1Hour"}
    if not vsym or tf not in frames or not hasattr(broker, "history"):
        raise HTTPException(400, f"{sym}: no intraday on this venue (analyse-only "
                                 "or unmapped symbol)")
    try:
        b = broker.history(vsym, bars, frames[tf])
    except Exception as e:
        raise HTTPException(502, f"intraday fetch failed: {str(e).splitlines()[0][:120]}")
    return {"symbol": sym, "tf": tf, "bars": b}


def _clsname(sym: str) -> str:
    return CLS.get(sym) or (state.get("meta", {}).get(sym, ("", ""))[1] or "")


@app.get("/api/news")
async def news(sym: str):
    """Merged, sentiment-tagged headlines from the venue feed (Alpaca) AND
    yfinance (Yahoo), deduped. Analyse-only symbols still get Yahoo coverage."""
    if sym not in state["hist"]:
        return {"symbol": sym, "items": []}
    cls = _clsname(sym)
    broker = state["broker"]
    vmap = state.get("vmap")
    vsym = sym if vmap is None else vmap.get(sym)
    items: list[dict] = []
    if hasattr(broker, "news") and vsym:          # venue feed (Alpaca)
        items += await asyncio.to_thread(broker.news, vsym, 25)
    from . import intel                            # yfinance / Yahoo feed
    items += await asyncio.to_thread(intel.news, sym, cls)
    seen, merged = set(), []                       # dedupe on headline, newest first
    for it in sorted(items, key=lambda x: x.get("ts", ""), reverse=True):
        k = (it.get("headline", "")[:60]).lower()
        if k and k not in seen:
            seen.add(k)
            merged.append(it)
    merged = merged[:30]
    # uniform sentiment tags (FinBERT if available, else lexicon)
    from . import nlp
    tags = await asyncio.to_thread(
        nlp.tag, [it.get("headline", "") + " " + it.get("summary", "") for it in merged])
    for it, tg in zip(merged, tags):
        it.update(tg)
    return {"symbol": sym, "items": merged, "engine": nlp.engine(),
            "narrative": await _news_narrative(sym, merged)}


_NARR_CACHE: dict = {}


async def _news_narrative(sym: str, items: list[dict]) -> str:
    """LLM synthesis of the headlines — cached per symbol for 10 min."""
    llm = getattr(state.get("daemon"), "llm_fn", None)
    if not llm or not items:
        return ""
    hit = _NARR_CACHE.get(sym)
    if hit and time.time() - hit[0] < 600:
        return hit[1]
    from . import nlp
    txt = await asyncio.to_thread(nlp.narrative, sym,
                                  [it.get("headline", "") for it in items], llm)
    _NARR_CACHE[sym] = (time.time(), txt)
    return txt


_ALERTS_FILE = os.path.join(HERE, "universe_cache", "alerts.json")
_ALERT_TYPES = {"price_above", "price_below", "change_above", "rsi_above", "rsi_below"}


def _rsi(closes, n=14):
    if len(closes) < n + 1:
        return None
    d = [closes[i] - closes[i - 1] for i in range(len(closes) - n, len(closes))]
    g = sum(x for x in d if x > 0) / n
    l = -sum(x for x in d if x < 0) / n
    return 100.0 if l == 0 else 100 - 100 / (1 + g / l)


def _eval_alert(a):
    sym = a["symbol"]
    q = state["daemon"].context["quotes"].get(sym, {})
    last, chg, val, typ = q.get("last"), q.get("chg_pct"), a["value"], a["type"]
    if last is None:
        return False, None
    if typ == "price_above" and last >= val:
        return True, f"{sym} {last:.2f} ≥ {val:g}"
    if typ == "price_below" and last <= val:
        return True, f"{sym} {last:.2f} ≤ {val:g}"
    if typ == "change_above" and chg is not None and abs(chg) >= val:
        return True, f"{sym} moved {chg:+.2f}% (≥{val:g}%)"
    if typ in ("rsi_above", "rsi_below"):
        r = _rsi([b["c"] for b in state["hist"].get(sym, [])])
        if r is None:
            return False, None
        if typ == "rsi_above" and r >= val:
            return True, f"{sym} RSI {r:.0f} ≥ {val:g}"
        if typ == "rsi_below" and r <= val:
            return True, f"{sym} RSI {r:.0f} ≤ {val:g}"
    return False, None


def _save_alerts():
    import json
    try:
        with open(_ALERTS_FILE, "w") as f:
            json.dump(state.get("alerts", []), f)
    except Exception:
        pass


def _load_alerts():
    import json
    try:
        with open(_ALERTS_FILE) as f:
            state["alerts"] = json.load(f)
    except Exception:
        state["alerts"] = []


async def _alerts_loop():
    await asyncio.sleep(15)
    while True:
        try:
            feed = state.setdefault("alert_feed", [])
            for a in state.get("alerts", []):
                if not a.get("armed", True):
                    continue
                fired, msg = _eval_alert(a)
                if fired and time.time() - a.get("last_fired", 0) > 300:  # 5-min debounce
                    a["last_fired"] = time.time()
                    feed.insert(0, {"ts": time.time(), "id": a["id"],
                                    "message": msg, "symbol": a["symbol"]})
                    del feed[100:]
                    state["daemon"].log("Alerts", "🔔 " + msg, "warn")
        except Exception:
            pass
        await asyncio.sleep(8)


@app.get("/api/alerts")
def alerts_list():
    return {"rules": state.get("alerts", []), "feed": state.get("alert_feed", [])}


@app.post("/api/alerts")
def alerts_create(body: dict):
    import uuid
    typ = body.get("type")
    sym = str(body.get("symbol", "")).upper().strip()
    if typ not in _ALERT_TYPES or not sym:
        raise HTTPException(400, "need a valid type and symbol")
    if sym not in state["hist"]:                 # resolve so the tick loop quotes it
        try:
            resolve(sym)
        except Exception:
            pass
    a = {"id": uuid.uuid4().hex[:8], "type": typ, "symbol": sym,
         "value": float(body.get("value", 0)), "note": body.get("note", ""),
         "armed": True, "created": time.time(), "last_fired": 0}
    state.setdefault("alerts", []).append(a)
    _save_alerts()
    return a


@app.delete("/api/alerts/{aid}")
def alerts_delete(aid: str):
    state["alerts"] = [a for a in state.get("alerts", []) if a["id"] != aid]
    _save_alerts()
    return {"ok": True}


# BQL-style field catalog: name -> (category, description). Pull any of these for
# any symbol via /api/data (JSON or CSV) — the "data-out" surface for Excel/Python.
_FIELDS = {
    "last": ("price", "last trade price"), "chg_pct": ("price", "daily % change"),
    "prev_close": ("price", "previous close"), "open": ("price", "last bar open"),
    "high": ("price", "last bar high"), "low": ("price", "last bar low"),
    "rsi14": ("technical", "14-period RSI (daily)"),
    "sma20": ("technical", "20-day simple MA"), "sma100": ("technical", "100-day simple MA"),
    "atr14": ("technical", "14-day ATR"), "mom_63": ("technical", "3-month momentum %"),
    "mom_252": ("technical", "12-month momentum %"),
    "vol_ann": ("technical", "annualized realized vol %"),
    "pe": ("fundamental", "trailing P/E"), "forward_pe": ("fundamental", "forward P/E"),
    "peg": ("fundamental", "PEG"), "eps": ("fundamental", "trailing EPS"),
    "rev_growth": ("fundamental", "revenue growth %"),
    "earnings_growth": ("fundamental", "earnings growth %"),
    "margin": ("fundamental", "profit margin %"), "debt_equity": ("fundamental", "debt/equity"),
    "beta": ("fundamental", "beta"), "div_yield": ("fundamental", "dividend yield %"),
    "target": ("fundamental", "mean analyst target"),
    "analyst": ("fundamental", "analyst recommendation"),
    "mcap": ("fundamental", "market cap"), "sector": ("fundamental", "GICS sector"),
    "industry": ("fundamental", "industry"),
    "news_count": ("news", "# recent headlines"),
    "sentiment": ("news", "mean net sentiment score"),
}
_FUND_KEY = {"pe": "pe", "forward_pe": "forward_pe", "peg": "peg", "eps": "eps",
             "rev_growth": "rev_growth_pct", "earnings_growth": "earnings_growth_pct",
             "margin": "profit_margin_pct", "debt_equity": "debt_to_equity",
             "beta": "beta", "div_yield": "div_yield_pct", "target": "target_mean",
             "analyst": "analyst", "mcap": "market_cap", "sector": "sector",
             "industry": "industry"}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _data_row(sym, fields) -> dict:
    if sym not in state["hist"]:
        try:
            resolve(sym)
        except Exception:
            pass
    bars = state["hist"].get(sym, [])
    closes = [b["c"] for b in bars]
    q = state["daemon"].context["quotes"].get(sym, {})
    cats = {_FIELDS.get(f, ("", ""))[0] for f in fields}
    m = {}
    if "fundamental" in cats:
        from . import intel
        try:
            m = intel.fundamentals(sym, _clsname(sym) or "Equity").get("metrics", {}) or {}
        except Exception:
            m = {}
    news = []
    if "news" in cats:
        from . import intel
        try:
            news = intel.news(sym, _clsname(sym) or "Equity")
        except Exception:
            news = []
    out = {}
    for f in fields:
        out[f] = _field_value(f, sym, q, bars, closes, m, news)
    return out


def _field_value(f, sym, q, bars, closes, m, news):
    import numpy as np
    n = len(closes)
    if f == "last":
        return q.get("last") or (closes[-1] if closes else None)
    if f == "chg_pct":
        return q.get("chg_pct")
    if f == "prev_close":
        return closes[-2] if n >= 2 else None
    if f in ("open", "high", "low") and bars:
        return bars[-1].get({"open": "o", "high": "h", "low": "l"}[f])
    if f == "rsi14":
        return _rsi(closes)
    if f == "sma20":
        return float(np.mean(closes[-20:])) if n >= 20 else None
    if f == "sma100":
        return float(np.mean(closes[-100:])) if n >= 100 else None
    if f == "atr14" and len(bars) > 15:
        tr = [max(bars[i]["h"] - bars[i]["l"], abs(bars[i]["h"] - bars[i - 1]["c"]),
                  abs(bars[i]["l"] - bars[i - 1]["c"])) for i in range(len(bars) - 14, len(bars))]
        return float(np.mean(tr))
    if f == "mom_63":
        return round(closes[-1] / closes[-63] - 1, 4) * 100 if n > 63 else None
    if f == "mom_252":
        return round(closes[-1] / closes[-252] - 1, 4) * 100 if n > 252 else None
    if f == "vol_ann" and n > 21:
        r = np.diff(closes[-21:]) / np.array(closes[-21:-1])
        return round(float(np.std(r) * np.sqrt(252)) * 100, 2)
    if f in _FUND_KEY:
        return m.get(_FUND_KEY[f])
    if f == "news_count":
        return len(news)
    if f == "sentiment":
        return round(_mean([x.get("sent_score") for x in news]) or 0, 2) if news else None
    return None


@app.get("/api/data/fields")
def data_fields():
    """The BQL-style field catalog you can request from /api/data."""
    return {"fields": [{"name": k, "category": v[0], "desc": v[1]}
                       for k, v in _FIELDS.items()]}


@app.get("/api/data")
async def data_api(symbols: str, fields: str = "last,chg_pct", format: str = "json"):
    """Tabular data-out (BQL-style): any fields for any symbols, JSON or CSV.
    e.g. /api/data?symbols=AAPL,MSFT&fields=last,pe,rsi14,mom_252&format=csv
    Pull straight into pandas: pd.read_csv(URL)  or Excel Power Query."""
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()][:50]
    flds = [f.strip() for f in fields.split(",") if f.strip() and f.strip() in _FIELDS]
    if not syms or not flds:
        raise HTTPException(400, "need symbols and valid fields (see /api/data/fields)")
    rows = await asyncio.to_thread(
        lambda: [dict(symbol=s, **_data_row(s, flds)) for s in syms])
    if format == "csv":
        import csv
        import io
        from fastapi.responses import PlainTextResponse
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["symbol"] + flds)
        for r in rows:
            w.writerow([r.get("symbol")] + [r.get(f) for f in flds])
        return PlainTextResponse(buf.getvalue(), media_type="text/csv")
    return {"fields": flds, "rows": rows}


@app.get("/api/screen")
async def screen():
    """Fundamental screener universe: sector constituents + your book +
    watchlist, each enriched with fundamentals (yfinance) and the last daily
    scan's technicals. Client filters/sorts. Pre-warmed on boot for speed."""
    return {"rows": await asyncio.to_thread(_screen_rows, state["broker"])}


def _screen_rows(broker) -> list[dict]:
    from . import intel, sectors, universe
    import concurrent.futures
    syms = set()
    for members in sectors.CONSTITUENTS.values():
        syms.update(members)
    try:
        syms.update(p.symbol for p in broker.get_positions()
                    if "/" not in p.symbol and len(p.symbol) <= 5)
    except Exception:
        pass
    syms.update(v for v in (state.get("vmap") or {}).values() if "/" not in v)
    syms = sorted(syms)[:180]
    tech = {i["asset"]: i for i in (universe.load_last_result("1Day") or {}).get("instruments", [])}

    def one(s):
        try:
            return s, (intel.fundamentals(s, "Equity").get("metrics") or {})
        except Exception:
            return s, {}
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
        for s, m in ex.map(one, syms):
            if not m or not m.get("pe") and not m.get("market_cap"):
                continue
            t = tech.get(s, {})
            rows.append({
                "symbol": s, "sector": m.get("sector"), "industry": m.get("industry"),
                "market_cap": m.get("market_cap"), "pe": m.get("pe"),
                "forward_pe": m.get("forward_pe"), "peg": m.get("peg"),
                "eps": m.get("eps"), "rev_growth": m.get("rev_growth_pct"),
                "earnings_growth": m.get("earnings_growth_pct"),
                "margin": m.get("profit_margin_pct"), "debt_equity": m.get("debt_to_equity"),
                "beta": m.get("beta"), "div_yield": m.get("div_yield_pct"),
                "target": m.get("target_mean"), "analyst": m.get("analyst"),
                "mom_63": t.get("mom_63"), "mom_252": t.get("mom_252"),
                "rvol": t.get("rvol_20")})
    return rows


@app.get("/api/calendar")
async def calendar():
    """Economic releases (FRED) + earnings/dividend dates (yfinance) for your
    held positions, watchlist, and sector bellwethers."""
    from . import calendars, sectors
    syms = set()
    try:
        syms.update(p.symbol for p in state["broker"].get_positions()
                    if "/" not in p.symbol and len(p.symbol) <= 5)
    except Exception:
        pass
    syms.update(v for v in (state.get("vmap") or {}).values() if "/" not in v)
    for members in sectors.CONSTITUENTS.values():
        syms.update(members[:4])                 # a few bellwethers per sector
    econ = await asyncio.to_thread(calendars.economic)
    corp = await asyncio.to_thread(calendars.corporate, list(syms))
    return {"economic": econ, "earnings": corp.get("earnings", []),
            "dividends": corp.get("dividends", [])}


@app.get("/api/options/{sym}")
async def options_chain(sym: str, exp: str = ""):
    """Live option chain for an underlying, greeks/IV enriched. Grouped by
    expiration; each strike carries its call and put side by side."""
    broker = state["broker"]
    if not hasattr(broker, "option_chain"):
        raise HTTPException(400, "options need an Alpaca venue")
    return await asyncio.to_thread(_build_chain, broker, sym.upper(), exp)


def _build_chain(broker, sym, exp) -> dict:
    from . import intel, options
    contracts = broker.option_chain(sym)
    if not contracts:
        return {"underlying": sym, "spot": None, "expirations": [],
                "chain": [], "note": "no options for this underlying"}
    try:
        spot = broker.get_quote(sym)
    except Exception:
        spot = None
    r = 0.04
    try:                                        # risk-free from FRED 3-month
        v = intel._fred_latest("DGS3MO")
        if v:
            r = float(v) / 100.0
    except Exception:
        pass
    enriched = options.enrich_chain(contracts, spot or 0.0, r)
    exps = sorted({c["expiration"] for c in enriched})
    pick = exp if exp in exps else (exps[0] if exps else "")
    byk: dict = {}
    for c in enriched:
        if c["expiration"] != pick:
            continue
        row = byk.setdefault(c["strike"], {"strike": c["strike"], "call": None, "put": None})
        row[c["type"]] = {k: c[k] for k in ("symbol", "bid", "ask", "last", "mid",
                                            "open_interest", "iv", "delta", "gamma",
                                            "theta", "vega")}
    chain = [byk[k] for k in sorted(byk)]
    return {"underlying": sym, "spot": spot, "r": r, "expiration": pick,
            "expirations": exps, "chain": chain}


@app.get("/api/industry")
async def industry(sym: str):
    """Sector of `sym` + its constituents, each with LIVE % change (computed in
    Python from Alpaca snapshots) and market-cap weight; plus the sector's
    market-cap-weighted daily change."""
    return await asyncio.to_thread(_build_industry, state["broker"], sym)


def _build_industry(broker, sym) -> dict:
    from . import intel, sectors
    import concurrent.futures
    cls = _clsname(sym) or "Equity"
    f = intel.fundamentals(sym, cls)
    sector = (f.get("metrics") or {}).get("sector")
    members = sectors.constituents(sector)
    etf = sectors.SECTOR_ETF.get(sector or "")
    if not members:
        return {"sector": sector or "—", "constituents": [],
                "total_change_pct": None, "etf": etf,
                "note": "sector constituents are available for equities only"}
    if sym not in members and cls == "Equity":
        members = [sym] + members
    # live % change from one batched snapshot (latest trade vs previous close)
    changes = {}
    try:
        from alpaca.data.requests import StockSnapshotRequest
        snaps = broker.d.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=members))
        for m, sn in (snaps or {}).items():
            lt = getattr(sn, "latest_trade", None)
            db = getattr(sn, "daily_bar", None)
            pdb = getattr(sn, "previous_daily_bar", None)
            last = float(lt.price) if lt else (float(db.close) if db else None)
            prev = float(pdb.close) if pdb else (float(db.open) if db else None)
            if last and prev:
                changes[m] = (last, prev, last / prev - 1.0)
    except Exception:
        pass
    # market caps, fetched concurrently (cached in intel after first hit)
    def _mc(m):
        try:
            return m, (intel.fundamentals(m, "Equity").get("metrics") or {}).get("market_cap") or 0
        except Exception:
            return m, 0
    mcaps = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        for m, mc in ex.map(_mc, members):
            mcaps[m] = float(mc or 0)
    total_mcap = sum(mcaps.values()) or 1.0
    rows = []
    for m in members:
        last, prev, chg = changes.get(m, (None, None, None))
        rows.append({"symbol": m, "last": last, "change_pct": chg,
                     "mcap": mcaps.get(m, 0.0),
                     "weight": mcaps.get(m, 0.0) / total_mcap})
    rows.sort(key=lambda r: r["weight"], reverse=True)
    total = sum(r["weight"] * (r["change_pct"] or 0.0) for r in rows)
    return {"sector": sector, "etf": etf, "constituents": rows,
            "total_change_pct": total, "asof": time.time()}


@app.get("/api/fundamentals")
async def fundamentals(sym: str):
    """Normalised fundamentals for the instrument (equity ratios / crypto
    metrics / FX-commodity macro drivers)."""
    if sym not in state["hist"]:
        raise HTTPException(404, "unknown symbol")
    from . import intel
    return await asyncio.to_thread(intel.fundamentals, sym, _clsname(sym))


@app.get("/api/quote")
async def quote_one(sym: str):
    """Fresh single-symbol last trade — for second-by-second updates on the
    symbol the user is watching."""
    if sym not in state["hist"]:
        raise HTTPException(404, "unknown symbol")
    broker = state["broker"]
    vmap = state.get("vmap")
    vsym = sym if vmap is None else vmap.get(sym)
    if vsym:
        try:
            p = await asyncio.to_thread(broker.get_quote, vsym)
            if p and not math.isnan(p):
                state["daemon"].context["quotes"].setdefault(sym, {})["last"] = p
                return {"symbol": sym, "last": p, "asof": "live"}
        except Exception:
            pass
    q = state["daemon"].context["quotes"].get(sym, {})
    return {"symbol": sym, "last": q.get("last"), "asof": q.get("asof", "")}


@app.get("/api/resolve")
def resolve(q: str):
    """Search the venue for any ticker and add it live: US equities/ETFs and
    crypto pairs (Alpaca). Forex beyond the bundled majors needs IBKR/Oanda."""
    broker = state["broker"]
    venue = state.get("venue", "?")
    raw = (q or "").strip().upper().replace(" ", "")
    if not raw:
        raise HTTPException(400, "empty query")
    if raw in state["hist"]:
        return {"added": False, "symbol": raw, "row": _quote_row(raw)}
    if not hasattr(broker, "history"):
        raise HTTPException(400, f"ticker search is not supported on {venue}")
    # candidates: explicit crypto pair, else try equity then crypto-vs-USD
    cands = ([(raw.split("/")[0], raw, "Crypto")] if "/" in raw
             else [(raw, raw, "Equity"), (raw, raw + "/USD", "Crypto")])
    last_err = "no data"
    for eng, ven, cls in cands:
        try:
            hist = broker.history(ven, 400)
            if not hist or len(hist) < 5:
                last_err = "no history"
                continue
            name = f"{eng} — {cls} (live via {venue})"
            state["hist"][eng] = hist
            state["meta"][eng] = (name, cls)
            CLS[eng] = cls
            TRADABLE.add(eng)
            if state.get("vmap") is not None:
                state["vmap"][eng] = ven
            state["daemon"].log("Research Analyst",
                                f"universe += {eng} ({cls}) via {venue}")
            return {"added": True, "symbol": eng, "row": _quote_row(eng)}
        except Exception as e:                       # not served as this class
            last_err = str(e).split("\n")[0][:140]
    raise HTTPException(404, f"'{raw}' not found on {venue} ({last_err}). "
                             "Forex beyond the bundled majors needs IBKR/Oanda.")


@app.get("/api/account")
def account():
    a = state["broker"].get_account()
    a["halted"] = state["gw"].halted
    a["halt_reason"] = state["gw"].halt_reason
    return a


@app.get("/api/positions")
def positions():
    b = state["broker"]
    out = []
    for p in b.get_positions():
        try:                             # options/delisted/odd symbols may not
            last = b.get_quote(p.symbol)  # quote — never let one blank the book
            if last is None or (isinstance(last, float) and math.isnan(last)):
                last = p.avg_price
        except Exception:
            last = p.avg_price
        out.append(p.to_dict(last))
    return out


@app.get("/api/orders")
def orders(open_only: bool = False):
    return [o.to_dict() for o in state["broker"].get_orders(open_only)][::-1]


@app.post("/api/orders")
def place(order: dict):
    vmap = state.get("vmap")
    vsym = order["symbol"] if vmap is None else vmap.get(order["symbol"])
    if order["symbol"] not in TRADABLE or not vsym:
        return JSONResponse({"status": "rejected",
                             "reason": f"{order['symbol']} is analyse-only here "
                                       "(this venue does not serve it)"},
                            status_code=400)
    o = Order(vsym, order["side"], float(order["qty"]),
              order.get("type", "market"),
              float(order["limit_price"]) if order.get("limit_price") else None)
    res = state["gw"].submit(o)
    code = 200 if res.status != "rejected" else 400
    return JSONResponse(res.to_dict(), status_code=code)


@app.post("/api/orders/{oid}/cancel")
def cancel(oid: str): return {"cancelled": state["broker"].cancel(oid)}


@app.post("/api/kill")
def kill():
    state["gw"].halt("manual kill switch")
    state["daemon"].log("system", "KILL SWITCH — book flattened, trading halted",
                        "error")
    return {"halted": True}


@app.post("/api/resume")
def resume(): state["gw"].resume(); return {"halted": False}


@app.get("/api/posture")
async def api_posture():
    # posture stats are the REAL backtest stats; tracking compares them to the
    # account's actual realised performance
    return {"current": state.get("posture", "BALANCED"),
            "stats": state.get("posture_stats", {}),
            **await asyncio.to_thread(_tracking)}


@app.post("/api/posture")
async def api_posture_set(body: dict):
    p = str(body.get("posture", "")).upper()
    if p not in POSTURE_SCALE:
        raise HTTPException(400, "posture must be SURVIVAL | BALANCED | AGGRESSIVE")
    state["posture"] = p
    d = state["daemon"]
    d.db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v REAL)")
    d.db.execute("INSERT OR REPLACE INTO kv VALUES ('posture_name', ?)",
                 (float(("SURVIVAL", "BALANCED", "AGGRESSIVE").index(p)),))
    d.db.commit()
    scale = POSTURE_SCALE[p]
    base = state["base_limits"]
    gw = state["gw"]
    gw.limits.max_order_notional = base["order"] * scale
    gw.limits.max_position_notional = base["position"] * scale
    d.log("Risk Officer", f"POSTURE set to {p} — every agent now sizes at "
          f"{state['posture_stats'][p]['risk_per_trade']:.2%}/trade; "
          f"gateway order cap scaled x{scale}", "warn")
    return {"current": p, "stats": state["posture_stats"]}


@app.get("/api/tracking")
async def api_tracking():
    """Backtest baseline vs the account's REAL realised trades, and the drift."""
    return await asyncio.to_thread(_tracking)


@app.get("/api/strategies")
def strategies():
    """The REAL research registry (registry_summary.csv), with the agent's
    verdict per strategy. This is a snapshot from the last `python -m qtsys.sweep`
    run — it does NOT auto-update; asof is the file's mtime."""
    import csv
    p = os.path.join(HERE, "registry_summary.csv")
    if not os.path.exists(p):
        return {"asof": None, "rows": []}

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            d = num(r.get("dsr"))
            rows.append({
                "id": r.get("id"), "family": r.get("family"),
                "train_n": num(r.get("train_n")), "train_exp": num(r.get("train_exp")),
                "test_n": num(r.get("test_n")), "test_wr": num(r.get("test_wr")),
                "test_exp": num(r.get("test_exp")), "test_pf": num(r.get("test_pf")),
                "dsr": d, "status": r.get("status", ""),
                "verdict": "pass" if (d or 0) >= 0.95 else
                           "watch" if (d or 0) >= 0.80 else "reject"})
    rows.sort(key=lambda x: (x["dsr"] or 0), reverse=True)
    return {"asof": os.path.getmtime(p), "rows": rows}


@app.post("/api/universe/scan")
async def universe_scan(cap: int = 3000, tf: str = "1Day"):
    """Kick off a full-universe morning scan in the background at timeframe `tf`
    (1Day|1Hour|15Min|5Min). Non-blocking; poll /api/universe/status."""
    if state.get("uscan", {}).get("running"):
        return {"running": True, "note": "a scan is already in progress"}
    if tf not in ("1Day", "1Hour", "15Min", "5Min"):
        raise HTTPException(400, "tf must be 1Day | 1Hour | 15Min | 5Min")
    broker = state["broker"]
    if not hasattr(broker, "history"):
        raise HTTPException(400, "full-universe scan needs an Alpaca venue")
    vmap = state.get("vmap") or {}
    watch = [v for v in vmap.values()]
    state["uscan"] = {"running": True, "progress": "starting…", "started": time.time()}

    def prog(m):
        state["uscan"]["progress"] = m

    asyncio.create_task(_run_universe_scan(broker, watch, cap, prog, tf))
    return {"running": True, "cap": cap, "tf": tf, "watchlist": len(watch)}


async def _prewarm_screener():
    """Pre-fetch the screener universe's fundamentals so the SCREEN tab is
    instant when first opened."""
    await asyncio.sleep(20)
    try:
        await asyncio.to_thread(_screen_rows, state["broker"])
    except Exception:
        pass


async def _daily_scan_loop():
    """Run the full-universe scan once per calendar day, automatically, so the
    selector's warm-up accumulates without manual clicks. Best-effort; skips if a
    scan already ran today or one is in progress."""
    import datetime
    await asyncio.sleep(45)                       # let boot settle
    broker = state.get("broker")
    if not hasattr(broker, "history"):
        return                                    # only on Alpaca
    cap = int(os.environ.get("QTSYS_SCAN_CAP", "3000"))
    # timeframes the daily auto-scan runs — daily first (ML), then intraday
    tfs = [t.strip() for t in os.environ.get("QTSYS_SCAN_TFS", "1Day,1Hour").split(",")
           if t.strip()]
    while True:
        try:
            from . import selector
            today = datetime.date.today().isoformat()
            db = selector._db()
            done = db.execute("SELECT 1 FROM feat WHERE date=? LIMIT 1",
                              (today,)).fetchone()
            db.close()
            if not done and not state.get("uscan", {}).get("running"):
                watch = [v for v in (state.get("vmap") or {}).values()]
                for tf in tfs:                    # daily + configured intraday tfs
                    state["uscan"] = {"running": True, "started": time.time(),
                                      "progress": f"auto {tf} scan…"}
                    await _run_universe_scan(
                        broker, watch, cap,
                        lambda m: state["uscan"].__setitem__("progress", m), tf)
        except Exception:
            pass
        await asyncio.sleep(3600)                 # re-check hourly


async def _run_universe_scan(broker, watch, cap, prog, tf="1Day"):
    try:
        from . import universe, selector
        res = await asyncio.to_thread(universe.run_scan, broker, watch, cap, 2e6, prog, tf)
        await asyncio.to_thread(selector.train)      # retrain after every scan
        state["uscan"] = {"running": False, "progress": "done",
                          "result": res, "finished": time.time()}
    except Exception as e:
        state["uscan"] = {"running": False,
                          "progress": f"error: {str(e).splitlines()[0][:160]}"}


@app.get("/api/universe/status")
def universe_status():
    from . import selector
    u = state.get("uscan", {})
    return {"running": u.get("running", False), "progress": u.get("progress"),
            "last": {k: u["result"][k] for k in
                     ("phase", "universe", "scanned", "n_setups", "took", "asof")}
            if u.get("result") else None,
            "selector": selector.status()}


@app.get("/api/universe/results")
def universe_results(tf: str = ""):
    if tf:                                        # per-timeframe persisted result
        from . import universe
        r = universe.load_last_result(tf) or {}
    else:
        r = state.get("uscan", {}).get("result") or {}
    return {"tf": r.get("tf"), "setups": r.get("setups", []),
            "instruments": r.get("instruments", [])}


@app.post("/api/universe/train")
def universe_train():
    """Retrain the ML selector on accumulated scan history (advances the phase
    once the recall gate passes)."""
    from . import selector
    return selector.train()


@app.get("/api/audit")
def audit():
    """Detailed Strategy-Engineer audit: every strategy × every instrument it
    analysed, with the per-pair outcome (trades, win rate, expectancy, PF) and
    the strategy's DSR verdict. Plus the exact instrument set the morning
    analysis actually covers."""
    import csv

    def num(v):
        try:
            f = float(v)
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None
    HERE_ = HERE
    summ = {}
    sp = os.path.join(HERE_, "registry_summary.csv")
    if os.path.exists(sp):
        with open(sp) as f:
            for r in csv.DictReader(f):
                d = num(r.get("dsr"))
                summ[r["id"]] = {
                    "family": r.get("family"), "dsr": d, "status": r.get("status", ""),
                    "verdict": "pass" if (d or 0) >= 0.95 else
                               "watch" if (d or 0) >= 0.80 else "reject",
                    "train_exp": num(r.get("train_exp")), "test_exp": num(r.get("test_exp"))}
    rows, assets, strategies = [], set(), set()
    rp = os.path.join(HERE_, "registry_results.csv")
    if os.path.exists(rp):
        with open(rp) as f:
            for r in csv.DictReader(f):
                s = summ.get(r["spec"], {})
                a = r["asset"]
                if "/" not in a:
                    assets.add(a)
                strategies.add(r["spec"])
                rows.append({
                    "strategy": r["spec"], "asset": a, "family": s.get("family"),
                    "n": num(r.get("n")), "win_rate": num(r.get("win_rate")),
                    "expectancy": num(r.get("expectancy")),
                    "profit_factor": num(r.get("profit_factor")),
                    "dsr": s.get("dsr"), "verdict": s.get("verdict", ""),
                    "status": s.get("status", "")})
    rows.sort(key=lambda x: (x["expectancy"] if x["expectancy"] is not None else -9),
              reverse=True)
    return {"asof": os.path.getmtime(rp) if os.path.exists(rp) else None,
            "results": rows, "universe": sorted(assets),
            "strategies": sorted(strategies), "specs": summ}


@app.get("/api/reports")
def reports_list():
    """Journals/reports the agents have filed to reports/ (briefings, risk,
    daily wraps)."""
    d = os.path.join(HERE, "reports")
    out = []
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d), reverse=True):
            if fn.endswith(".txt"):
                fp = os.path.join(d, fn)
                out.append({"name": fn, "ts": os.path.getmtime(fp),
                            "size": os.path.getsize(fp)})
    return {"reports": out}


@app.get("/api/reports/{name}")
def report_read(name: str):
    fp = os.path.join(HERE, "reports", os.path.basename(name))   # no traversal
    if not os.path.isfile(fp):
        raise HTTPException(404, "no such report")
    with open(fp) as f:
        return {"name": os.path.basename(name), "text": f.read()}


@app.get("/api/journal")
def journal():
    """The trade journal weekly review (empty until trades are logged)."""
    jp = os.path.join(HERE, "journal.db")
    if not os.path.exists(jp):
        return {"text": "No trades journaled yet — the trade journal fills once "
                        "live/paper fills are recorded."}
    try:
        from .journal import Journal
        return {"text": Journal(jp).weekly_review()}
    except Exception as e:
        return {"text": f"journal read error: {e}"}


@app.get("/api/scan")
async def api_scan():
    """Morning scan (cards 1/4): fresh setups ranked by their own real
    out-of-sample track records. Cached for an hour."""
    import time as _t
    if _t.time() - state.get("scan_ts", 0) > 3600:
        from .routine import scan
        df = await asyncio.to_thread(scan)
        state["scan"] = df.to_dict(orient="records") if len(df) else []
        state["scan_ts"] = _t.time()
    return {"asof": state["scan_ts"], "setups": state["scan"]}


@app.get("/api/agents")
def agents(): return state["daemon"].status()


@app.post("/api/agents/toggle")
def toggle(body: dict):
    state["daemon"].toggle(body.get("name"), bool(body["enabled"]))
    return state["daemon"].status()


@app.get("/api/agents/log")
def agent_log(limit: int = 60): return state["daemon"].recent_log(limit)


@app.get("/api/fills")
def fills(): return _fills(state["broker"])[::-1][:50]


@app.get("/")
def index(): return FileResponse(os.path.join(HERE, "terminal.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
