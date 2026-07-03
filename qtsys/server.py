"""server.py — run:  pip install fastapi uvicorn && python -m qtsys.server

Serves the terminal at http://localhost:8000 in LIVE mode: the identical
terminal.html that runs standalone in demo mode detects this API and switches
to real state — PaperBroker by default ($0), or any venue via make_broker()
("alpaca" paper, "ibkr", "ccxt" sandbox, "oanda" practice, "tradier" paper).
"""
from __future__ import annotations

import asyncio
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

# 100% REAL bundled data. In sandbox/demo the tape REPLAYS real daily history
# (1 bar/second, clearly labelled); with broker keys (user machine) quotes come
# from the venue's live feed instead. No simulated prices exist in this system.
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
# engine symbol -> venue symbol, per venue. Anything not mapped is NOT sent to
# the venue (engine codes like "WTI" collide with unrelated NYSE tickers).
VENUE_SYMBOLS = {
    "alpaca": {"BTC": "BTC/USD", "ETH": "ETH/USD"},
    "ibkr": None,   # None = venue serves the engine symbols directly
}
MONTHLY = {"GOLD", "SPX"}          # page-only: pinned to their latest real bar
REPLAY_BARS = 250          # the live tape replays the last 250 REAL daily bars

app = FastAPI(title="qtsys terminal API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

state: dict = {}


def _equity(broker) -> float:
    # PaperBroker has equity(); real adapters expose it via get_account()
    if hasattr(broker, "equity"):
        return broker.equity()
    return float(broker.get_account().get("equity", 0.0))


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
    broker.day_open_equity = _equity(broker)

    daemon = AgentDaemon(
        os.path.join(HERE, "qtsys_agents.db"),
        context={"quotes": {}, "account": broker.get_account})
    from .llm import make_llm_fn
    daemon.llm_fn = make_llm_fn()
    if daemon.llm_fn:
        daemon.log("__system__", f"LLM backends: {daemon.llm_fn.backends}")
    await daemon.start()
    state.update(broker=broker, gw=gw, hist=hist, daemon=daemon,
                 vmap=VENUE_SYMBOLS.get(venue),
                 meta={s: (n, c) for s, n, c, _ in UNIVERSE})
    asyncio.create_task(_tick_loop())


async def _tick_loop() -> None:
    """LIVE quotes only. Every cycle, poll the venue for a fresh price on each
    tradable symbol; symbols the venue doesn't serve fall back to their latest
    REAL recorded close (marked with its date). No replay, no simulated tape."""
    import math
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
                      "chg_pct": (price / prev_close - 1) * 100,
                      "asof": asof}
        # deterministic circuit breaker (portfolio_risk.LIMITS): -3% on the day
        gw = state["gw"]
        if not gw.halted and broker.day_open_equity:
            day = _equity(broker) / broker.day_open_equity - 1
            if day <= -0.03:
                gw.halt(f"daily loss limit hit ({day:.1%}) — new entries blocked")
        await asyncio.sleep(5.0)     # venue-poll cadence (rate-limit friendly)


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


@app.get("/api/health")
def health(): return {"ok": True, "mode": "live",
                      "note": "live venue quotes and fills; no simulation",
                      "ts": time.time()}


@app.get("/api/quotes")
def quotes(): return [_quote_row(s) for s in state["hist"]]


@app.get("/api/history/{sym}")
def history(sym: str, bars: int = 380):
    if sym not in state["hist"]:
        raise HTTPException(404, "unknown symbol")
    return {"symbol": sym, "bars": state["hist"][sym][-bars:]}


@app.get("/api/account")
def account():
    a = state["broker"].get_account()
    a["halted"] = state["gw"].halted
    a["halt_reason"] = state["gw"].halt_reason
    return a


@app.get("/api/positions")
def positions():
    b: PaperBroker = state["broker"]
    return [p.to_dict(b.get_quote(p.symbol)) for p in b.get_positions()]


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


@app.get("/api/scan")
async def api_scan():
    """Morning scan (cards 1/4): fresh setups ranked by their own real
    out-of-sample track records. Cached for an hour."""
    import time as _t
    if _t.time() - state.get("scan_ts", 0) > 3600:
        from .routine import morning_briefing, scan
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
def fills(): return state["broker"].fills[::-1][:50]


@app.get("/")
def index(): return FileResponse(os.path.join(HERE, "terminal.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
