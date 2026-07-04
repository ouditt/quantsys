"""universe.py — broad Alpaca universe: enumerate → liquidity-filter → batch-fetch
→ scan with the single-instrument strategy grid.

Flow (all Alpaca, batched, cached):
  1. candidate_symbols(): tradable equities on major exchanges + crypto + watchlist
  2. liquid_universe(): cheap snapshot pre-filter (price & dollar-volume) — avoids
     fetching 2y of bars for thousands of illiquid names
  3. fetch_bars(): daily OHLCV for the liquid set, multi-symbol requests, disk-cached
  4. scan_universe(): run the REGISTRY's single-instrument strategies over every
     instrument, emit fresh setups + per-instrument features (for the ML selector)

Scan compute parallelises across CPU cores. Nothing here runs on a request path —
it's a morning job. See selector.py for the ML narrowing that eventually shrinks
the scanned set.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
CACHE = os.path.join(HERE, "universe_cache")
_GOOD_EXCH = {"NYSE", "NASDAQ", "ARCA", "BATS", "AMEX", "NYSEARCA"}


# ---------------------------------------------------------------- enumeration
def candidate_symbols(broker, watchlist=(), cap: int = 8000) -> list[str]:
    """Tradable US equities on major exchanges + all crypto + the watchlist."""
    from alpaca.trading.requests import GetAssetsRequest
    from alpaca.trading.enums import AssetClass, AssetStatus
    out: list[str] = []
    try:
        eq = broker.c.get_all_assets(GetAssetsRequest(
            status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY))
        for a in eq:
            ex = getattr(a, "exchange", None)
            if a.tradable and ex and getattr(ex, "value", str(ex)) in _GOOD_EXCH:
                out.append(a.symbol)
        cr = broker.c.get_all_assets(GetAssetsRequest(
            status=AssetStatus.ACTIVE, asset_class=AssetClass.CRYPTO))
        out += [a.symbol for a in cr if a.tradable]
    except Exception:
        pass
    out = out[:cap]
    for w in watchlist:                       # watchlist is always included
        if w not in out:
            out.append(w)
    return out


# ------------------------------------------------------------- liquidity filter
def liquid_universe(broker, symbols, min_price=1.0, min_dollar_vol=2e6,
                    cap=3000) -> list[str]:
    """Cheap snapshot pre-filter: keep names above a price and dollar-volume floor,
    ranked by dollar volume, capped. One snapshot call per ~1000 symbols."""
    from alpaca.data.requests import StockSnapshotRequest
    scored: list[tuple[str, float]] = []
    eq = [s for s in symbols if "/" not in s]
    cry = [s for s in symbols if "/" in s]
    for i in range(0, len(eq), 900):
        chunk = eq[i:i + 900]
        try:
            snaps = broker.d.get_stock_snapshot(
                StockSnapshotRequest(symbol_or_symbols=chunk))
        except Exception:
            continue
        for sym, sn in (snaps or {}).items():
            bar = getattr(sn, "daily_bar", None) or getattr(sn, "minute_bar", None)
            if not bar:
                continue
            px, vol = float(bar.close or 0), float(bar.volume or 0)
            if px >= min_price and px * vol >= min_dollar_vol:
                scored.append((sym, px * vol))
    scored.sort(key=lambda x: x[1], reverse=True)
    out = [s for s, _ in scored[:cap]]
    return out + cry                          # crypto always kept (liquid)


# ------------------------------------------------------------------ batch bars
_TF_FRAMES = {"1Min": (1, "Minute"), "5Min": (5, "Minute"), "15Min": (15, "Minute"),
              "1Hour": (1, "Hour"), "1Day": (1, "Day")}
_PER_DAY = {"1Min": 390, "5Min": 78, "15Min": 26, "1Hour": 7, "1Day": 1}


def fetch_bars(broker, symbols, n=400, tf="1Day", use_cache=True) -> dict[str, pd.DataFrame]:
    """OHLCV per symbol at any timeframe via multi-symbol requests, disk-cached
    (parquet, keyed by tf). Returns {sym: DataFrame(index=ts, OHLCV)}."""
    os.makedirs(CACHE, exist_ok=True)
    ttl = 20 if tf == "1Day" else 2                # intraday bars go stale fast
    out: dict[str, pd.DataFrame] = {}
    todo = []
    for s in symbols:
        fp = os.path.join(CACHE, _safe(s) + "_" + tf + ".parquet")
        if use_cache and os.path.exists(fp) and _fresh(fp, ttl):
            try:
                out[s] = pd.read_parquet(fp)
                continue
            except Exception:
                pass
        todo.append(s)
    eq = [s for s in todo if "/" not in s]
    cry = [s for s in todo if "/" in s]
    _fetch_into(broker, eq, n, out, False, tf)
    _fetch_into(broker, cry, n, out, True, tf)
    return out


def _fetch_into(broker, symbols, n, out, crypto, tf):
    if not symbols:
        return
    from datetime import datetime, timedelta, timezone
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    amt, unit = _TF_FRAMES.get(tf, (1, "Day"))
    frame = TimeFrame(amt, getattr(TimeFrameUnit, unit))
    per = _PER_DAY.get(tf, 1)
    span = int(n * 1.7) + 7 if tf == "1Day" else max(2, int(n / per * 1.6) + 3)
    start = datetime.now(timezone.utc) - timedelta(days=span)
    intraday = tf != "1Day"
    step = 200 if not intraday else 40        # bound intraday response size
    for i in range(0, len(symbols), step):
        chunk = symbols[i:i + step]
        try:
            if crypto:
                from alpaca.data.requests import CryptoBarsRequest
                bs = broker.dc.get_crypto_bars(CryptoBarsRequest(
                    symbol_or_symbols=chunk, timeframe=frame, start=start))
            else:
                from alpaca.data.requests import StockBarsRequest
                bs = broker.d.get_stock_bars(StockBarsRequest(
                    symbol_or_symbols=chunk, timeframe=frame, start=start))
        except Exception:
            continue
        data = getattr(bs, "data", {}) or {}
        for sym, rows in data.items():
            df = pd.DataFrame([{"ts": b.timestamp, "open": float(b.open),
                                "high": float(b.high), "low": float(b.low),
                                "close": float(b.close), "volume": float(b.volume)}
                               for b in rows])
            if len(df) < 60:
                continue
            idx = pd.to_datetime(df["ts"]) if intraday else pd.to_datetime(df["ts"].dt.date)
            df = df.set_index(idx).drop(columns="ts")
            out[sym] = df.tail(n)
            try:
                df.tail(n).to_parquet(os.path.join(CACHE, _safe(sym) + "_" + tf + ".parquet"))
            except Exception:
                pass


# ------------------------------------------------------------------- the scan
def scan_universe(bars: dict[str, pd.DataFrame], fresh_bars=5, workers=None):
    """Run the single-instrument strategies over every instrument. Returns
    (setups, features): setups = fresh signals ranked; features = one row per
    instrument for the ML selector."""
    from .strategies import REGISTRY
    specs = [s for s in REGISTRY if s.kind == "single"]
    items = list(bars.items())
    setups, feats = [], []
    for sym, df in items:
        try:
            r = _scan_one(sym, df, specs, fresh_bars)
        except Exception:
            continue
        setups += r["setups"]
        feats.append(r["features"])
    setups.sort(key=lambda x: x.get("hist_exp", -9) if x.get("hist_exp") is not None
                else -9, reverse=True)
    return setups, feats


def _scan_one(sym, df, specs, fresh_bars):
    c = df["close"]
    n = len(c)
    setups = []
    n_signals = 0
    for spec in specs:
        try:
            evs = spec.fn(df, sym, **spec.params)
        except Exception:
            continue
        for e in evs:
            n_signals += 1
            if e.i_signal >= n - fresh_bars:
                setups.append({"asset": sym, "strategy": spec.id,
                               "family": spec.family,
                               "side": "LONG" if e.side > 0 else "SHORT",
                               "signal_date": str(df.index[e.i_signal].date()),
                               "hist_exp": None})
    # per-instrument features for the ML selector (point-in-time as of last bar)
    ret = c.pct_change()
    net = sum(1 if s["side"] == "LONG" else -1 for s in setups)
    feats = {
        "asset": sym, "last": float(c.iloc[-1]),
        "dollar_vol": float((c * df["volume"]).tail(20).mean()) if "volume" in df else 0.0,
        "rvol_20": float(ret.tail(20).std() * np.sqrt(252)) if n > 21 else 0.0,
        "rvol_ratio": float((ret.tail(20).std() / (ret.tail(100).std() + 1e-9)))
        if n > 101 else 1.0,
        "mom_63": float(c.iloc[-1] / c.iloc[-63] - 1) if n > 63 else 0.0,
        "mom_252": float(c.iloc[-1] / c.iloc[-252] - 1) if n > 252 else 0.0,
        "dist_52w_high": float(c.iloc[-1] / c.tail(252).max() - 1) if n > 20 else 0.0,
        "n_fresh_signals": len(setups),
        "n_signals_total": n_signals, "bars": n,
        "fresh": sorted({s["strategy"] for s in setups}),
        "net_side": "LONG" if net > 0 else "SHORT" if net < 0 else "—",
    }
    return {"setups": setups, "features": feats}


# ------------------------------------------------------- morning orchestration
def run_scan(broker, watchlist=(), cap=3000, min_dollar_vol=2e6,
             on_progress=None, tf="1Day") -> dict:
    """Full morning pipeline at any timeframe: enumerate → liquidity-filter →
    (ML narrow if active) → batch-fetch bars at `tf` → scan → log (daily only,
    to keep the ML selector's label horizon consistent). Returns results."""
    from datetime import date as _date
    from . import selector
    t0 = time.time()

    def prog(msg):
        if on_progress:
            on_progress(msg)
    prog("enumerating universe…")
    cands = candidate_symbols(broker, watchlist)
    prog(f"{len(cands)} candidates; liquidity filter…")
    liq = liquid_universe(broker, cands, min_dollar_vol=min_dollar_vol, cap=cap)
    st = selector.status()
    scan_syms = liq
    if st["phase"] == "active":                       # narrow the fetch by ML
        last = selector.last_features(liq)
        if last:
            picks, _ = selector.shortlist([dict(asset=a, **f) for a, f in last.items()])
            scan_syms = [s for s in picks if s in set(liq)] or liq
    n_bars = 400 if tf == "1Day" else 250     # fewer bars per symbol intraday
    prog(f"fetching {tf} bars for {len(scan_syms)} instruments…")
    bars = fetch_bars(broker, scan_syms, n=n_bars, tf=tf)
    prog(f"scanning {len(bars)} instruments ({tf})…")
    setups, feats = scan_universe(bars)
    for s in setups:
        s["tf"] = tf
    labelled = 0
    if tf == "1Day":                                  # ML trains on daily only
        day = _date.today().isoformat()
        selector.log_scan(day, feats, setups)
        labelled = selector.update_labels(bars)
    prog("done")
    return {"asof": time.time(), "tf": tf, "phase": st["phase"], "universe": len(liq),
            "scanned": len(bars), "n_setups": len(setups),
            "setups": setups[:200], "instruments": feats, "labelled_now": labelled,
            "took": round(time.time() - t0, 1), "selector": selector.status()}


# ------------------------------------------------------------------- helpers
def _safe(sym):
    return sym.replace("/", "_")


def _fresh(fp, max_age_h=20):
    return time.time() - os.path.getmtime(fp) < max_age_h * 3600
