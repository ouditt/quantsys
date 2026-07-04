"""intel.py — per-instrument fundamentals + multi-source news.

Turns raw fundamentals and headlines into NORMALISED records the agents (and the
terminal) can actually reason over:
  - `fundamentals(sym, cls)` -> {metrics{}, rows[], brief} standardised per asset
    class (equities/crypto via yfinance; FX/commodities via FRED macro).
  - `news(sym, cls)` -> deduped, sentiment-tagged headlines from yfinance (Yahoo)
    to merge with the venue's own feed.

Everything is cached with sane TTLs (fundamentals hours, news minutes) and every
network call is best-effort — a failure returns the last good value or an empty
record, never an exception into the agent loop.

IMPORTANT: these are LIVE (as-of-now) fundamentals. They are safe for agent
commentary and the UI, but must NOT be fed into a backtest — using today's P/E
on old prices is look-ahead. Point-in-time history is a separate, later job.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

from .sentiment import score as _sent

_CACHE: dict[str, tuple[float, object]] = {}
FUND_TTL = 6 * 3600          # fundamentals refresh a few times a day
NEWS_TTL = 300               # yfinance news every 5 min
MACRO_TTL = 12 * 3600


def _cached(key: str, ttl: int, fn):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    try:
        val = fn()
    except Exception:
        val = hit[1] if hit else None
    _CACHE[key] = (now, val)
    return val


# ----------------------------------------------------------- symbol mapping
def _yf_symbol(sym: str, cls: str) -> str:
    if cls == "Crypto":
        return sym.split("/")[0] + "-USD"
    if cls == "FX":
        return (sym[:6] if len(sym) >= 6 else sym) + "=X"
    if cls == "Commodity":
        return {"WTI": "CL=F", "BRENT": "BZ=F", "NATGAS": "NG=F"}.get(sym, sym)
    return sym                                   # equity / ETF


def _pct(x):
    return None if x is None else round(float(x) * 100, 2)


# ----------------------------------------------------------------- fundamentals
def fundamentals(sym: str, cls: str) -> dict:
    return _cached(f"fund:{sym}", FUND_TTL, lambda: _fundamentals(sym, cls)) or \
        {"metrics": {}, "rows": [], "brief": "no fundamentals available"}


def _fundamentals(sym: str, cls: str) -> dict:
    if cls in ("FX", "Commodity", "Index — analyse-only", "Monthly — page-only"):
        return _macro_fundamentals(sym, cls)
    import yfinance as yf
    info = yf.Ticker(_yf_symbol(sym, cls)).info or {}
    g = info.get
    if cls == "Crypto":
        m = {"market_cap": g("marketCap"), "volume_24h": g("volume24Hr") or g("volume"),
             "circulating_supply": g("circulatingSupply"),
             "52w_change_pct": _pct(g("52WeekChange")),
             "vs_50d_ma": g("regularMarketPrice") and g("fiftyDayAverage")
             and _pct(g("regularMarketPrice") / g("fiftyDayAverage") - 1)}
        rows = [("Market cap", _money(m["market_cap"])),
                ("24h volume", _money(m["volume_24h"])),
                ("Circulating supply", _num(m["circulating_supply"])),
                ("52w change", _pcts(m["52w_change_pct"])),
                ("Price vs 50d MA", _pcts(m["vs_50d_ma"]))]
        brief = f"{sym}: mcap {_money(m['market_cap'])}, 52w {_pcts(m['52w_change_pct'])}"
        return {"metrics": m, "rows": rows, "brief": brief, "cls": cls}
    # equity / ETF
    m = {"pe": _r(g("trailingPE")), "forward_pe": _r(g("forwardPE")),
         "peg": _r(g("pegRatio")), "eps": _r(g("trailingEps")),
         "rev_growth_pct": _pct(g("revenueGrowth")),
         "earnings_growth_pct": _pct(g("earningsGrowth")),
         "profit_margin_pct": _pct(g("profitMargins")),
         "debt_to_equity": _r(g("debtToEquity")), "beta": _r(g("beta")),
         "market_cap": g("marketCap"), "div_yield_pct": _r(g("dividendYield")),
         "target_mean": _r(g("targetMeanPrice")),
         "analyst": g("recommendationKey"),
         "sector": g("sector"), "industry": g("industry")}
    rows = [("Sector", m["sector"] or "—"), ("Industry", m["industry"] or "—"),
            ("Market cap", _money(m["market_cap"])), ("P/E (ttm)", _num(m["pe"])),
            ("Forward P/E", _num(m["forward_pe"])), ("PEG", _num(m["peg"])),
            ("EPS (ttm)", _num(m["eps"])),
            ("Revenue growth", _pcts(m["rev_growth_pct"])),
            ("Earnings growth", _pcts(m["earnings_growth_pct"])),
            ("Profit margin", _pcts(m["profit_margin_pct"])),
            ("Debt/Equity", _num(m["debt_to_equity"])), ("Beta", _num(m["beta"])),
            ("Div yield", _pcts(m["div_yield_pct"])),
            ("Analyst target", _num(m["target_mean"])),
            ("Analyst view", (m["analyst"] or "—").upper())]
    brief = (f"{sym}: {m['sector'] or '?'}, P/E {_num(m['pe'])}, rev growth "
             f"{_pcts(m['rev_growth_pct'])}, margin {_pcts(m['profit_margin_pct'])}, "
             f"analysts {m['analyst'] or 'n/a'}")
    return {"metrics": m, "rows": rows, "brief": brief, "cls": cls}


def _macro_fundamentals(sym: str, cls: str) -> dict:
    """FX/commodity 'fundamentals' are macro drivers, not a balance sheet."""
    macro = _cached("macro", MACRO_TTL, _fred_macro) or {}
    rows = [(k, v) for k, v in macro.get("rows", [])]
    drivers = {"WTI": "EIA crude inventories & OPEC supply",
               "BRENT": "global crude supply/demand, OPEC",
               "NATGAS": "EIA storage, weather, LNG exports"}.get(sym)
    if cls == "FX":
        drivers = "rate differential vs USD, risk sentiment, terms of trade"
    if drivers:
        rows = [("Primary drivers", drivers)] + rows
    brief = f"{sym}: macro-driven ({drivers or 'macro'}); " + macro.get("brief", "")
    return {"metrics": macro.get("metrics", {}), "rows": rows, "brief": brief,
            "cls": cls}


def _fred_macro() -> dict:
    """A small shared macro snapshot from FRED (free CSV; key optional)."""
    series = {"Fed funds rate": "FEDFUNDS", "10y Treasury yield": "DGS10",
              "10y real yield": "DFII10", "US Dollar index": "DTWEXBGS"}
    metrics, rows = {}, []
    for label, sid in series.items():
        v = _fred_latest(sid)
        metrics[sid] = v
        rows.append((label, _num(v) + ("%" if "yield" in label or "rate" in label else "")))
    brief = f"Fed funds {_num(metrics.get('FEDFUNDS'))}%, 10y {_num(metrics.get('DGS10'))}%"
    return {"metrics": metrics, "rows": rows, "brief": brief}


def _fred_latest(series_id: str):
    key = os.environ.get("FRED_API_KEY")
    if key:
        url = (f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}"
               f"&api_key={key}&file_type=json&sort_order=desc&limit=1")
        with urllib.request.urlopen(url, timeout=15) as r:
            obs = json.loads(r.read())["observations"]
        return float(obs[0]["value"]) if obs and obs[0]["value"] != "." else None
    # keyless fallback: FRED's public CSV endpoint
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    with urllib.request.urlopen(url, timeout=15) as r:
        lines = [x for x in r.read().decode().splitlines() if "," in x]
    for row in reversed(lines[1:]):
        val = row.split(",")[-1].strip()
        if val and val != ".":
            return float(val)
    return None


# ------------------------------------------------------------------------ news
def news(sym: str, cls: str) -> list[dict]:
    return _cached(f"news:{sym}", NEWS_TTL, lambda: _yf_news(sym, cls)) or []


def _yf_news(sym: str, cls: str) -> list[dict]:
    import yfinance as yf
    raw = yf.Ticker(_yf_symbol(sym, cls)).news or []
    out = []
    for a in raw:
        c = a.get("content") if isinstance(a.get("content"), dict) else a
        title = c.get("title") or a.get("title") or ""
        if not title:
            continue
        prov = c.get("provider") if isinstance(c.get("provider"), dict) else None
        source = (prov or {}).get("displayName") or a.get("publisher") or "Yahoo"
        url = ((c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict)
               else a.get("link")) or ""
        ts = c.get("pubDate") or ""
        if not ts and a.get("providerPublishTime"):
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                               time.gmtime(a["providerPublishTime"]))
        summ = c.get("summary") or ""
        lbl, net = _sent(title + " " + summ)
        out.append({"ts": ts, "headline": title, "source": source, "url": url,
                    "summary": summ[:280], "sentiment": lbl, "sent_score": net})
    return out


# ------------------------------------------------------------------- formatting
def _r(x, d=2):
    try:
        return round(float(x), d)
    except (TypeError, ValueError):
        return None


def _num(x):
    return "—" if x is None else (f"{x:,.2f}" if isinstance(x, float) else str(x))


def _pcts(x):
    return "—" if x is None else f"{x:+.2f}%"


def _money(x):
    if x is None:
        return "—"
    x = float(x)
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= div:
            return f"${x / div:.2f}{unit}"
    return f"${x:.0f}"
