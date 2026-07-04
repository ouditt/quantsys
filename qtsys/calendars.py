"""calendars.py — economic + corporate event calendars from free sources.

  economic()  : upcoming key US macro releases (FRED release-dates feed,
                filtered to the market-moving ones).
  corporate() : per-ticker earnings / dividend / ex-dividend dates (yfinance),
                fetched concurrently for a supplied symbol list.

Cached (economic 12h, corporate 6h); every fetch is best-effort.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

_CACHE: dict = {}

# release-name substrings that matter to markets (FRED names)
_KEY_RELEASES = (
    "Consumer Price Index", "Employment Situation", "Gross Domestic Product",
    "FOMC", "Personal Income and Outlays", "Producer Price Index",
    "Retail Sales", "Advance Monthly Sales", "ISM", "Consumer Sentiment",
    "Unemployment Insurance", "Job Openings", "Housing Starts",
    "New Residential", "Durable Goods", "Industrial Production",
    "GDP", "Personal Consumption",
)


def _cached(key, ttl, fn):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    try:
        v = fn()
    except Exception:
        v = hit[1] if hit else []
    _CACHE[key] = (now, v)
    return v


# ------------------------------------------------------------------- economic
def economic(days_ahead: int = 45) -> list[dict]:
    return _cached("econ", 12 * 3600, lambda: _economic(days_ahead))


def _economic(days_ahead: int) -> list[dict]:
    import datetime
    key = os.environ.get("FRED_API_KEY")
    if not key:
        return []
    today = datetime.date.today()
    end = today + datetime.timedelta(days=days_ahead)
    url = ("https://api.stlouisfed.org/fred/releases/dates?"
           f"api_key={key}&file_type=json&include_release_dates_with_no_data=true"
           f"&sort_order=asc&realtime_start={today}&realtime_end={end}&limit=1000")
    with urllib.request.urlopen(url, timeout=20) as r:
        data = json.loads(r.read())
    # FRED's release feed lists a date per day a release *could* update; collapse
    # to the NEXT upcoming date per release name (approx. scheduled date).
    nextd: dict[str, str] = {}
    for row in sorted(data.get("release_dates", []), key=lambda x: x.get("date", "")):
        d = row.get("date", "")
        name = row.get("release_name", "")
        if not d or d < str(today) or d > str(end):
            continue
        if any(k.lower() in name.lower() for k in _KEY_RELEASES) and name not in nextd:
            nextd[name] = d
    return sorted(({"date": d, "name": n} for n, d in nextd.items()),
                  key=lambda x: x["date"])


# ------------------------------------------------------------------ corporate
def corporate(symbols) -> dict:
    symbols = tuple(sorted({s for s in symbols if s and "/" not in s}))[:60]
    return _cached("corp:" + ",".join(symbols), 6 * 3600,
                   lambda: _corporate(symbols))


def _corporate(symbols) -> dict:
    import concurrent.futures
    import datetime
    import yfinance as yf
    today = str(datetime.date.today())

    def one(sym):
        try:
            cal = yf.Ticker(sym).calendar or {}
        except Exception:
            return sym, None
        return sym, cal

    earnings, dividends = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        for sym, cal in ex.map(one, symbols):
            if not cal:
                continue
            ed = cal.get("Earnings Date")
            if isinstance(ed, list) and ed:
                ed = ed[0]
            if ed and str(ed) >= today:
                lo, hi = cal.get("Earnings Low"), cal.get("Earnings High")
                earnings.append({"symbol": sym, "date": str(ed),
                                 "eps_low": lo, "eps_high": hi})
            xd = cal.get("Ex-Dividend Date")
            dd = cal.get("Dividend Date")
            if xd and str(xd) >= today:
                dividends.append({"symbol": sym, "ex_date": str(xd),
                                  "pay_date": str(dd) if dd else None})
    earnings.sort(key=lambda x: x["date"])
    dividends.sort(key=lambda x: x["ex_date"])
    return {"earnings": earnings, "dividends": dividends}
