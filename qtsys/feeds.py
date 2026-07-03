"""feeds.py — one `get_history()` / `get_quote()` for every blueprint source.

Free tier first: yfinance and ccxt need no keys. Alpaca's data API is free with
a (free) paper account. The paid sources (Polygon, Databento) are wired behind
the same interface so upgrading later changes ONE string, not the codebase.
All imports guarded — each adapter names its `pip install` when first used.
"""
from __future__ import annotations

import pandas as pd

from .data import quality_check


def get_history(source: str, symbol: str, start: str = "2018-01-01",
                interval: str = "1d", **kw) -> pd.DataFrame:
    s = source.lower()
    if s == "yfinance":
        from .data import fetch_yfinance
        return fetch_yfinance(symbol, start, interval)
    if s == "ccxt":
        from .data import fetch_ccxt
        return fetch_ccxt(symbol, kw.get("exchange", "binance"),
                          {"1d": "1d", "1h": "1h"}.get(interval, "1d"),
                          kw.get("limit", 1500))
    if s == "alpaca":                      # free with a free paper account
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        c = StockHistoricalDataClient(kw["api_key"], kw["secret"])
        tf = TimeFrame.Day if interval == "1d" else TimeFrame.Hour
        bars = c.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbol,
                                                 timeframe=tf, start=start)).df
        df = bars.reset_index().set_index("timestamp")[
            ["open", "high", "low", "close", "volume"]]
        return quality_check(df)
    if s == "ibkr":
        from ib_insync import IB, Stock, util
        ib = IB(); ib.connect(kw.get("host", "127.0.0.1"), kw.get("port", 7497),
                              clientId=kw.get("client_id", 9))
        bars = ib.reqHistoricalData(Stock(symbol, "SMART", "USD"), "",
                                    kw.get("duration", "5 Y"),
                                    "1 day" if interval == "1d" else "1 hour",
                                    "TRADES", useRTH=True)
        df = util.df(bars).set_index("date")[["open", "high", "low",
                                              "close", "volume"]]
        return quality_check(df)
    if s == "polygon":                     # paid upgrade, same interface
        from polygon import RESTClient
        c = RESTClient(kw["api_key"])
        aggs = c.get_aggs(symbol, 1, "day", start, kw.get("end", "2100-01-01"))
        df = pd.DataFrame([{"t": a.timestamp, "open": a.open, "high": a.high,
                            "low": a.low, "close": a.close, "volume": a.volume}
                           for a in aggs])
        df.index = pd.to_datetime(df.pop("t"), unit="ms", utc=True)
        return quality_check(df)
    if s == "databento":                   # paid upgrade, same interface
        import databento as db
        c = db.Historical(kw["api_key"])
        df = c.timeseries.get_range(dataset=kw.get("dataset", "XNAS.ITCH"),
                                    symbols=[symbol], schema="ohlcv-1d",
                                    start=start).to_df()
        return quality_check(df.rename(columns=str.lower))
    if s == "oanda":
        import oandapyV20
        import oandapyV20.endpoints.instruments as instruments
        api = oandapyV20.API(access_token=kw["token"],
                             environment=kw.get("env", "practice"))
        r = instruments.InstrumentsCandles(
            symbol, params={"granularity": "D", "count": kw.get("limit", 1500)})
        rows = [{"t": c_["time"], "open": float(c_["mid"]["o"]),
                 "high": float(c_["mid"]["h"]), "low": float(c_["mid"]["l"]),
                 "close": float(c_["mid"]["c"]), "volume": c_["volume"]}
                for c_ in api.request(r)["candles"] if c_["complete"]]
        df = pd.DataFrame(rows); df.index = pd.to_datetime(df.pop("t"))
        return quality_check(df)
    if s == "fred":                        # free macro series
        url = ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + symbol)
        df = pd.read_csv(url, parse_dates=[0], index_col=0)
        df.columns = ["close"]
        return quality_check(df.dropna())
    if s == "edgar":                       # free point-in-time fundamentals
        import requests
        cik = str(kw["cik"]).zfill(10)
        r = requests.get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
            headers={"User-Agent": kw.get("user_agent", "qtsys research")})
        return r.json()                    # raw facts; caller shapes them
    raise ValueError(f"unknown source '{source}' — options: yfinance, ccxt, "
                     "alpaca, ibkr, polygon, databento, oanda, fred, edgar")
