"""Broker-grade data adapters: Alpaca and Interactive Brokers, plus a unified
fetch() dispatcher across every source. Strategy code never changes when the
venue does — the blueprint's "one fetch() interface" rule.

Free-tier notes (verify current plans at runtime — providers drift):
  * ALPACA — a free paper account gives free API keys; historical STOCK bars on
    the free plan use the IEX feed (pass feed="iex"; "sip" needs a paid plan).
    CRYPTO bars need NO keys at all.
  * IBKR — a free paper account works; requires Trader Workstation or IB
    Gateway running locally (paper ports: TWS 7497, Gateway 4002). Without a
    market-data subscription, request delayed data (market_data_type=3).

Each adapter is split into a thin network call + a PURE normalizer, so the
normalizers are unit-tested offline at $0 (validate.py, T7) against the exact
dataframe shapes the installed SDKs produce. Everything returns the same
UTC-indexed open/high/low/close/volume frame and must pass quality_check().
"""
from __future__ import annotations

import os

import pandas as pd

from .data import fetch_ccxt, fetch_yfinance, quality_check

_STD = ["open", "high", "low", "close", "volume"]


# ------------------------------------------------------------------ ALPACA
def _alpaca_timeframe(tf: str):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    tf = tf.lower()
    table = {"1d": TimeFrame.Day, "1h": TimeFrame.Hour, "1m": TimeFrame.Minute,
             "1w": TimeFrame.Week}
    if tf in table:
        return table[tf]
    if tf.endswith("m"):
        return TimeFrame(int(tf[:-1]), TimeFrameUnit.Minute)
    if tf.endswith("h"):
        return TimeFrame(int(tf[:-1]), TimeFrameUnit.Hour)
    raise ValueError(f"unsupported timeframe: {tf}")


def normalize_alpaca_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Pure: alpaca-py's MultiIndex (symbol, timestamp) bars -> standard frame."""
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")
    df = df[_STD].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "ts"
    return quality_check(df.sort_index())


def fetch_alpaca(symbol: str, start: str = "2018-01-01", end: str | None = None,
                 timeframe: str = "1d", asset_class: str = "stock",
                 api_key: str | None = None, api_secret: str | None = None,
                 feed: str = "iex") -> pd.DataFrame:
    """Free real data via Alpaca. Stocks need free paper-account keys
    (env: APCA_API_KEY_ID / APCA_API_SECRET_KEY, or ALPACA_API_KEY /
    ALPACA_SECRET_KEY); crypto needs none. feed='iex' is the free stock feed."""
    api_key = api_key or os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    api_secret = (api_secret or os.getenv("APCA_API_SECRET_KEY")
                  or os.getenv("ALPACA_SECRET_KEY"))
    tf = _alpaca_timeframe(timeframe)
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end) if end else None
    if asset_class == "crypto":
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest

        client = CryptoHistoricalDataClient()          # keyless
        req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=tf,
                                start=start_ts, end=end_ts)
        raw = client.get_crypto_bars(req).df
    else:
        if not (api_key and api_secret):
            raise RuntimeError("Alpaca stock data needs free paper-account keys; "
                               "set APCA_API_KEY_ID / APCA_API_SECRET_KEY")
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest

        client = StockHistoricalDataClient(api_key, api_secret)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf,
                               start=start_ts, end=end_ts, feed=feed)
        raw = client.get_stock_bars(req).df
    return normalize_alpaca_df(raw, symbol)


# -------------------------------------------------------------------- IBKR
def _ib():
    """Prefer the maintained fork ib_async; fall back to ib_insync (API-compatible)."""
    try:
        import ib_async as ib
    except ImportError:
        import ib_insync as ib
    return ib


def _ib_duration(start: str) -> str:
    days = max((pd.Timestamp.now(tz="UTC") - pd.Timestamp(start, tz="UTC")).days, 1)
    return f"{-(-days // 365)} Y" if days > 365 else f"{days} D"


def normalize_ib_df(df: pd.DataFrame) -> pd.DataFrame:
    """Pure: util.df(bars) output (date/open/high/low/close/volume/...) -> standard."""
    df = df.rename(columns={"date": "ts"}).set_index("ts")[_STD].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    return quality_check(df.sort_index())


def fetch_ibkr(symbol: str, start: str = "2018-01-01", bar_size: str = "1 day",
               sec_type: str = "STK", exchange: str = "SMART", currency: str = "USD",
               host: str = "127.0.0.1", port: int = 7497, client_id: int = 17,
               what_to_show: str | None = None, use_rth: bool = True,
               market_data_type: int = 3) -> pd.DataFrame:
    """Historical bars from a locally running TWS / IB Gateway (paper acct free;
    port 7497 = paper TWS, 4002 = paper Gateway). market_data_type=3 requests
    delayed data so no paid subscription is required."""
    ib_mod = _ib()
    contract = {"STK": ib_mod.Stock(symbol, exchange, currency),
                "CASH": ib_mod.Forex(symbol),
                "FUT": ib_mod.Future(symbol, exchange=exchange),
                "CRYPTO": ib_mod.Crypto(symbol, "PAXOS", currency)}[sec_type]
    show = what_to_show or ("MIDPOINT" if sec_type == "CASH" else "ADJUSTED_LAST")
    ib = ib_mod.IB()
    ib.connect(host, port, clientId=client_id, timeout=10)
    try:
        ib.reqMarketDataType(market_data_type)
        bars = ib.reqHistoricalData(contract, endDateTime="",
                                    durationStr=_ib_duration(start),
                                    barSizeSetting=bar_size, whatToShow=show,
                                    useRTH=use_rth, formatDate=2)
        return normalize_ib_df(ib_mod.util.df(bars))
    finally:
        ib.disconnect()


# --------------------------------------------------------- unified dispatcher
SOURCES = {
    "yfinance": lambda symbol, **kw: fetch_yfinance(symbol, **kw),
    "ccxt": lambda symbol, **kw: fetch_ccxt(symbol, **kw),
    "alpaca": lambda symbol, **kw: fetch_alpaca(symbol, **kw),
    "ibkr": lambda symbol, **kw: fetch_ibkr(symbol, **kw),
}


def fetch(source: str, symbol: str, **kw) -> pd.DataFrame:
    """One interface, every venue: fetch('alpaca', 'SPY'), fetch('ibkr', 'EURUSD',
    sec_type='CASH'), fetch('ccxt', 'BTC/USDT'), fetch('yfinance', 'GC=F')...
    Output is always the same QC-passed OHLCV frame."""
    if source not in SOURCES:
        raise ValueError(f"unknown source '{source}'; options: {sorted(SOURCES)}")
    return SOURCES[source](symbol, **kw)
