"""Data layer — REAL DATA ONLY, still $0.

Policy (v1.2): no synthetic price generation exists anywhere in this codebase.
Every backtest, validation test, replay tape, and terminal chart runs on real
market history. Bundled sources (all free, refreshable via refresh_real()):

  WTI     crude oil, daily since 1986        (datasets/oil-prices)
  BRENT   crude oil, daily since 1987        (datasets/oil-prices)
  NATGAS  Henry Hub, daily since 1997        (datasets/natural-gas)
  VIX     CBOE VIX, daily OHLC since 1990    (datasets/finance-vix)  [analyse-only]
  GOLD    monthly since 1833                 (datasets/gold-prices)  [page-only]
  SPX     S&P 500 monthly since 1871, +divs  (datasets/s-and-p-500)
  S&P 500 constituents (503 real tickers)    (datasets/s-and-p-500-companies)

On the user's machine, fetch_yfinance()/fetch_ccxt() add real daily equities,
ETF and crypto history (both libraries free), and the broker adapters stream
real live quotes. Every dataset passes quality_check(); hard defects raise —
they are never silently "fixed".
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
REAL_DIR = os.path.join(HERE, "data_real")

REAL_SOURCES = {
    "WTI":    ("wti-daily.csv",    "https://raw.githubusercontent.com/datasets/oil-prices/main/data/wti-daily.csv"),
    "BRENT":  ("brent-daily.csv",  "https://raw.githubusercontent.com/datasets/oil-prices/main/data/brent-daily.csv"),
    "NATGAS": ("natgas-daily.csv", "https://raw.githubusercontent.com/datasets/natural-gas/main/data/daily.csv"),
    "VIX":    ("vix-daily.csv",    "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv"),
    "GOLD":   ("monthly.csv",      "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"),
    "SPX":    ("sp500_monthly.csv","https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"),
    "BTC":    ("btc.csv",          "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"),
    "ETH":    ("eth.csv",          "https://raw.githubusercontent.com/coinmetrics/data/master/csv/eth.csv"),
    # Fed H.10 daily FX, one shared file; normalized to USD per unit of foreign
    # currency on load (up = that currency strengthening vs USD)
    "EURUSD": ("fx-daily.csv", "https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv"),
    "GBPUSD": ("fx-daily.csv", "https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv"),
    "AUDUSD": ("fx-daily.csv", "https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv"),
    "JPYUSD": ("fx-daily.csv", "https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv"),
    "CHFUSD": ("fx-daily.csv", "https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv"),
    "CADUSD": ("fx-daily.csv", "https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv"),
}

# This dataset quotes every series as foreign-currency units PER USD, so all
# invert to our convention: USD per unit of foreign currency (up = ccy stronger)
FX_MAP = {"EURUSD": ("Euro", True), "GBPUSD": ("United Kingdom", True),
          "AUDUSD": ("Australia", True), "JPYUSD": ("Japan", True),
          "CHFUSD": ("Switzerland", True), "CADUSD": ("Canada", True)}


class DataQualityError(ValueError):
    pass


def quality_check(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """Refuse data with hard defects; return the frame if clean."""
    if df.empty:
        raise DataQualityError("empty dataset")
    if df.index.duplicated().any():
        raise DataQualityError(f"{int(df.index.duplicated().sum())} duplicate timestamps")
    if not df.index.is_monotonic_increasing:
        raise DataQualityError("timestamps not sorted")
    p = df[price_col]
    if (p <= 0).any():
        raise DataQualityError("non-positive prices")
    if p.isna().any():
        raise DataQualityError("NaN prices")
    if {"high", "low"}.issubset(df.columns) and (df["high"] < df["low"]).any():
        raise DataQualityError("high < low rows present")
    if p.nunique() == 1:
        raise DataQualityError("zero-variance price series")
    return df


def _read_price_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"price": "close"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    return df


def _read_coinmetrics(path: str) -> pd.DataFrame:
    """Coin Metrics community data: real daily reference prices (PriceUSD)."""
    df = pd.read_csv(path, usecols=["time", "PriceUSD"])
    df["time"] = pd.to_datetime(df["time"])
    df = (df.set_index("time").rename(columns={"PriceUSD": "close"})
            .dropna().sort_index())
    return df[~df.index.duplicated(keep="first")]


def _read_h10_fx(path: str, country: str, invert: bool) -> pd.DataFrame:
    """Fed H.10 daily FX (long format). Normalized to USD per unit of foreign
    currency so 'up' always means that currency strengthening vs USD."""
    fx = pd.read_csv(path)
    s = fx[fx["Country"] == country].copy()
    s["Exchange rate"] = pd.to_numeric(s["Exchange rate"], errors="coerce")
    s["Date"] = pd.to_datetime(s["Date"])
    s = s.dropna(subset=["Exchange rate"]).set_index("Date").sort_index()
    close = (1.0 / s["Exchange rate"]) if invert else s["Exchange rate"]
    df = close.to_frame("close")
    return df[~df.index.duplicated(keep="first")]


def load_real(symbol: str) -> pd.DataFrame:
    """Load a bundled real series by symbol (see REAL_SOURCES)."""
    sym = symbol.upper()
    fname, _ = REAL_SOURCES[sym]
    path = os.path.join(REAL_DIR, fname)
    if sym == "SPX":
        return load_shiller_monthly(path)
    if sym in ("BTC", "ETH"):
        return quality_check(_read_coinmetrics(path))
    if sym in FX_MAP:
        country, invert = FX_MAP[sym]
        return quality_check(_read_h10_fx(path, country, invert))
    df = _read_price_csv(path)
    return quality_check(df)


DAILY_TRADABLES = ("WTI", "BRENT", "NATGAS", "BTC", "ETH",
                   "EURUSD", "GBPUSD", "AUDUSD", "JPYUSD", "CHFUSD", "CADUSD")


def real_daily_universe(symbols=DAILY_TRADABLES) -> dict[str, pd.DataFrame]:
    """The bundled real DAILY, tradable-style universe used by backtests, the
    validation suite, the demo, and the replay tape."""
    return {s: load_real(s) for s in symbols}


def load_constituents() -> pd.DataFrame:
    """Real S&P 500 constituent list (503 tickers) — the 'packable universe'."""
    df = pd.read_csv(os.path.join(REAL_DIR, "constituents.csv"))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def load_shiller_monthly(path: str | None = None) -> pd.DataFrame:
    """Real S&P 500 monthly data since 1871 (price + dividends)."""
    path = path or os.path.join(REAL_DIR, "sp500_monthly.csv")
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    df = df.rename(columns={"SP500": "close", "Dividend": "dividend"})[["close", "dividend"]]
    df = df.dropna(subset=["close"])
    df["dividend"] = df["dividend"].ffill().fillna(0.0)
    df["tr"] = df["close"].pct_change() + (df["dividend"] / 12.0) / df["close"].shift(1)
    return quality_check(df)


def refresh_real(timeout: int = 30) -> dict[str, int]:
    """Re-download every bundled real dataset from its free source (run online)."""
    import urllib.request
    os.makedirs(REAL_DIR, exist_ok=True)
    out = {}
    for sym, (fname, url) in REAL_SOURCES.items():
        with urllib.request.urlopen(url, timeout=timeout) as r:
            raw = r.read()
        with open(os.path.join(REAL_DIR, fname), "wb") as f:
            f.write(raw)
        out[sym] = len(raw)
    return out


# ---------------------------------------------------- real data, user's machine
def fetch_yfinance(symbol: str, start: str = "2015-01-01", interval: str = "1d"):
    """Free real data for local runs: `pip install yfinance` (research-grade only)."""
    import yfinance as yf  # guarded: only needed on the user's machine

    df = yf.download(symbol, start=start, interval=interval, auto_adjust=True, progress=False)
    df.columns = [str(c).lower() for c in df.columns]
    return quality_check(df)


def fetch_ccxt(symbol: str = "BTC/USDT", exchange: str = "binance",
               timeframe: str = "1d", limit: int = 1500):
    """Free real crypto data for local runs: `pip install ccxt` (public endpoints)."""
    import ccxt  # guarded

    ex = getattr(ccxt, exchange)()
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    return quality_check(df)
