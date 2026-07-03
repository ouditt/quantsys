"""T7 — broker adapters verified OFFLINE against the installed SDKs (no
network, no keys, no price data): request builders and normalizers are checked
against alpaca-py and ib_async types directly."""
from __future__ import annotations

import numpy as np
import pandas as pd



def t7_broker_adapters_offline():
    """Verify the Alpaca/IBKR adapters against the REAL installed SDKs with zero
    network and zero keys: request/contract objects must construct, and the pure
    normalizers must turn each SDK's native bar shape into the standard
    QC-passed frame. Skips gracefully if an SDK isn't installed."""
    from .adapters import (SOURCES, _alpaca_timeframe, _ib, _ib_duration, fetch,
                           normalize_alpaca_df, normalize_ib_df)

    assert set(SOURCES) == {"yfinance", "ccxt", "alpaca", "ibkr"}
    try:
        fetch("nope", "SPY")
        raise AssertionError("dispatcher accepted an unknown source")
    except ValueError:
        pass

    checks = []
    # ---- Alpaca: real SDK objects, mocked bar payload in the SDK's own shape
    try:
        from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest

        tf = _alpaca_timeframe("1d")
        StockBarsRequest(symbol_or_symbols="SPY", timeframe=tf,
                         start=pd.Timestamp("2024-01-01"), feed="iex")
        CryptoBarsRequest(symbol_or_symbols="BTC/USD", timeframe=_alpaca_timeframe("1h"),
                          start=pd.Timestamp("2024-01-01"))
        ts = pd.date_range("2024-01-02", periods=6, freq="D", tz="UTC")
        raw = pd.DataFrame(
            {"open": 100.0, "high": 101.0, "low": 99.0,
             "close": [100.2, 100.4, 100.1, 100.9, 101.2, 100.7],
             "volume": 5e5, "trade_count": 1000, "vwap": 100.3},
            index=pd.MultiIndex.from_product([["SPY"], ts],
                                             names=["symbol", "timestamp"]))
        norm = normalize_alpaca_df(raw, "SPY")
        assert list(norm.columns) == ["open", "high", "low", "close", "volume"]
        assert str(norm.index.tz) == "UTC"
        checks.append("alpaca(iex/crypto) requests + normalizer OK")
    except ImportError:
        checks.append("alpaca-py not installed — skipped")

    # ---- IBKR: real contract objects + util.df(BarData) through the normalizer
    try:
        ib_mod = _ib()
        ib_mod.Stock("SPY", "SMART", "USD"); ib_mod.Forex("EURUSD")
        assert _ib_duration("2024-01-01").endswith(("Y", "D"))
        BarData = ib_mod.util.dataclassAsDict and ib_mod.BarData  # attr exists
        bars = [BarData(date=pd.Timestamp("2024-01-02", tz="UTC") + pd.Timedelta(days=i),
                        open=100, high=101, low=99, close=100 + 0.1 * i,
                        volume=1e5, average=100.1, barCount=500) for i in range(6)]
        norm = normalize_ib_df(ib_mod.util.df(bars))
        assert list(norm.columns) == ["open", "high", "low", "close", "volume"]
        assert str(norm.index.tz) == "UTC"
        checks.append(f"ibkr ({ib_mod.__name__}) contracts + normalizer OK")
    except ImportError:
        checks.append("ib_async/ib_insync not installed — skipped")

    print("T7 PASS  broker adapters verified offline against installed SDKs: "
          + "; ".join(checks))



def run():
    t7_broker_adapters_offline()
