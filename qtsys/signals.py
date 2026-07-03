"""Look-ahead-safe indicators + primary strategies.

Every value at bar t uses data <= t only (no centered windows, no repainting).
Primary strategies emit *candidate trade events*; whether an event is actually
taken is decided later by the trade-selection layer (select.py), and every
event is executed at the NEXT bar's close (backtest.py) so a signal can never
trade on the bar that produced it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------- indicators
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def realized_vol(close: pd.Series, n: int = 20) -> pd.Series:
    return close.pct_change().rolling(n).std()


def autocorr1(close: pd.Series, n: int = 60) -> pd.Series:
    """Rolling 1-bar autocorrelation: >0 trend-persistent, <0 mean-reverting."""
    r = close.pct_change()
    return r.rolling(n).corr(r.shift(1))


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Event-time features for the trade-selection model (all info <= t)."""
    c = df["close"]
    v20, v100 = realized_vol(c, 20), realized_vol(c, 100)
    f = pd.DataFrame(index=df.index)
    f["trend_z"] = (c / sma(c, 100) - 1) / (v20 * np.sqrt(100))
    f["mom60_z"] = c.pct_change(60) / (v20 * np.sqrt(60))
    f["rsi14"] = rsi(c, 14) / 100.0
    f["vol_ratio"] = v20 / v100            # >1 = volatility expanding (regime proxy)
    f["vol_z"] = (v20 - v20.rolling(250).mean()) / v20.rolling(250).std()
    f["dd_52w"] = c / c.rolling(252).max() - 1
    f["acorr"] = autocorr1(c, 60)
    return f


# --------------------------------------------------------------------- events
@dataclass(frozen=True)
class Event:
    asset: str
    i_signal: int          # integer bar index of the signal (execution is i_signal+1)
    side: int              # +1 long, -1 short
    strategy: str


def momentum_events(df: pd.DataFrame, asset: str,
                    fast: int = 20, slow: int = 100) -> list[Event]:
    """Time-series momentum: enter long on fast/slow SMA cross-up while price is
    above the slow average; short on the mirror condition. Edge thesis:
    behavioral under-reaction makes trends persist (regime-conditional)."""
    c = df["close"]
    f, s = sma(c, fast), sma(c, slow)
    up = (f > s) & (f.shift(1) <= s.shift(1)) & (c > s)
    dn = (f < s) & (f.shift(1) >= s.shift(1)) & (c < s)
    ev = [Event(asset, i, +1, "momentum") for i in np.flatnonzero(up.to_numpy())]
    ev += [Event(asset, i, -1, "momentum") for i in np.flatnonzero(dn.to_numpy())]
    return sorted(ev, key=lambda e: e.i_signal)


def meanrev_events(df: pd.DataFrame, asset: str,
                   rsi_n: int = 2, buy_th: float = 12.0) -> list[Event]:
    """Short-term mean reversion: buy an oversold dip (RSI(2) < th) only while the
    long-term trend is up. Edge thesis: over-reaction on short horizons."""
    c = df["close"]
    cond = (rsi(c, rsi_n) < buy_th) & (c > sma(c, 200))
    cond &= ~cond.shift(1, fill_value=False)          # first bar of each dip only
    return [Event(asset, i, +1, "meanrev") for i in np.flatnonzero(cond.to_numpy())]
