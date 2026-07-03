"""Strategy library — 15 families, every one specified as pure functions that
emit candidate Events for the shared triple-barrier simulator. REAL DATA ONLY:
each function is arithmetic on real recorded closes (ratios/spreads of two real
series are transformations of real data, like an SMA — nothing is generated).

Timeframe-agnostic by construction: every window is in BARS, every threshold is
vol-scaled per bar. Feed 1-minute bars locally and the same code runs; the
bundled verification uses real daily history.

All signals at bar t use data <= t; execution is always t+1 (backtest.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import BarrierSpec, FeeModel
from .signals import Event, sma, rsi, realized_vol, momentum_events, meanrev_events


# ------------------------------------------------------------------ helpers
def _ev(name, idx, side, strategy="lib"):
    return [Event(name, int(i), int(s), strategy) for i, s in zip(idx, side)]


def ratio_frame(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Spread/pairs series: ratio of two REAL price series on their common
    calendar. A derived real series (long numerator / short denominator)."""
    j = a[["close"]].join(b[["close"]], how="inner", lsuffix="_a", rsuffix="_b")
    out = pd.DataFrame(index=j.index)
    out["close"] = j["close_a"] / j["close_b"]
    return out


def vix_percentile(vix: pd.DataFrame, window: int = 252) -> pd.Series:
    v = vix["close"].to_numpy()
    out = np.full(len(v), np.nan)
    for i in range(window, len(v)):
        w = v[i - window + 1:i + 1]
        out[i] = (w <= v[i]).mean()
    return pd.Series(out, index=vix.index)


def gate_by_vix(events: list[Event], df: pd.DataFrame, vix_pct: pd.Series,
                max_pct: float | None = None, min_pct: float | None = None):
    """Keep events only when the as-of VIX percentile satisfies the gate."""
    aligned = vix_pct.reindex(df.index, method="ffill")
    keep = []
    for e in events:
        p = aligned.iloc[e.i_signal]
        if np.isnan(p):
            continue
        if max_pct is not None and p > max_pct:
            continue
        if min_pct is not None and p < min_pct:
            continue
        keep.append(e)
    return keep


# ------------------------------------------------------------------ families
def boll_squeeze_breakout(df, name, n=20, k=2.0, sq_win=120, sq_q=0.25):
    """Bollinger squeeze -> breakout: enter with the break after compression."""
    c = df["close"]; m = sma(c, n); sd = c.rolling(n).std()
    upper, lower = m + k * sd, m - k * sd
    bw = (upper - lower) / m
    squeezed = bw < bw.rolling(sq_win).quantile(sq_q)
    lo = squeezed.shift(1) & (c > upper)
    sh = squeezed.shift(1) & (c < lower)
    idx = np.where(lo | sh)[0]
    return _ev(name, idx, np.where(lo.to_numpy()[idx], 1, -1))


def boll_fade(df, name, n=20, k=2.2, trend_n=200):
    """Band fade (mean reversion) taken only WITH the long-run trend."""
    c = df["close"]; m = sma(c, n); sd = c.rolling(n).std(); t = sma(c, trend_n)
    lo = (c < m - k * sd) & (c > t)
    sh = (c > m + k * sd) & (c < t)
    idx = np.where(lo | sh)[0]
    return _ev(name, idx, np.where(lo.to_numpy()[idx], 1, -1))


def macd_reversion(df, name, fast=12, slow=26, sig=9, z=1.5, zwin=250):
    """'Momentum flips first': stretched MACD histogram that has just turned."""
    c = df["close"]
    macd = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    hist = macd - macd.ewm(span=sig, adjust=False).mean()
    hz = (hist - hist.rolling(zwin).mean()) / hist.rolling(zwin).std()
    turn_up = (hist > hist.shift(1)) & (hist.shift(1) <= hist.shift(2))
    turn_dn = (hist < hist.shift(1)) & (hist.shift(1) >= hist.shift(2))
    lo = (hz < -z) & turn_up
    sh = (hz > z) & turn_dn
    idx = np.where(lo | sh)[0]
    return _ev(name, idx, np.where(lo.to_numpy()[idx], 1, -1))


def donchian_breakout(df, name, n=55):
    """Turtle-style channel breakout on close extremes (close-only honest)."""
    c = df["close"]
    hi = c.rolling(n).max().shift(1); lo_ = c.rolling(n).min().shift(1)
    lo = c > hi; sh = c < lo_
    idx = np.where(lo | sh)[0]
    return _ev(name, idx, np.where(lo.to_numpy()[idx], 1, -1))


def rolling_high_momentum(df, name, n=252, near=0.02, mom_n=60):
    """Long-only strength: near the rolling high with positive medium momentum."""
    c = df["close"]
    cond = (c / c.rolling(n).max() - 1 > -near) & (c.pct_change(mom_n) > 0)
    fresh = cond & ~cond.shift(1).fillna(False)
    idx = np.where(fresh)[0]
    return _ev(name, idx, np.ones(len(idx)))


def tsmom(df, name, look=252, skip=21, every=21):
    """Time-series momentum (12-1 style): sign of look-back return, refreshed
    on a fixed cadence in bars."""
    c = df["close"]
    r = c.shift(skip) / c.shift(look) - 1
    idx = [i for i in range(look + skip, len(c), every) if not np.isnan(r.iloc[i]) and r.iloc[i] != 0]
    return _ev(name, idx, [1 if r.iloc[i] > 0 else -1 for i in idx])


def vol_breakout(df, name, k=2.0, vol_n=20, trend_f=20, trend_s=100):
    """Volatility expansion in the direction of the prevailing trend."""
    c = df["close"]; r = c.pct_change()
    v = realized_vol(c, vol_n)
    trend = sma(c, trend_f) > sma(c, trend_s)
    lo = (r > k * v.shift(1)) & trend
    sh = (r < -k * v.shift(1)) & ~trend
    idx = np.where(lo | sh)[0]
    return _ev(name, idx, np.where(lo.to_numpy()[idx], 1, -1))


def big_move_fade(df, name, k=2.5, vol_n=20, trend_n=200):
    """Fade an outsized one-bar move, longs only above the long trend,
    shorts only below it (never catch a falling knife against regime)."""
    c = df["close"]; r = c.pct_change(); v = realized_vol(c, vol_n)
    t = sma(c, trend_n)
    lo = (r < -k * v.shift(1)) & (c > t)
    sh = (r > k * v.shift(1)) & (c < t)
    idx = np.where(lo | sh)[0]
    return _ev(name, idx, np.where(lo.to_numpy()[idx], 1, -1))


def pair_spread_z(df_ratio, name, n=60, z=2.0):
    """Stat-arb on a real ratio series: fade z-score extremes of log-ratio."""
    lr = np.log(df_ratio["close"])
    zz = (lr - lr.rolling(n).mean()) / lr.rolling(n).std()
    lo = (zz < -z) & (zz.shift(1) >= -z)
    sh = (zz > z) & (zz.shift(1) <= z)
    idx = np.where(lo | sh)[0]
    return _ev(name, idx, np.where(lo.to_numpy()[idx], 1, -1))


def fx_xsec_momentum(fx_universe: dict[str, pd.DataFrame], look=63, every=21,
                     top=2) -> dict[str, list[Event]]:
    """Cross-sectional FX momentum: every `every` bars rank the pairs on
    look-back return; long the top, short the bottom. Multi-asset by nature."""
    common = None
    for df in fx_universe.values():
        common = df.index if common is None else common.intersection(df.index)
    px = pd.DataFrame({s: df["close"].reindex(common) for s, df in fx_universe.items()})
    rets = px.pct_change(look)
    out = {s: [] for s in fx_universe}
    pos_of = {s: {d: i for i, d in enumerate(fx_universe[s].index)} for s in fx_universe}
    for i in range(look, len(common), every):
        row = rets.iloc[i].dropna()
        if len(row) < top * 2:
            continue
        ranked = row.sort_values()
        d = common[i]
        for s in ranked.index[-top:]:
            out[s].append(Event(s, pos_of[s][d], +1, "fx_xsec"))
        for s in ranked.index[:top]:
            out[s].append(Event(s, pos_of[s][d], -1, "fx_xsec"))
    return out


def turn_of_month(df, name, before=1, after=3):
    """Calendar effect: long from the last `before` sessions of the month
    through the first `after` of the next (documented in equities/commodities)."""
    dts = df.index
    idx = []
    for i in range(1, len(dts) - 1):
        if dts[i].month != dts[i + 1].month:          # last session of month
            j = max(0, i - before + 1)
            idx.append(j)
    return _ev(name, idx, np.ones(len(idx)))


def vix_contrarian_entries(df, name, vix_pct, min_pct=0.90):
    """'Trade the crowd', honestly: extreme recorded fear (VIX percentile) as a
    long entry in risk assets — sentiment from a real, tradable-adjacent gauge."""
    aligned = vix_pct.reindex(df.index, method="ffill")
    cond = (aligned > min_pct) & (aligned.shift(1) <= min_pct)
    idx = np.where(cond.fillna(False))[0]
    return _ev(name, idx, np.ones(len(idx)))


# ------------------------------------------------------------------ registry
@dataclass(frozen=True)
class Spec:
    id: str
    family: str
    kind: str                       # single | pair | xsec | vixgate | vixcontra
    fn: object
    params: dict = field(default_factory=dict)
    barriers: BarrierSpec = BarrierSpec(2.0, 1.5, 15)
    assets: tuple = ()              # empty = all daily tradables
    fee_mult: float = 1.0           # pairs pay both legs
    note: str = ""


MR = BarrierSpec(1.2, 1.5, 7)       # mean-reversion style exits
TR = BarrierSpec(2.0, 1.5, 15)      # trend style exits
LG = BarrierSpec(3.0, 2.0, 40)      # long-horizon exits

REGISTRY: list[Spec] = [
    Spec("mom_sma_20_100",  "MA cross",        "single", momentum_events, {}, TR, note="ride the trend"),
    Spec("meanrev_rsi2",    "RSI reversion",   "single", meanrev_events,  {}, MR, note="buy fear in uptrends"),
    Spec("squeeze_brk_20",  "Bollinger",       "single", boll_squeeze_breakout, {}, TR, note="squeeze -> breakout"),
    Spec("boll_fade_22",    "Bollinger",       "single", boll_fade, {"k": 2.2}, MR, note="fade the extremes with trend"),
    Spec("macd_rev_15",     "MACD reversion",  "single", macd_reversion, {"z": 1.5}, MR, note="momentum flips first"),
    Spec("donchian_55",     "Channel breakout","single", donchian_breakout, {"n": 55}, TR, note="turtle 55"),
    Spec("donchian_20",     "Channel breakout","single", donchian_breakout, {"n": 20}, TR, note="turtle 20"),
    Spec("roll_high_252",   "Momentum",        "single", rolling_high_momentum, {}, LG, note="near-highs strength, long only"),
    Spec("tsmom_252_21",    "TS momentum",     "single", tsmom, {}, LG, note="12-1 style"),
    Spec("volbrk_2.0",      "Vol breakout",    "single", vol_breakout, {"k": 2.0}, TR, note="expansion with trend"),
    Spec("bigfade_2.5",     "Move fade",       "single", big_move_fade, {"k": 2.5}, MR, note="fade shocks with regime"),
    Spec("pair_BRENT_WTI",  "Stat-arb pair",   "pair",   pair_spread_z, {"n": 60, "z": 2.0}, MR,
         assets=("BRENT", "WTI"), fee_mult=2.0, note="two real legs, double fees"),
    Spec("pair_ETH_BTC",    "Stat-arb pair",   "pair",   pair_spread_z, {"n": 60, "z": 2.0}, MR,
         assets=("ETH", "BTC"), fee_mult=2.0, note="crypto ratio"),
    Spec("fx_xsec_63",      "X-sec momentum",  "xsec",   fx_xsec_momentum, {"look": 63}, BarrierSpec(3.0, 2.0, 21),
         assets=("EURUSD", "GBPUSD", "AUDUSD", "JPYUSD", "CHFUSD", "CADUSD"), note="rank FX, long top short bottom"),
    Spec("mom_riskon_vix",  "Regime-gated",    "vixgate", momentum_events, {"max_pct": 0.80}, TR,
         note="momentum only when VIX pct<80"),
    Spec("vix_contra_90",   "Sentiment",       "vixcontra", vix_contrarian_entries, {"min_pct": 0.90}, LG,
         assets=("WTI", "BRENT", "BTC", "ETH"), note="trade the crowd: extreme real fear -> long risk"),
    Spec("turn_of_month",   "Calendar",        "single", turn_of_month, {}, BarrierSpec(3.0, 2.0, 5),
         note="TOM effect, long only"),
]
