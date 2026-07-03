"""Event-driven trade simulator with full fee accounting.

Honesty rules enforced structurally:
  * a signal at bar i is executed at bar i+1's close (no same-bar trading);
  * exits use the triple-barrier method — volatility-scaled profit-take and
    stop, plus a time limit — checked on closes from the bar AFTER entry;
  * EVERY side of EVERY trade is charged commission + half-spread + slippage,
    so a trade's P&L, and therefore the win rate, expectancy and profit
    factor built from it, are NET of all trading fees. A trade that gains
    less than its round-trip cost is counted as a LOSS.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .signals import Event, realized_vol


# ------------------------------------------------------------------------ fees
@dataclass(frozen=True)
class FeeModel:
    """All costs in basis points of notional, charged PER SIDE."""
    commission_bps: float = 0.0     # broker commission (0 for US retail equities)
    half_spread_bps: float = 2.0    # you cross half the bid-ask each side
    slippage_bps: float = 1.0       # adverse move between decision and fill

    @property
    def per_side(self) -> float:
        return (self.commission_bps + self.half_spread_bps + self.slippage_bps) / 1e4

    @property
    def round_trip(self) -> float:
        return 2 * self.per_side


US_EQUITY_FEES = FeeModel(0.0, 2.0, 1.0)      # liquid ETFs/large caps, ~6 bps RT
CRYPTO_FEES = FeeModel(10.0, 2.0, 3.0)        # taker fee dominates, ~30 bps RT


# ----------------------------------------------------------------------- trades
@dataclass
class Trade:
    asset: str
    strategy: str
    side: int
    i_signal: int
    i_entry: int
    i_exit: int
    ts_entry: object      # pd.Timestamp — real calendar time drives CV purging
    ts_exit: object
    entry: float
    exit: float
    gross_ret: float
    fees: float
    net_ret: float
    exit_reason: str
    features: dict = field(default_factory=dict)

    @property
    def is_win(self) -> bool:            # win means NET of all fees
        return self.net_ret > 0


@dataclass(frozen=True)
class BarrierSpec:
    """Triple-barrier exit: vol-scaled TP/SL plus a time limit."""
    tp_mult: float = 2.0
    sl_mult: float = 1.5
    max_hold: int = 15
    vol_lookback: int = 20

    def widths(self, vol_daily: float) -> tuple[float, float]:
        w = vol_daily * np.sqrt(self.max_hold)
        return self.tp_mult * w, self.sl_mult * w


def simulate_trades(df: pd.DataFrame, events: list[Event], barriers: BarrierSpec,
                    fees: FeeModel, features: pd.DataFrame | None = None,
                    allow_overlap: bool = False) -> list[Trade]:
    """Run each event through entry -> barrier exit. One position per asset at a
    time unless allow_overlap (keeps the trade ledger unambiguous)."""
    close = df["close"].to_numpy()
    vol = realized_vol(df["close"], barriers.vol_lookback).to_numpy()
    n = len(close)
    trades: list[Trade] = []
    busy_until = -1
    for ev in events:
        i_entry = ev.i_signal + 1                      # next-bar execution
        if i_entry >= n - 1 or np.isnan(vol[ev.i_signal]):
            continue
        if not allow_overlap and i_entry <= busy_until:
            continue
        entry = close[i_entry]
        tp_w, sl_w = barriers.widths(vol[ev.i_signal])
        tp = entry * (1 + ev.side * tp_w)
        sl = entry * (1 - ev.side * sl_w)
        i_exit, reason = min(i_entry + barriers.max_hold, n - 1), "time"
        for j in range(i_entry + 1, min(i_entry + barriers.max_hold, n - 1) + 1):
            if (ev.side == 1 and close[j] >= tp) or (ev.side == -1 and close[j] <= tp):
                i_exit, reason = j, "tp"; break
            if (ev.side == 1 and close[j] <= sl) or (ev.side == -1 and close[j] >= sl):
                i_exit, reason = j, "sl"; break
        gross = ev.side * (close[i_exit] / entry - 1)
        net = gross - fees.round_trip
        feat = {}
        if features is not None:
            row = features.iloc[ev.i_signal]
            feat = {k: float(v) for k, v in row.items()} if not row.isna().any() else {}
        trades.append(Trade(ev.asset, ev.strategy, ev.side, ev.i_signal, i_entry,
                            i_exit, df.index[i_entry], df.index[i_exit], entry,
                            close[i_exit], gross, fees.round_trip, net, reason, feat))
        busy_until = i_exit
    return trades


def equity_curve(trades: list[Trade], n_bars: int, index: pd.Index,
                 max_slots: int = 10) -> pd.Series:
    """Daily portfolio return series from the trade ledger: each trade occupies
    one of `max_slots` equal capital slots; fees hit on entry/exit bars. This is
    the series the Sharpe/drawdown/DSR are computed on — consistent with the
    per-trade ledger by construction."""
    ret = np.zeros(n_bars)
    for t in sorted(trades, key=lambda x: x.i_entry):
        path = t.exit / t.entry
        # spread the gross return over holding bars geometrically, fees at ends
        h = max(t.i_exit - t.i_entry, 1)
        per_bar = path ** (1 / h) - 1
        for j in range(t.i_entry + 1, t.i_exit + 1):
            ret[j] += t.side * per_bar / max_slots
        ret[t.i_entry] -= (t.fees / 2) / max_slots
        ret[t.i_exit] -= (t.fees / 2) / max_slots
    return pd.Series(ret, index=index[:n_bars], name="strategy_ret")


def equity_curve_real(trades: list[Trade], universe: dict[str, pd.DataFrame],
                      max_slots: int = 10) -> pd.Series:
    """Daily portfolio return series on the UNION calendar, marked to market
    from REAL daily closes (no smoothing): each open trade contributes
    side * (close_j / close_{j-1} - 1) / max_slots on each held day; fees hit
    on entry/exit days. Even spreading of a trade's P&L would manufacture
    autocorrelation and silently inflate inference statistics (PSR/DSR assume
    i.i.d.) — measured here as a 23% false-pass rate before this fix."""
    acc: dict = {}
    for t in trades:
        c = universe[t.asset]["close"].to_numpy()
        idx = universe[t.asset].index
        for j in range(t.i_entry + 1, t.i_exit + 1):
            d = idx[j]
            acc[d] = acc.get(d, 0.0) + t.side * (c[j] / c[j - 1] - 1) / max_slots
        acc[t.ts_entry] = acc.get(t.ts_entry, 0.0) - (t.fees / 2) / max_slots
        acc[t.ts_exit] = acc.get(t.ts_exit, 0.0) - (t.fees / 2) / max_slots
    if not acc:
        return pd.Series(dtype=float, name="strategy_ret")
    s = pd.Series(acc, name="strategy_ret").sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq="B")
    return s.reindex(full, fill_value=0.0)
