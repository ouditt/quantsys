"""Risk layer. Two laws above everything: never size to recover losses
(no martingale), and cap leverage for the 5-sigma day, not the average day."""
from __future__ import annotations

import numpy as np
import pandas as pd


def vol_target_scalar(strategy_ret: pd.Series, target_ann_vol: float = 0.10,
                      lookback: int = 60, periods: int = 252,
                      max_leverage: float = 2.0) -> pd.Series:
    """Scale exposure so realized portfolio vol tracks the budget; exposure
    shrinks automatically when volatility spikes (risk comes off exactly when
    markets are most dangerous). Uses only past data (shifted)."""
    rv = strategy_ret.rolling(lookback).std() * np.sqrt(periods)
    scal = (target_ann_vol / rv).clip(upper=max_leverage).shift(1)
    return scal.fillna(0.0)


def fixed_fractional_units(equity: float, risk_frac: float, entry: float,
                           stop: float) -> float:
    """Risk a fixed fraction of equity to the stop distance."""
    per_unit = abs(entry - stop)
    return 0.0 if per_unit <= 0 else (equity * risk_frac) / per_unit


def drawdown_throttle(dd: float) -> float:
    """The anti-martingale: exposure falls as drawdown deepens; 0 at the hard
    limit, where the kill switch flattens the book."""
    dd = abs(dd)
    if dd < 0.05:
        return 1.0
    if dd < 0.08:
        return 0.6
    if dd < 0.12:
        return 0.3
    return 0.0                       # kill-switch territory


KILL_SWITCH = {"daily_loss_limit": -0.03, "max_drawdown_limit": -0.12}
