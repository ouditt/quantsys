"""Performance + reliability metrics.

The headline objective this system maximises, per the mandate:

    NET EXPECTANCY / trade = win_rate * avg_win_net  -  (1 - win_rate) * avg_loss_net

i.e. exactly "win rate x profit in the win", minus the loss side, with every
number NET of all trading fees. Win rate and average win are first-class
reported diagnostics — but they are never maximised in isolation, because a
high win rate with a low profit factor is the classic blow-up profile (many
small wins, rare catastrophic losses). Guardrails: profit factor and the
Deflated Sharpe Ratio must pass regardless of how pretty the win rate is.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

_N = NormalDist()


# ----------------------------------------------------------------- trade-level
@dataclass
class TradeStats:
    n: int
    win_rate: float
    avg_win: float
    avg_loss: float          # positive magnitude
    expectancy: float        # net, per trade
    profit_factor: float
    total_net: float

    def row(self) -> dict:
        return {
            "trades": self.n,
            "win rate (net)": f"{self.win_rate:6.1%}",
            "avg win (net)": f"{self.avg_win:+7.2%}",
            "avg loss (net)": f"{-self.avg_loss:+7.2%}",
            "expectancy/trade": f"{self.expectancy:+7.3%}",
            "profit factor": f"{self.profit_factor:5.2f}",
        }


def trade_stats(net_returns: np.ndarray) -> TradeStats:
    net = np.asarray(net_returns, dtype=float)
    if net.size == 0:
        return TradeStats(0, np.nan, np.nan, np.nan, np.nan, np.nan, 0.0)
    wins, losses = net[net > 0], net[net <= 0]
    wr = len(wins) / len(net)
    aw = wins.mean() if len(wins) else 0.0
    al = -losses.mean() if len(losses) else 0.0
    pf = wins.sum() / max(-losses.sum(), 1e-12) if len(losses) else np.inf
    return TradeStats(len(net), wr, aw, al, wr * aw - (1 - wr) * al, pf, net.sum())


# ----------------------------------------------------------------- series-level
def sharpe(r: pd.Series, periods: int = 252) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(periods)) if r.std() > 0 else 0.0


def sortino(r: pd.Series, periods: int = 252) -> float:
    r = r.dropna()
    dn = r[r < 0].std()
    return float(r.mean() / dn * np.sqrt(periods)) if dn and dn > 0 else 0.0


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r.fillna(0)).cumprod()
    return float((eq / eq.cummax() - 1).min())


def cagr(r: pd.Series, periods: int = 252) -> float:
    r = r.dropna()
    total = float((1 + r).prod())
    yrs = len(r) / periods
    return total ** (1 / yrs) - 1 if yrs > 0 and total > 0 else np.nan


# ------------------------------------------------------------------ reliability
def prob_sharpe(sr: float, n: int, skew: float, kurt: float,
                sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado): P(true SR > benchmark)
    given sample length, skew and fat tails. `sr` is per-period (not annualised)."""
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    z = (sr - sr_benchmark) * np.sqrt(max(n - 1, 1)) / denom
    return float(_N.cdf(z))


def expected_max_sharpe(n_trials: int, var_trials: float) -> float:
    """E[max SR] across n_trials of zero-skill strategies — the bar luck sets."""
    if n_trials <= 1:
        return 0.0
    gamma = 0.5772156649015329
    e = np.e
    return float(np.sqrt(max(var_trials, 1e-12)) *
                 ((1 - gamma) * _N.inv_cdf(1 - 1 / n_trials)
                  + gamma * _N.inv_cdf(1 - 1 / (n_trials * e))))


def deflated_sharpe(r: pd.Series, n_trials: int,
                    var_trials_sr: float | None = None) -> float:
    """DSR: probability the edge is real AFTER correcting for how many
    strategy/parameter variants were tried. The most important number when a
    result was found by search. Verdict scale: >=0.95 likely real; 0.80-0.95
    unconfirmed; <0.80 likely overfit — do not deploy."""
    r = r.dropna()
    if len(r) < 30 or r.std() == 0:
        return 0.0
    sr = float(r.mean() / r.std())                       # per-period SR
    sk = float(pd.Series(r).skew())
    ku = float(pd.Series(r).kurt()) + 3.0                # pandas gives excess
    if var_trials_sr is None:                            # conservative proxy
        var_trials_sr = (1 - sk * sr + (ku - 1) / 4 * sr**2) / max(len(r) - 1, 1)
    sr_star = expected_max_sharpe(n_trials, var_trials_sr)
    return prob_sharpe(sr, len(r), sk, ku, sr_benchmark=sr_star)


def min_track_record(sr: float, skew: float, kurt: float,
                     sr_benchmark: float = 0.0, conf: float = 0.95) -> float:
    """Bars of live evidence needed before the observed per-period SR is
    believable at `conf` against the benchmark."""
    if sr <= sr_benchmark:
        return np.inf
    z = _N.inv_cdf(conf)
    return 1 + (1 - skew * sr + (kurt - 1) / 4 * sr**2) * (z / (sr - sr_benchmark)) ** 2


def verdict(dsr: float) -> str:
    if dsr >= 0.95:
        return "LIKELY REAL EDGE — proceed to paper trading"
    if dsr >= 0.80:
        return "PROMISING, UNCONFIRMED — needs more out-of-sample evidence"
    return "LIKELY NOISE/OVERFIT — do not deploy; more tuning makes it worse"
