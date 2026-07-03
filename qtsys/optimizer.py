"""Constrained portfolio optimizer (card 17): maximize Sharpe subject to
leverage, per-asset bounds, group (sector/class) caps, and a turnover penalty.
Scales to 100+ assets (SLSQP on the analytic gradient-free objective is fine
into the hundreds). Also ships inverse-vol and risk-parity for when you—
correctly—distrust mean estimates.

The honest label: optimizing weights ON PAST returns is estimation, not edge.
Use it to allocate across CERTIFIED strategies/sleeves, feed it shrunk
estimates, and re-run walk-forward. In-sample Sharpe below is descriptive.

Run:  python -m qtsys.optimizer     (real 11-asset demonstration)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data import real_daily_universe


def max_sharpe(rets: pd.DataFrame, bounds=(-0.5, 0.5), gross_cap=2.0,
               groups: dict[str, list[str]] | None = None, group_cap=1.0,
               w_prev: np.ndarray | None = None, turnover_pen=0.0) -> pd.Series:
    mu, cov = rets.mean().to_numpy() * 252, rets.cov().to_numpy() * 252
    n = len(mu)
    w0 = np.full(n, 1.0 / n)
    cols = list(rets.columns)

    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w) + 1e-12
        pen = turnover_pen * np.abs(w - w_prev).sum() if w_prev is not None else 0.0
        return -(w @ mu) / vol + pen

    cons = [{"type": "ineq", "fun": lambda w: gross_cap - np.abs(w).sum()}]
    if groups:
        for _, syms in groups.items():
            ix = [cols.index(s) for s in syms if s in cols]
            cons.append({"type": "ineq",
                         "fun": lambda w, ix=ix: group_cap - np.abs(w[ix]).sum()})
    res = minimize(neg_sharpe, w0, bounds=[bounds] * n, constraints=cons,
                   method="SLSQP", options={"maxiter": 500})
    return pd.Series(res.x, index=cols)


def inverse_vol(rets: pd.DataFrame) -> pd.Series:
    iv = 1.0 / rets.std()
    return iv / iv.sum()


def risk_parity(rets: pd.DataFrame, iters=500) -> pd.Series:
    cov = rets.cov().to_numpy() * 252
    n = cov.shape[0]
    w = np.full(n, 1.0 / n)
    for _ in range(iters):                                    # cyclical update
        rc = w * (cov @ w)
        w *= (rc.mean() / (rc + 1e-12)) ** 0.5
        w = np.clip(w, 1e-6, None); w /= w.sum()
    return pd.Series(w, index=rets.columns)


def _demo():
    uni = real_daily_universe()
    rets = (pd.DataFrame({s: d["close"] for s, d in uni.items()})
            .ffill().pct_change().tail(750).dropna(how="all"))
    groups = {"commod": ["WTI", "BRENT", "NATGAS"], "crypto": ["BTC", "ETH"],
              "fx": ["EURUSD", "GBPUSD", "AUDUSD", "JPYUSD", "CHFUSD", "CADUSD"]}
    w = max_sharpe(rets.fillna(0), groups=groups, group_cap=1.0)
    port = (rets.fillna(0) @ w)
    print("max-Sharpe (bounds ±50%, gross ≤2, class caps ≤1):")
    for s, x in w.round(3).items():
        if abs(x) > 0.01:
            print(f"  {s:7s} {x:+.1%}")
    print(f"gross {np.abs(w).sum():.2f} | in-sample ann Sharpe "
          f"{port.mean()/port.std()*np.sqrt(252):.2f}  (descriptive, not edge)")
    rp = risk_parity(rets.fillna(0))
    print("risk-parity (no mean estimates):",
          ", ".join(f"{s} {x:.1%}" for s, x in rp.round(3).items() if x > 0.03))


if __name__ == "__main__":
    _demo()
