"""Cross-asset factor model (card 12): momentum, value (LT reversal), low-vol,
built as monthly rank long-short portfolios on the REAL universe; then any
return stream can be regressed on them to split its P&L into factor exposure
vs residual alpha. Locally, point `build_factors` at 500 stocks via yfinance
and the same code becomes an equity factor model.

Run:  python -m qtsys.factors
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import real_daily_universe


def build_factors(uni=None, top=3) -> pd.DataFrame:
    uni = uni or real_daily_universe()
    px = pd.DataFrame({s: d["close"] for s, d in uni.items()}).ffill()
    m = px.resample("ME").last()
    r1 = m.pct_change()                                     # next-month realized
    sigs = {"momentum": m.pct_change(12).shift(1),          # 12m, skip formation month
            "value_rev": -m.pct_change(60).shift(1),        # 5y reversal
            "low_vol": -px.pct_change().rolling(63).std().resample("ME").last().shift(1)}
    out = {}
    for name, s in sigs.items():
        f = []
        for dt in m.index:
            row = s.loc[dt].dropna()
            nxt = r1.shift(-1).loc[dt]
            if len(row) < top * 2:
                f.append(np.nan); continue
            ranked = row.sort_values()
            f.append(nxt[ranked.index[-top:]].mean() - nxt[ranked.index[:top]].mean())
        out[name] = pd.Series(f, index=m.index)
    return pd.DataFrame(out).dropna(how="all")


def exposures(port_ret_m: pd.Series, factors: pd.DataFrame) -> dict:
    """OLS betas + annualized residual alpha of a monthly return stream."""
    df = factors.join(port_ret_m.rename("p"), how="inner").dropna()
    X = np.column_stack([np.ones(len(df)), df[factors.columns].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, df["p"].to_numpy(), rcond=None)
    resid = df["p"].to_numpy() - X @ beta
    return {"alpha_ann": beta[0] * 12,
            **{f"beta_{c}": b for c, b in zip(factors.columns, beta[1:])},
            "resid_vol_ann": resid.std() * np.sqrt(12),
            "r2": 1 - resid.var() / df["p"].var()}


def _demo():
    uni = real_daily_universe()
    f = build_factors(uni)
    print("factor long-short monthly returns (real universe, top/bottom 3):")
    print((f.mean() * 12).round(3).to_string(), "\n  ^ annualized means")
    px = pd.DataFrame({s: d["close"] for s, d in uni.items()}).ffill()
    eqw = px.pct_change().mean(axis=1).resample("ME").apply(lambda x: (1 + x).prod() - 1)
    ex = exposures(eqw, f)
    print("\nequal-weight book decomposed:",
          {k: round(v, 3) for k, v in ex.items()})
    print("reading: sizable betas = returns you could buy cheaply; "
          "pay yourself only for alpha_ann that survives with r2 low.")


if __name__ == "__main__":
    _demo()
