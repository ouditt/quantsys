"""Portfolio risk engine (cards 7 + 14) — VaR, CVaR, real-crash stress tests,
correlation clustering, and hard limit protocols. 100% real data: VaR is
historical simulation on actual recorded returns, and every stress scenario is
an ACTUAL dated window replayed against the current book — no hypothetical
shocks, only things markets have really done.

Run:  python -m qtsys.portfolio_risk
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import real_daily_universe

# real, dated crisis windows fully covered by the bundled history
STRESS_WINDOWS = {
    "GFC oil collapse      2008-07-14→2008-12-19": ("2008-07-14", "2008-12-19"),
    "COVID crash           2020-02-14→2020-03-23": ("2020-02-14", "2020-03-23"),
    "2022 rates/crypto     2022-04-01→2022-06-30": ("2022-04-01", "2022-06-30"),
    "GBP flash/Brexit      2016-06-01→2016-10-31": ("2016-06-01", "2016-10-31"),
    "Crypto winter 2018    2018-01-08→2018-12-14": ("2018-01-08", "2018-12-14"),
}

LIMITS = {"daily_loss": -0.03, "weekly_loss": -0.06, "max_drawdown": -0.12,
          "max_cluster_positions": 2, "cluster_corr": 0.70}


def _returns_panel(uni=None, lookback: int = 500) -> pd.DataFrame:
    uni = uni or real_daily_universe()
    px = pd.DataFrame({s: df["close"] for s, df in uni.items()})
    return px.ffill().pct_change().tail(lookback)


def portfolio_series(weights: dict[str, float], uni=None,
                     lookback: int = 500) -> pd.Series:
    """Daily P&L (as fraction of equity) of the CURRENT book replayed over the
    last `lookback` real days. weights = signed notional / equity per symbol."""
    r = _returns_panel(uni, lookback)
    w = pd.Series(weights).reindex(r.columns).fillna(0.0)
    return (r * w).sum(axis=1)


def var_cvar(pnl: pd.Series, level: float = 0.99) -> tuple[float, float]:
    q = pnl.quantile(1 - level)
    return float(q), float(pnl[pnl <= q].mean())


def stress_test(weights: dict[str, float], uni=None) -> pd.DataFrame:
    uni = uni or real_daily_universe()
    px = pd.DataFrame({s: df["close"] for s, df in uni.items()}).ffill()
    rows = []
    for name, (a, b) in STRESS_WINDOWS.items():
        win = px.loc[a:b]
        if len(win) < 5:
            continue
        rets = win.pct_change().dropna(how="all")
        w = pd.Series(weights).reindex(rets.columns).fillna(0.0)
        covered = [s for s, x in weights.items() if s in rets and rets[s].notna().sum() > 3]
        pnl = (rets * w).sum(axis=1, min_count=1).fillna(0.0)
        eq = (1 + pnl).cumprod()
        rows.append({"scenario": name, "total": eq.iloc[-1] - 1,
                     "worst_day": pnl.min(), "days": len(pnl),
                     "book_coverage": f"{len(covered)}/{sum(1 for v in weights.values() if v)}"})
    return pd.DataFrame(rows)


def clusters(uni=None, lookback: int = 120, thr: float | None = None) -> list[set]:
    """Union-find grouping of symbols with |corr| above the limit — the
    correlation rule: positions in one cluster count as ONE bet."""
    thr = thr or LIMITS["cluster_corr"]
    r = _returns_panel(uni, lookback)
    c = r.corr().abs()
    syms = list(c.columns)
    parent = {s: s for s in syms}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, a in enumerate(syms):
        for b in syms[i + 1:]:
            if c.loc[a, b] >= thr:
                parent[find(a)] = find(b)
    groups: dict[str, set] = {}
    for s in syms:
        groups.setdefault(find(s), set()).add(s)
    return [g for g in groups.values() if len(g) > 1]


def limit_protocol() -> str:
    return (f"PROTOCOL — every limit has ONE pre-agreed action, so emotion never votes:\n"
            f"  daily loss ≤ {LIMITS['daily_loss']:.0%}: gateway halts NEW entries for the session; "
            f"exits still allowed; log the cause before resuming tomorrow\n"
            f"  weekly loss ≤ {LIMITS['weekly_loss']:.0%}: risk drops one posture level for 2 weeks; "
            f"Validation Officer re-runs the sweep before restore\n"
            f"  drawdown ≤ {LIMITS['max_drawdown']:.0%}: KILL SWITCH — flatten, halt, full review; "
            f"resume needs a typed confirm and a written cause\n"
            f"  correlation: ≥{LIMITS['cluster_corr']:.0%}-correlated symbols form a cluster; "
            f"max {LIMITS['max_cluster_positions']} concurrent positions per cluster")


def report(weights: dict[str, float] | None = None) -> str:
    weights = weights or {"WTI": 0.28, "BRENT": -0.18, "BTC": 0.19, "EURUSD": -0.23, "ETH": 0.11}
    uni = real_daily_universe()
    pnl = portfolio_series(weights, uni)
    v99, c99 = var_cvar(pnl, 0.99)
    v95, c95 = var_cvar(pnl, 0.95)
    lines = ["PORTFOLIO RISK REPORT — historical simulation on real returns",
             f"book (notional/equity): {weights}",
             f"1-day VaR 99%: {v99:+.2%}   CVaR 99% (tail mean): {c99:+.2%}",
             f"1-day VaR 95%: {v95:+.2%}   CVaR 95%: {c95:+.2%}",
             f"ann. vol of book: {pnl.std() * np.sqrt(252):.1%}", "",
             "STRESS — the current book replayed through REAL crises:"]
    st = stress_test(weights, uni)
    for _, r in st.iterrows():
        lines.append(f"  {r['scenario']}  total {r['total']:+.1%}  worst day {r['worst_day']:+.2%}"
                     f"  ({r['days']} days, coverage {r['book_coverage']})")
    cl = clusters(uni)
    lines.append("\nCORRELATION CLUSTERS (|ρ|≥0.70, 120d): " +
                 ("; ".join(sorted("+".join(sorted(g)) for g in cl)) if cl else "none"))
    lines.append("\n" + limit_protocol())
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
