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


# ------------------------------------------------- factor model (PORT-lite)
# observable daily factors built from the SAME real bundled history the book
# trades — no fitted latent factors, every beta is checkable by hand
FACTOR_DEFS = {
    "ENERGY":  (("WTI", "BRENT"), +1.0),
    "CRYPTO":  (("BTC", "ETH"), +1.0),
    "USD":     (("EURUSD", "GBPUSD", "AUDUSD", "CADUSD", "CHFUSD"), -1.0),
    "RISKOFF": (("VIX",), +1.0),
}


def factor_panel(uni=None, lookback: int = 500) -> pd.DataFrame:
    r = _returns_panel(uni, lookback)
    out = {}
    for name, (syms, sign) in FACTOR_DEFS.items():
        cols = [s for s in syms if s in r.columns]
        if cols:
            out[name] = sign * r[cols].mean(axis=1)
    return pd.DataFrame(out).dropna(how="all")


def factor_exposures(weights: dict[str, float], uni=None,
                     lookback: int = 500) -> dict:
    """OLS of book P&L on the observable factors: betas, t-stats, R², and each
    factor's share of book variance (beta_k·cov(f_k,pnl)/var(pnl))."""
    import math
    uni = uni or real_daily_universe()
    pnl = portfolio_series(weights, uni, lookback)
    F = factor_panel(uni, lookback).reindex(pnl.index).dropna()
    pnl = pnl.reindex(F.index)
    if len(F) < 60 or float(pnl.abs().sum()) == 0:
        return {"factors": [], "r2": None, "idio_pct": None}
    X = np.column_stack([np.ones(len(F)), F.to_numpy()])
    y = pnl.to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    var_p = float(np.var(y))
    r2 = 1 - float(resid @ resid) / max(len(y) * var_p, 1e-18)
    rows = []
    for i, name in enumerate(F.columns):
        b = float(beta[i + 1])
        t = b / math.sqrt(max(cov[i + 1, i + 1], 1e-18))
        share = b * float(np.cov(F[name].to_numpy(), y)[0, 1]) / max(var_p, 1e-18)
        rows.append({"factor": name, "beta": round(b, 3), "t": round(t, 1),
                     "var_share_pct": round(share * 100, 1)})
    rows.sort(key=lambda x: -abs(x["var_share_pct"]))
    return {"factors": rows, "r2": round(r2, 3),
            "idio_pct": round((1 - r2) * 100, 1)}


def attribution(weights: dict[str, float], uni=None,
                lookback: int = 500, level: float = 0.99) -> dict:
    """Euler-consistent tail attribution on the historical simulation: each
    position's mean P&L over the tail days sums EXACTLY to portfolio CVaR,
    so the shares are additive and honest. Plus standalone VaR95 per
    position and the diversification ratio."""
    uni = uni or real_daily_universe()
    r = _returns_panel(uni, lookback)
    w = pd.Series(weights).reindex(r.columns).fillna(0.0)
    parts = r * w                                  # per-position daily P&L
    pnl = parts.sum(axis=1)
    q = pnl.quantile(1 - level)
    tail = pnl <= q
    cvar = float(pnl[tail].mean())
    rows, sa_sum = [], 0.0
    for s in weights:
        if s not in parts.columns or not weights[s]:
            continue
        comp = float(parts.loc[tail, s].mean())
        sa = float(parts[s].quantile(0.05))
        sa_sum += abs(sa)
        rows.append({"symbol": s, "weight": round(weights[s], 4),
                     "cvar_contrib": round(comp, 5),
                     "cvar_share_pct": round(comp / cvar * 100, 1) if cvar else None,
                     "standalone_var95": round(sa, 5)})
    rows.sort(key=lambda x: x["cvar_contrib"])
    v95, _ = var_cvar(pnl, 0.95)
    div = round(sa_sum / abs(v95), 2) if v95 else None
    return {"level": level, "cvar": round(cvar, 5), "rows": rows,
            "diversification_ratio": div,
            "note": "contributions sum to CVaR (Euler/tail-conditional)"}


def report(weights: dict[str, float] | None = None) -> str:
    if not weights:
        return ("PORTFOLIO RISK REPORT — book is FLAT (no live positions "
                "covered by the risk history). Nothing at risk; VaR/stress/"
                "attribution resume with the first covered position.\n\n"
                + limit_protocol())
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
    fx = factor_exposures(weights, uni)
    if fx["factors"]:
        lines.append("\nFACTOR EXPOSURES (OLS on observable real-data factors):")
        for f in fx["factors"]:
            lines.append(f"  {f['factor']:8s} β {f['beta']:+.3f} (t {f['t']:+.1f})"
                         f"  drives {f['var_share_pct']:+.1f}% of book variance")
        lines.append(f"  systematic R² {fx['r2']:.0%} · idiosyncratic {fx['idio_pct']:.0f}%")
    att = attribution(weights, uni)
    if att["rows"]:
        lines.append("\nRISK ATTRIBUTION (tail-conditional — shares sum to CVaR99):")
        for a in att["rows"]:
            lines.append(f"  {a['symbol']:7s} w {a['weight']:+.2f}  CVaR contrib "
                         f"{a['cvar_contrib']:+.4f} ({a['cvar_share_pct']}%)  "
                         f"standalone VaR95 {a['standalone_var95']:+.4f}")
        lines.append(f"  diversification ratio {att['diversification_ratio']}x "
                     "(standalone risk vs actual book risk)")
    cl = clusters(uni)
    lines.append("\nCORRELATION CLUSTERS (|ρ|≥0.70, 120d): " +
                 ("; ".join(sorted("+".join(sorted(g)) for g in cl)) if cl else "none"))
    lines.append("\n" + limit_protocol())
    return "\n".join(lines)


def _selftest():
    demo = {"WTI": 0.28, "BRENT": -0.18, "BTC": 0.19, "EURUSD": -0.23, "ETH": 0.11}
    uni = real_daily_universe()
    att = attribution(demo, uni)
    total = sum(a["cvar_contrib"] for a in att["rows"])
    assert abs(total - att["cvar"]) < 1e-9, "Euler contributions must sum to CVaR"
    fx = factor_exposures(demo, uni)
    assert fx["r2"] is not None and 0 <= fx["r2"] <= 1
    assert any(f["factor"] == "ENERGY" for f in fx["factors"])
    flat = report(None)
    assert "FLAT" in flat and "PROTOCOL" in flat
    print("portfolio_risk self-test ✓  Euler sums to CVaR, R² sane, flat-book honest")
    print()
    print(report(demo))


if __name__ == "__main__":
    _selftest()
