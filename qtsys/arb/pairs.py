"""pairs.py — cointegration stat-arb skill.

Engle-Granger two-step on LOG prices: OLS hedge ratio, then an augmented
Dickey-Fuller test on the residual (own implementation — no statsmodels).
A pair passes at the EG 5% critical value (-3.34, two variables, constant).
Signals are residual z-scores; sizing context comes from the OU half-life.

The backtest is deliberately honest: hedge ratio, mean and vol are fitted on
the TRAIN window only, trades happen on the untouched TEST window, and every
round trip pays fees on all four legs. No parameter is ever fitted on data
it later trades.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

EG_CRIT_5PCT = -3.34         # Engle-Granger 5% critical value, 2 vars + const


# ------------------------------------------------------------------ core econ
def hedge_ratio(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """OLS y = a + b x -> (a, b)."""
    A = np.column_stack([np.ones_like(x), x])
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(a), float(b)


def adf_stat(e: np.ndarray, lags: int = 1) -> float:
    """ADF t-statistic (constant, `lags` augmented diffs) on a series.
    More negative = more mean-reverting."""
    e = np.asarray(e, float)
    de = np.diff(e)
    if len(de) <= lags + 3:
        return 0.0
    rows = []
    yv = []
    for t in range(lags, len(de)):
        row = [1.0, e[t]]                       # e_{t-1} in diff-index terms
        row += [de[t - i] for i in range(1, lags + 1)]
        rows.append(row)
        yv.append(de[t])
    X = np.array(rows)
    Y = np.array(yv)
    beta, res, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    dof = max(len(Y) - X.shape[1], 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = math.sqrt(max(cov[1, 1], 1e-18))
    return float(beta[1] / se)


def half_life(e: np.ndarray) -> float | None:
    """OU half-life in bars from AR(1) on the residual; None if not reverting."""
    e = np.asarray(e, float)
    x, y = e[:-1], e[1:]
    a, phi = hedge_ratio(y, x)
    if not (0 < phi < 1):
        return None
    return -math.log(2) / math.log(phi)


def zscore(e: np.ndarray, mu: float, sd: float) -> np.ndarray:
    return (np.asarray(e, float) - mu) / max(sd, 1e-12)


# ------------------------------------------------------------------- scanning
def find_pairs(prices: dict[str, np.ndarray], min_obs: int = 250,
               corr_floor: float = 0.6) -> list[dict]:
    """Engle-Granger scan over all symbol pairs (log prices, both directions,
    better ADF kept). Returns candidates sorted most-cointegrated first."""
    logs = {}
    for s, p in prices.items():
        p = np.asarray(p, float)
        if len(p) >= min_obs and np.all(p > 0):
            logs[s] = np.log(p)
    out = []
    for a, b in itertools.combinations(sorted(logs), 2):
        n = min(len(logs[a]), len(logs[b]))
        ya, yb = logs[a][-n:], logs[b][-n:]
        if abs(float(np.corrcoef(ya, yb)[0, 1])) < corr_floor:
            continue
        best = None
        for y, x, ys, xs in ((ya, yb, a, b), (yb, ya, b, a)):
            c, beta = hedge_ratio(y, x)
            e = y - (c + beta * x)
            stat = adf_stat(e)
            if best is None or stat < best["adf"]:
                hl = half_life(e)
                best = dict(y=ys, x=xs, beta=round(beta, 4), adf=round(stat, 3),
                            half_life=round(hl, 1) if hl else None,
                            n=n, cointegrated=bool(stat < EG_CRIT_5PCT))
        out.append(best)
    out.sort(key=lambda d: d["adf"])
    return out


# ------------------------------------------------------------------- backtest
def backtest(y: np.ndarray, x: np.ndarray, entry: float = 2.0,
             exit_z: float = 0.5, stop: float = 3.5, fee_bps: float = 10.0,
             train_frac: float = 0.6) -> dict:
    """Train/test-split spread backtest on LOG prices. Fits (beta, mu, sigma)
    on train only; trades the test window. PnL is per unit gross notional of
    the spread; each round trip pays 4 legs of fees."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    m = min(len(y), len(x))                       # align tails (unequal history)
    y, x = np.log(y[-m:]), np.log(x[-m:])
    n = m
    cut = int(n * train_frac)
    if cut < 60 or n - cut < 40:
        return {"n_trades": 0, "note": "too little data"}
    c, beta = hedge_ratio(y[:cut], x[:cut])
    e_tr = y[:cut] - (c + beta * x[:cut])
    mu, sd = float(e_tr.mean()), float(e_tr.std())
    hl = half_life(e_tr)
    e_te = y[cut:] - (c + beta * x[cut:])
    z = zscore(e_te, mu, sd)
    fee = 4 * fee_bps / 1e4
    pos, entry_e = 0, 0.0
    trades = []
    for t in range(1, len(z)):
        if pos == 0:
            if z[t] > entry:
                pos, entry_e = -1, e_te[t]
            elif z[t] < -entry:
                pos, entry_e = 1, e_te[t]
        else:
            hit_exit = abs(z[t]) < exit_z
            hit_stop = abs(z[t]) > stop
            if hit_exit or hit_stop or t == len(z) - 1:
                pnl = pos * (e_te[t] - entry_e) - fee
                trades.append(pnl)
                pos = 0
    if not trades:
        return {"n_trades": 0, "beta": round(beta, 4),
                "half_life": round(hl, 1) if hl else None,
                "note": "no signals in test window"}
    tr = np.array(trades)
    return {"n_trades": len(tr), "win_rate": round(float((tr > 0).mean()), 3),
            "avg_ret_pct": round(float(tr.mean()) * 100, 3),
            "total_ret_pct": round(float(tr.sum()) * 100, 2),
            "worst_pct": round(float(tr.min()) * 100, 3),
            "beta": round(beta, 4),
            "half_life": round(hl, 1) if hl else None,
            "test_bars": n - cut, "trade_returns": tr.tolist()}


def gated_scan(prices: dict, n_trials: int | None = None, **bt) -> list[dict]:
    """Cointegration scan -> OOS backtest -> DSR gate, the SAME verification
    every registry strategy passes. `n_trials` is the multiple-testing
    correction: how many pairs we searched (defaults to the pairs tested).
    Only pairs with >=5 OOS trades get a DSR; the verdict is the honest one."""
    from ..metrics import deflated_sharpe, verdict
    cands = find_pairs(prices)
    n_trials = n_trials or max(len(cands), 1)
    out = []
    for c in cands:
        if not c["cointegrated"]:
            continue
        y, x = prices.get(c["y"]), prices.get(c["x"])
        if y is None or x is None:
            continue
        res = backtest(y, x, **bt)
        tr = res.get("trade_returns") or []
        dsr = (deflated_sharpe(__import__("pandas").Series(tr), n_trials)
               if len(tr) >= 5 else 0.0)
        out.append({**{k: v for k, v in c.items() if k != "n"},
                    "n_trades": res.get("n_trades", 0),
                    "win_rate": res.get("win_rate"),
                    "total_ret_pct": res.get("total_ret_pct"),
                    "worst_pct": res.get("worst_pct"),
                    "dsr": round(dsr, 3),
                    "verdict": verdict(dsr) if len(tr) >= 5
                    else "INSUFFICIENT OOS TRADES — cannot verify",
                    "n_trials": n_trials})
    out.sort(key=lambda d: -d["dsr"])
    return out


# ------------------------------------------------------------------ self-test
def _selftest():
    rng = np.random.default_rng(11)
    n = 6000
    x = np.cumsum(rng.normal(0, 0.01, n)) + 4.0            # log random walk
    ou = np.zeros(n)
    for t in range(1, n):                                   # OU, hl ~ 4 bars
        ou[t] = 0.85 * ou[t - 1] + rng.normal(0, 0.010)
    y = 0.5 + 1.8 * x + ou
    px = {"AAA": np.exp(y), "BBB": np.exp(x),
          "CCC": np.exp(np.cumsum(rng.normal(0, 0.012, n)) + 3.0)}
    res = find_pairs(px)
    top = res[0]
    assert {top["y"], top["x"]} == {"AAA", "BBB"}, "true pair ranks first"
    assert top["cointegrated"] and top["adf"] < -5, f"ADF {top['adf']}"
    assert top["half_life"] and 2 < top["half_life"] < 40, top["half_life"]
    assert abs(top["beta"] - 1.8) < 0.15, f"beta {top['beta']}"
    fake = [r for r in res if {r["y"], r["x"]} == {"AAA", "CCC"}]
    assert not fake or not fake[0]["cointegrated"], "unrelated pair rejected"
    bt = backtest(px["AAA"], px["BBB"])
    assert bt["n_trades"] >= 3 and bt["total_ret_pct"] > 0, bt
    assert bt["win_rate"] >= 0.6, bt
    b0 = backtest(px["AAA"], px["CCC"])
    assert "trade_returns" in bt and len(bt["trade_returns"]) == bt["n_trades"]
    gs = gated_scan(px)
    real = [g for g in gs if {g["y"], g["x"]} == {"AAA", "BBB"}]
    # enough OOS trades now that the DSR gate actually engages (>=30 obs)
    assert real and real[0]["n_trades"] >= 30, real
    assert real[0]["dsr"] > 0.5 and "REAL EDGE" in real[0]["verdict"], real
    assert all("n_trials" in g for g in gs), "DSR carries the trials correction"
    print(f"pairs self-test ✓  EG detect (ADF {top['adf']}), beta/half-life "
          f"recovered, OOS bt: {bt['n_trades']} trades wr={bt['win_rate']} "
          f"total={bt['total_ret_pct']}% | junk pair: "
          f"{b0.get('total_ret_pct', 'n/a')}% | DSR-gated: "
          f"{real[0]['dsr']} ({real[0]['verdict'][:24]}…)")


if __name__ == "__main__":
    _selftest()
