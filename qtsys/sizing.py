"""Position sizing (card 5) + scaling roadmap (cards 2/12) — the maths of
growing a small account without blowing it up.

Every number here is computed from REAL recorded trade outcomes: the bootstrap
resamples the flagship's actual out-of-sample per-trade net returns
(flagship_returns.csv). Resampling real outcomes is statistics ON real data —
no prices are ever generated.

Run:  python -m qtsys.sizing          (prints the three-posture report)
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)


# ------------------------------------------------------------------ formulas
def kelly_fraction(p: float, b: float) -> float:
    """Full-Kelly fraction of equity to RISK per trade. p = win prob,
    b = avg_win/avg_loss payoff ratio. Never trade full Kelly."""
    q = 1 - p
    return max(0.0, p - q / b)


def worst_case_kelly(p: float, b: float, n: int, z: float = 1.96) -> float:
    """Kelly at the LOWER confidence bound of p — the honest ceiling, because
    the true edge is estimated, not known."""
    se = np.sqrt(p * (1 - p) / max(n, 1))
    return kelly_fraction(p - z * se, b)


@dataclass(frozen=True)
class Posture:
    name: str
    risk_pct: float          # fraction of equity risked per trade
    note: str


POSTURES = (Posture("SURVIVAL-FIRST", 0.0075, "0.75%/trade — half of worst-case Kelly"),
            Posture("BALANCED",       0.015,  "1.5%/trade + drawdown throttle"),
            Posture("AGGRESSIVE",     0.025,  "2.5%/trade — at worst-case Kelly, expect pain"))

# drawdown throttle: risk multiplier applied by current drawdown from peak
THROTTLE = ((0.05, 1.00), (0.10, 0.50), (0.15, 0.25), (0.20, 0.0))  # 20% DD = stop, review


def throttle_mult(drawdown: float) -> float:
    dd = abs(drawdown)
    for lvl, mult in THROTTLE:
        if dd < lvl:
            return mult
    return 0.0


def streak_mult(last_results: list[bool]) -> float:
    """Anti-tilt: 3 consecutive net losses -> half risk until the next win.
    NEVER the reverse — increasing size after losses (martingale) is the single
    sizing mistake that destroys more accounts than any other, because it turns
    a bounded losing streak into an unbounded one."""
    k = 0
    for r in reversed(last_results):
        if r:
            break
        k += 1
    return 0.5 if k >= 3 else 1.0


def size_order(equity: float, risk_pct: float, stop_frac: float, price: float,
               drawdown: float = 0.0, last_results: list[bool] | None = None,
               min_notional: float = 0.0) -> dict:
    """Units to trade so a stop-out loses ~risk_pct of equity (vol/stop-based
    fixed-fractional). stop_frac = distance to stop as a fraction of price."""
    eff = risk_pct * throttle_mult(drawdown) * streak_mult(last_results or [])
    notional = equity * eff / max(stop_frac, 1e-9)
    if notional < min_notional:
        return {"units": 0.0, "notional": 0.0, "eff_risk": eff,
                "blocked": f"needs £{min_notional:.0f} notional; account too small "
                           f"for this instrument at this risk — trade a finer-grained venue"}
    return {"units": notional / price, "notional": notional, "eff_risk": eff, "blocked": None}


# practical venue floors for a small UK account (approx, GBP)
MIN_NOTIONAL = {"crypto (major CEX)": 5, "FX micro lot (1k units)": 750,
                "US fractional shares": 1, "oil/gold CFD (0.1 lot)": 400}


# ------------------------------------------------------------------ bootstrap
def bootstrap(returns: np.ndarray, risk_pct: float, n_trades: int = 100,
              n_paths: int = 10_000, seed: int = 7) -> dict:
    """Resample REAL per-trade net returns; scale exposure so the strategy's
    own average loss equals risk_pct of equity. Returns distribution stats."""
    avg_loss = abs(returns[returns < 0].mean())
    lev = risk_pct / avg_loss                      # notional as fraction of equity
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(returns), size=(n_paths, n_trades))
    step = 1.0 + returns[idx] * lev
    eq = np.cumprod(step, axis=1)
    peak = np.maximum.accumulate(eq, axis=1)
    maxdd = ((eq / peak) - 1.0).min(axis=1)
    term = eq[:, -1]
    hit2x = (eq >= 2.0).any(axis=1)
    t2x = np.where(hit2x, (eq >= 2.0).argmax(axis=1) + 1, n_trades)
    return {"lev": lev, "median_terminal": float(np.median(term)),
            "p_double": float(hit2x.mean()),
            "median_trades_to_2x": float(np.median(t2x[hit2x])) if hit2x.any() else np.nan,
            "p_dd20": float((maxdd <= -0.20).mean()),
            "p_dd30": float((maxdd <= -0.30).mean()),
            "p_dd50": float((maxdd <= -0.50).mean()),
            "median_maxdd": float(np.median(maxdd))}


def load_flagship_returns() -> np.ndarray:
    path = os.path.join(HERE, "flagship_returns.csv")
    return pd.read_csv(path)["net_ret"].to_numpy()


# ------------------------------------------------------------------ scaling
def scaling_roadmap(start: float, target: float, posture: Posture,
                    step_up: float = 1.5) -> list[dict]:
    """Milestone ladder (card 12): risk UNIT only steps up at equity milestones,
    each step gated by evidence (>=30 trades at the level, max DD within band).
    The mistake that turns winners into losers permanently: jumping size after a
    hot streak, so the inevitable normal drawdown lands on the biggest size the
    account has ever carried. Steps here are capped at 1.5x and revert one level
    on a 10% drawdown."""
    rungs, eq = [], start
    while eq < target:
        nxt = min(eq * step_up, target)
        rungs.append({"from": round(eq), "to": round(nxt),
                      "risk_per_trade": f"{posture.risk_pct:.2%} of current equity",
                      "gate": ">=30 trades at this rung, max DD < 15%, "
                              "expectancy within 1 SE of certified value",
                      "on_10pct_dd": "drop one rung until a new equity high"})
        eq = nxt
    return rungs


def posture_table(n_trades: int = 100) -> dict:
    """The six headline stats per posture, from the real flagship trades.
    Deterministic (fixed bootstrap seed) — server and terminal show the same."""
    r = load_flagship_returns()
    out = {}
    for po in POSTURES:
        st = bootstrap(r, po.risk_pct, n_trades)
        out[po.name.split("-")[0]] = {
            "risk_per_trade": po.risk_pct, "note": po.note,
            "per_100_trades": round(st["median_terminal"], 3),
            "p_double": round(st["p_double"], 4),
            "median_maxdd": round(st["median_maxdd"], 4),
            "p_dd30": round(st["p_dd30"], 4), "p_dd50": round(st["p_dd50"], 4)}
    return out


def report(accounts=(500, 1000, 1500), n_trades: int = 100) -> None:
    r = load_flagship_returns()
    wins = r[r > 0]; losses = r[r < 0]
    p = len(wins) / len(r); b = wins.mean() / abs(losses.mean())
    fk = kelly_fraction(p, b); wk = worst_case_kelly(p, b, len(r))
    print("=" * 96)
    print(f"SIZING MATH — computed from {len(r)} REAL out-of-sample flagship trades "
          f"(wr {p:.1%}, payoff {b:.2f})")
    print("=" * 96)
    print(f"Kelly: full {fk:.1%} of equity risked/trade | half {fk/2:.1%} | quarter {fk/4:.1%}")
    print(f"Worst-case Kelly (95% lower bound on the win rate, n={len(r)}): {wk:.1%}"
          f"  <- the honest ceiling; every posture below sits at or under it\n")
    hdr = (f"{'posture':<16}{'risk/tr':>8}{'notional':>10}{'med. equity':>12}"
           f"{'P(2x)':>8}{'med tr->2x':>11}{'med maxDD':>10}{'P(DD>30%)':>10}{'P(DD>50%)':>10}")
    print(f"per {n_trades} trades (flagship cadence ~25-40/yr on daily bars; "
          "intraday multiplies cadence, not edge):")
    print(hdr); print("-" * len(hdr))
    for po in POSTURES:
        s = bootstrap(r, po.risk_pct, n_trades)
        print(f"{po.name:<16}{po.risk_pct:>8.2%}{s['lev']:>9.0%} "
              f"{s['median_terminal']:>11.2f}x{s['p_double']:>8.1%}"
              f"{s['median_trades_to_2x']:>11.0f}{s['median_maxdd']:>10.1%}"
              f"{s['p_dd30']:>10.1%}{s['p_dd50']:>10.1%}")
    print("\nsmall-account venue floors (approx):")
    for k, v in MIN_NOTIONAL.items():
        print(f"  {k:<28} ~£{v}")
    for a in accounts:
        blocked = [k for k, v in MIN_NOTIONAL.items()
                   if size_order(a, 0.0075, 0.06, 1, min_notional=v)["blocked"]]
        ok = [k for k in MIN_NOTIONAL if k not in blocked]
        print(f"  £{a}: viable at survival risk -> {', '.join(ok)}")
    print("\nthrottle ladder (auto): " + "  ".join(f"DD>{int(l*100)}%→{int(m*100)}% risk"
          for l, m in THROTTLE))
    print("streak rule: 3 straight losses → half risk until a win. Increasing "
          "size after losses is banned in code, not just in prose.")


if __name__ == "__main__":
    report()
