"""tracking.py — backtest baseline vs REAL executed performance.

The posture bar's distribution stats (per_100_trades, p_double, drawdowns) come
from `sizing.posture_table()` — a bootstrap of the flagship's REAL out-of-sample
per-trade net returns. That is the *backtest* the account is underwritten on.

This module answers the go-live question the blueprint's checklist actually
gates on: **is the live book tracking that backtest, or drifting from it?** It
reconstructs realised round-trips from the venue's ACTUAL fills, computes the
same per-trade stats (metrics.trade_stats) on them, and reports the difference —
including whether live expectancy sits within one standard error of the
certified backtest value (the "within 1 SE" gate in skill 10).

Nothing here is simulated: the baseline is real historical trades, the live side
is real fills. If there are no live trades yet, `live` is null and the terminal
shows the baseline alone with "0 live trades — need >=30 for a verdict".
"""
from __future__ import annotations

import math

import numpy as np

from .metrics import trade_stats
from .sizing import load_flagship_returns

# Round-trip cost by asset class, matched to backtest.py's fee models so realised
# net returns are charged the SAME way the certified baseline was.
_CLS_RT_FEE = {
    "Crypto": 0.0030,        # CRYPTO_FEES ~30 bps RT (taker-dominated)
    "Commodity": 0.0010,     # futures/CFD round-trip
    "FX": 0.0004,            # tight spot spreads
}
_DEFAULT_RT_FEE = 0.0006     # US_EQUITY_FEES ~6 bps RT
MIN_LIVE_TRADES = 30         # skill 10: >=30 trades at a rung before it counts


def _stats_dict(net: np.ndarray) -> dict:
    ts = trade_stats(net)
    n = int(ts.n)
    # per-trade expectancy standard error — the yardstick for "within 1 SE"
    se = float(np.std(net, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return {"n": n, "win_rate": _f(ts.win_rate), "avg_win": _f(ts.avg_win),
            "avg_loss": _f(ts.avg_loss), "expectancy": _f(ts.expectancy),
            "expectancy_se": se if not math.isnan(se) else None,
            "profit_factor": _f(ts.profit_factor), "total_net": _f(ts.total_net)}


def _f(x) -> float | None:
    x = float(x)
    return None if (math.isnan(x) or math.isinf(x)) else x


def backtest_baseline() -> dict:
    """Certified per-trade stats from the REAL out-of-sample flagship trades."""
    return _stats_dict(load_flagship_returns())


# ---------------------------------------------------------------- realised side
def realised_returns(fills: list[dict], cls_of) -> np.ndarray:
    """FIFO/VWAP-pair a normalised fill stream into per-trade NET returns.

    `fills`: dicts with symbol, side ('buy'|'sell'), qty, price (+ optional ts).
    `cls_of(symbol) -> str`: asset class, for the round-trip fee charge.
    A round-trip is netted by its class RT fee so it is comparable to the
    fee-net backtest baseline. Returns an array of realised net returns.
    """
    by_sym: dict[str, list[dict]] = {}
    for f in fills:
        by_sym.setdefault(f["symbol"], []).append(f)
    out: list[float] = []
    for sym, fs in by_sym.items():
        fee = _CLS_RT_FEE.get(cls_of(sym), _DEFAULT_RT_FEE)
        fs = sorted(fs, key=lambda x: x.get("ts", 0))
        pos = 0.0          # signed position
        entry = 0.0        # VWAP of the open position
        for f in fs:
            q = float(f["qty"]) * (1.0 if f["side"] == "buy" else -1.0)
            price = float(f["price"])
            if pos == 0.0 or (pos > 0) == (q > 0):        # open / add
                tot = abs(pos) + abs(q)
                entry = (entry * abs(pos) + price * abs(q)) / tot if tot else price
                pos += q
            else:                                          # reduce / close / flip
                closing = min(abs(q), abs(pos))
                direction = 1.0 if pos > 0 else -1.0
                gross = direction * (price / entry - 1.0)
                out.append(gross - fee)                    # one realised trade
                pos += q
                if (pos > 0) != (direction > 0) and abs(pos) > 1e-12:
                    entry = price                          # flipped: new position
    return np.asarray(out, dtype=float)


def realised_roundtrips(fills: list[dict], cls_of) -> list[dict]:
    """Same FIFO/VWAP pairing as realised_returns, but returns the DETAILED
    closed round-trips (for the 'closed positions' view): symbol, side, qty,
    entry & exit price, P&L in $ and %, timestamps. Newest close first."""
    by_sym: dict[str, list[dict]] = {}
    for f in fills:
        by_sym.setdefault(f["symbol"], []).append(f)
    trips: list[dict] = []
    for sym, fs in by_sym.items():
        fee = _CLS_RT_FEE.get(cls_of(sym), _DEFAULT_RT_FEE)
        fs = sorted(fs, key=lambda x: x.get("ts", 0))
        pos = 0.0
        entry = 0.0
        entry_ts = 0.0
        for f in fs:
            q = float(f["qty"]) * (1.0 if f["side"] == "buy" else -1.0)
            price = float(f["price"])
            if pos == 0.0 or (pos > 0) == (q > 0):        # open / add
                tot = abs(pos) + abs(q)
                entry = (entry * abs(pos) + price * abs(q)) / tot if tot else price
                if pos == 0.0:
                    entry_ts = f.get("ts", 0.0)
                pos += q
            else:                                          # reduce / close / flip
                closing = min(abs(q), abs(pos))
                direction = 1.0 if pos > 0 else -1.0
                gross = direction * (price / entry - 1.0)
                pnl = direction * (price - entry) * closing
                trips.append({
                    "symbol": sym, "side": "long" if direction > 0 else "short",
                    "qty": round(closing, 6), "entry_px": round(entry, 4),
                    "exit_px": round(price, 4),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round((gross - fee) * 100, 3),
                    "opened_ts": entry_ts, "closed_ts": f.get("ts", 0.0)})
                pos += q
                if (pos > 0) != (direction > 0) and abs(pos) > 1e-12:
                    entry = price
                    entry_ts = f.get("ts", 0.0)
    trips.sort(key=lambda t: t["closed_ts"], reverse=True)
    return trips


# ------------------------------------------------------------------- comparison
def compare(baseline: dict, live: dict | None) -> dict:
    """Backtest-vs-live deltas and the within-1-SE verdict."""
    if not live or live["n"] == 0:
        return {"live_trades": 0, "verdict": "no live trades yet",
                "within_1se": None,
                "note": f"need >={MIN_LIVE_TRADES} live round-trips for a "
                        "statistically meaningful comparison"}
    d_exp = live["expectancy"] - baseline["expectancy"]
    d_wr = (live["win_rate"] or 0) - (baseline["win_rate"] or 0)
    se = live.get("expectancy_se") or baseline.get("expectancy_se")
    z = (d_exp / se) if se else None
    within = (abs(z) <= 1.0) if z is not None else None
    if live["n"] < MIN_LIVE_TRADES:
        verdict = f"early — {live['n']}/{MIN_LIVE_TRADES} trades"
    elif within:
        verdict = "on-model (within 1 SE of backtest)"
    else:
        verdict = "DRIFT — live expectancy >1 SE from backtest"
    return {"live_trades": live["n"],
            "expectancy_diff": _f(d_exp), "win_rate_diff": _f(d_wr),
            "expectancy_z": _f(z) if z is not None else None,
            "within_1se": within, "verdict": verdict}


def tracking_report(fills: list[dict], cls_of) -> dict:
    """Everything the terminal's BACKTEST-vs-LIVE panel needs."""
    baseline = backtest_baseline()
    net = realised_returns(fills, cls_of)
    live = _stats_dict(net) if net.size else None
    return {"backtest": baseline, "live": live,
            "tracking": compare(baseline, live)}
