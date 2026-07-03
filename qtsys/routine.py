"""Pre-market routine (cards 1, 4, 6) — the whole 20-minute checklist in ~5s.

Run:  python -m qtsys.routine

For EVERY asset and every registered strategy it asks one question: is there a
fresh setup on the latest REAL bar? Each hit is then ranked by that exact
(strategy, asset) pair's own out-of-sample track record (registry_results.csv)
— reliability measured as net expectancy with the trial count charged at the
registry level, never as raw win rate. Best evidence first, so the highest-
probability opportunities are acted on before price leaves the entry zone.
Timeframe-agnostic: point `uni` at 1-minute frames locally and nothing changes.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .data import load_real, real_daily_universe
from .signals import realized_vol, rsi, sma
from .strategies import REGISTRY, gate_by_vix, ratio_frame, vix_percentile

HERE = os.path.dirname(__file__)
FRESH_BARS = 2          # a setup counts if it fired within the last N bars


def _attribution() -> pd.DataFrame:
    p = os.path.join(HERE, "registry_results.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(
        columns=["spec", "asset", "n", "win_rate", "expectancy", "profit_factor"])


def _summary() -> pd.DataFrame:
    p = os.path.join(HERE, "registry_summary.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=["id", "dsr"])


def regime(df: pd.DataFrame) -> dict:
    c = df["close"]
    t = "UP" if c.iloc[-1] > sma(c, 200).iloc[-1] else "DOWN"
    fast = "UP" if sma(c, 20).iloc[-1] > sma(c, 100).iloc[-1] else "DOWN"
    vr = float(realized_vol(c, 20).iloc[-1] / realized_vol(c, 100).iloc[-1])
    vol = "HIGH" if vr > 1.35 else "CALM" if vr < 0.8 else "NORMAL"
    return {"trend": t, "fast_trend": fast, "vol": vol, "vol_ratio": round(vr, 2),
            "rsi14": round(float(rsi(c).iloc[-1]), 1)}


def chart_read(sym: str, df: pd.DataFrame | None = None) -> dict:
    """Card 6 — the professional read: structure, levels, and what the chart is
    saying vs what it appears to say at first glance."""
    df = df if df is not None else load_real(sym)
    c = df["close"]; n = len(c)
    win = 5
    look = min(n - win - 1, 250)
    lastpx = float(c.iloc[-1])
    true_hi = [(df.index[i].date(), float(c.iloc[i])) for i in range(n - look, n - win)
               if c.iloc[i] == c.iloc[i - win:i + win + 1].max()]
    true_lo = [(df.index[i].date(), float(c.iloc[i])) for i in range(n - look, n - win)
               if c.iloc[i] == c.iloc[i - win:i + win + 1].min()]
    swings = sorted(true_hi + true_lo)
    swings_hi = [x for x in swings if x[1] > lastpx][-3:]   # levels overhead
    swings_lo = [x for x in swings if x[1] < lastpx][-3:]   # levels below
    sh, sl = true_hi[-2:], true_lo[-2:]
    r = regime(df)
    last = float(c.iloc[-1])
    hi52, lo52 = float(c.rolling(252).max().iloc[-1]), float(c.rolling(252).min().iloc[-1])
    structure = ("higher-highs/higher-lows" if len(sh) > 1 and len(sl) > 1
                 and sh[-1][1] > sh[-2][1] and sl[-1][1] > sl[-2][1]
                 else "lower-highs/lower-lows" if len(sh) > 1 and len(sl) > 1
                 and sh[-1][1] < sh[-2][1] and sl[-1][1] < sl[-2][1]
                 else "range / transition")
    surface = f"last move {'up' if c.iloc[-1] > c.iloc[-2] else 'down'} {abs(c.iloc[-1]/c.iloc[-2]-1):.2%}"
    deeper = (f"{r['trend']} regime, {structure}, vol {r['vol']} — the last candle is "
              f"{'with' if (r['trend']=='UP')==(c.iloc[-1]>c.iloc[-2]) else 'AGAINST'} the regime; "
              f"untrained eyes read the candle, the regime pays the bills")
    return {"symbol": sym, "as_of": str(df.index[-1].date()), "last": last,
            "regime": r, "structure": structure,
            "resistance": [f"{d} @ {v:.5g}" for d, v in swings_hi],
            "support": [f"{d} @ {v:.5g}" for d, v in swings_lo],
            "52w": f"{lo52:.5g} – {hi52:.5g} (now {(last-lo52)/(hi52-lo52+1e-12):.0%} of range)",
            "surface_read": surface, "deeper_read": deeper}


def scan(uni: dict[str, pd.DataFrame] | None = None, top: int = 10) -> pd.DataFrame:
    """Card 1+4: every strategy × every asset, fresh setups only, ranked by
    that pair's own real out-of-sample expectancy (fallback: spec-level)."""
    uni = uni or real_daily_universe()
    att, summ = _attribution(), _summary()
    vixp = vix_percentile(load_real("VIX"))
    spec_dsr = dict(zip(summ.get("id", []), summ.get("dsr", [])))
    spec_exp = dict(zip(summ.get("id", []), summ.get("test_exp", [])))
    hits = []
    for spec in REGISTRY:
        try:
            if spec.kind == "single":
                pairs = [(s, spec.fn(uni[s], s, **spec.params), uni[s]) for s in uni]
            elif spec.kind == "vixgate":
                pairs = [(s, gate_by_vix(spec.fn(uni[s], s), uni[s], vixp,
                                         max_pct=spec.params.get("max_pct")), uni[s]) for s in uni]
            elif spec.kind == "vixcontra":
                pairs = [(s, spec.fn(uni[s], s, vixp, min_pct=spec.params["min_pct"]),
                          uni[s]) for s in spec.assets]
            elif spec.kind == "pair":
                a, b = spec.assets
                dfr = ratio_frame(uni[a], uni[b])
                pairs = [(f"{a}/{b}", spec.fn(dfr, f"{a}/{b}", **spec.params), dfr)]
            else:
                continue                      # xsec scans on its own cadence
        except Exception:
            continue
        for s, evs, df in pairs:
            fresh = [e for e in evs if e.i_signal >= len(df) - FRESH_BARS]
            for e in fresh:
                row = att[(att["spec"] == spec.id) & (att["asset"] == s)]
                exp = float(row["expectancy"].iloc[0]) if len(row) else float(spec_exp.get(spec.id, np.nan))
                nn = int(row["n"].iloc[0]) if len(row) else 0
                dsr = float(spec_dsr.get(spec.id, np.nan))
                hits.append({"asset": s, "strategy": spec.id, "family": spec.family,
                             "side": "LONG" if e.side > 0 else "SHORT",
                             "signal_date": str(df.index[e.i_signal].date()),
                             "hist_exp": exp, "hist_n": nn, "spec_dsr": dsr,
                             "tier": ("SURVIVOR" if dsr >= 0.95 else
                                      "CANDIDATE" if dsr >= 0.80 else "WATCH-ONLY")})
    out = pd.DataFrame(hits)
    if len(out):
        rank = {"SURVIVOR": 0, "CANDIDATE": 1, "WATCH-ONLY": 2}
        out["_r"] = out["tier"].map(rank)
        out = (out.sort_values(["_r", "hist_exp"], ascending=[True, False])
                  .drop(columns="_r").head(top))
    return out


def morning_briefing() -> str:
    uni = real_daily_universe()
    lines = ["MORNING BRIEFING — real data as of " +
             max(str(df.index[-1].date()) for df in uni.values())]
    lines.append("\nREGIMES:")
    for s, df in uni.items():
        r = regime(df)
        lines.append(f"  {s:7s} {r['trend']:4s} trend | fast {r['fast_trend']:4s} | "
                     f"vol {r['vol']:6s} ({r['vol_ratio']}) | RSI {r['rsi14']}")
    sc = scan(uni)
    lines.append("\nRANKED SETUPS (best real-data evidence first — act top-down):")
    if len(sc) == 0:
        lines.append("  none fresh today — standing aside IS a position")
    else:
        for _, h in sc.iterrows():
            lines.append(f"  [{h['tier']:9s}] {h['side']:5s} {h['asset']:9s} via {h['strategy']:15s} "
                         f"hist exp {h['hist_exp']:+.2%}/trade (n={h['hist_n']}, spec DSR {h['spec_dsr']:.2f})")
    lines.append("\nRule: SURVIVOR tier is tradable at posture risk; CANDIDATE at half; "
                 "WATCH-ONLY is never traded, only journaled.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(morning_briefing())
    print("\n--- example chart read (card 6) ---")
    for k, v in chart_read("WTI").items():
        print(f"{k:12s}: {v}")
