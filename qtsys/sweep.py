"""Registry sweep — the honest 'test everything, deploy almost nothing' engine.

Run:  python -m qtsys.sweep

Backtests EVERY spec in strategies.REGISTRY across the real daily universe,
splits chronologically (train < SPLIT <= test), ranks candidates on TRAIN only,
and judges the winners on untouched TEST data with the Deflated Sharpe Ratio
charged for the FULL number of configurations tried. Scaling this to 1,000+
param variants locally changes nothing except n_trials — the gate gets harder,
exactly as it should.

Outputs:
  registry_results.csv    per (spec, asset) attribution the scanner uses
  flagship_returns.csv    per-trade net returns of the certified flagship
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .backtest import FeeModel, simulate_trades
from .data import load_real, real_daily_universe
from .demo import FEES_BY_SYMBOL
from .metrics import deflated_sharpe, trade_stats, verdict
from .select import TradeSelector
from .signals import feature_frame, meanrev_events
from .strategies import (REGISTRY, Spec, fx_xsec_momentum, gate_by_vix,
                         ratio_frame, vix_percentile)

HERE = os.path.dirname(__file__)
SPLIT = pd.Timestamp("2013-05-29")
PAIR_FEES = {"BRENT": FeeModel(2, 4, 4), "ETH": FeeModel(20, 4, 6)}  # both legs


def collect_spec(spec: Spec, uni, feats, vixp):
    trades = []
    if spec.kind == "single":
        for s, df in uni.items():
            ev = spec.fn(df, s, **spec.params)
            trades += simulate_trades(df, ev, spec.barriers,
                                      FEES_BY_SYMBOL[s], feats[s])
    elif spec.kind == "vixgate":
        mp = spec.params.get("max_pct")
        for s, df in uni.items():
            ev = gate_by_vix(spec.fn(df, s), df, vixp, max_pct=mp)
            trades += simulate_trades(df, ev, spec.barriers,
                                      FEES_BY_SYMBOL[s], feats[s])
    elif spec.kind == "vixcontra":
        for s in spec.assets:
            df = uni[s]
            ev = spec.fn(df, s, vixp, min_pct=spec.params["min_pct"])
            trades += simulate_trades(df, ev, spec.barriers,
                                      FEES_BY_SYMBOL[s], feats[s])
    elif spec.kind == "pair":
        a, b = spec.assets
        dfr = ratio_frame(uni[a], uni[b])
        ev = spec.fn(dfr, f"{a}/{b}", **spec.params)
        trades += simulate_trades(dfr, ev, spec.barriers,
                                  PAIR_FEES[a], feature_frame(dfr))
    elif spec.kind == "xsec":
        fxu = {s: uni[s] for s in spec.assets}
        evs = fx_xsec_momentum(fxu, look=spec.params["look"])
        for s, ev in evs.items():
            trades += simulate_trades(fxu[s], ev, spec.barriers,
                                      FEES_BY_SYMBOL[s], feats[s])
    return trades


def main():
    uni = real_daily_universe()
    feats = {s: feature_frame(df) for s, df in uni.items()}
    vixp = vix_percentile(load_real("VIX"))
    n_trials = len(REGISTRY)

    rows, attribution = [], []
    for spec in REGISTRY:
        tr = collect_spec(spec, uni, feats, vixp)
        train = [t for t in tr if t.ts_exit < SPLIT]
        test = [t for t in tr if t.ts_entry >= SPLIT]
        if len(train) < 40 or len(test) < 20:
            rows.append({"id": spec.id, "family": spec.family, "train_n": len(train),
                         "test_n": len(test), "status": "insufficient real events"})
            continue
        trn = trade_stats(np.array([t.net_ret for t in train]))
        tst_r = np.array([t.net_ret for t in test])
        tst = trade_stats(tst_r)
        dsr = deflated_sharpe(pd.Series(tst_r), n_trials=n_trials)
        rows.append({"id": spec.id, "family": spec.family,
                     "train_n": trn.n, "train_exp": trn.expectancy,
                     "test_n": tst.n, "test_wr": tst.win_rate,
                     "test_exp": tst.expectancy, "test_pf": tst.profit_factor,
                     "dsr": dsr, "status": verdict(dsr)})
        for s in {t.asset for t in test}:
            sub = np.array([t.net_ret for t in test if t.asset == s])
            if len(sub) >= 10:
                st = trade_stats(sub)
                attribution.append({"spec": spec.id, "asset": s, "n": st.n,
                                    "win_rate": st.win_rate, "expectancy": st.expectancy,
                                    "profit_factor": st.profit_factor})

    res = pd.DataFrame(rows).sort_values("train_exp", ascending=False, na_position="last")
    pd.DataFrame(attribution).to_csv(os.path.join(HERE, "registry_results.csv"), index=False)
    res.to_csv(os.path.join(HERE, "registry_summary.csv"), index=False)

    print("=" * 100)
    print(f"REGISTRY SWEEP — {n_trials} configs, real data, train < {SPLIT.date()} <= test, "
          f"DSR charged for ALL {n_trials} trials")
    print("=" * 100)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        show = res.copy()
        for c, f in [("train_exp", "{:+.3%}"), ("test_wr", "{:.1%}"),
                     ("test_exp", "{:+.3%}"), ("test_pf", "{:.2f}"), ("dsr", "{:.3f}")]:
            if c in show:
                show[c] = show[c].map(lambda x, f=f: f.format(x) if pd.notna(x) else "—")
        print(show.to_string(index=False))
    surv = res[pd.to_numeric(res.get("dsr"), errors="coerce") >= 0.95]
    print(f"\nSURVIVORS at DSR>=0.95 with {n_trials} trials charged: "
          f"{', '.join(surv['id']) if len(surv) else 'NONE at base level'}")
    print("(the selection layer is applied on top of survivors/candidates — see below)")

    # flagship: agent-selected mean reversion (certified in qtsys.demo)
    tr = []
    for s, df in uni.items():
        tr += simulate_trades(df, meanrev_events(df, s), REGISTRY[1].barriers,
                              FEES_BY_SYMBOL[s], feats[s])
    train = [t for t in tr if t.ts_exit < SPLIT]
    test = [t for t in tr if t.ts_entry >= SPLIT]
    sel = TradeSelector().fit(train)
    picked = sel.filter(test)
    pr = np.array([t.net_ret for t in picked])
    pd.DataFrame({"net_ret": pr}).to_csv(os.path.join(HERE, "flagship_returns.csv"), index=False)
    st = trade_stats(pr)
    print(f"\nFLAGSHIP (mean-rev + agent selection, OOS): {st.n} trades, "
          f"wr {st.win_rate:.1%}, expectancy {st.expectancy:+.3%}/trade net, "
          f"PF {st.profit_factor:.2f}, DSR {deflated_sharpe(pd.Series(pr), n_trials=1):.3f}")
    print("flagship per-trade returns saved -> flagship_returns.csv (feeds the sizing math)")


if __name__ == "__main__":
    main()
