"""Validation suite — REAL DATA ONLY, still $0.

Run:  python -m qtsys.validate

Every test below runs on real market history (bundled free datasets: WTI,
Brent, Henry Hub natural gas — daily, 1986/1987/1997 to now). No synthetic
price paths exist anywhere in this codebase. Where a test needs "no
information" it randomizes the STRATEGY (entry dates and sides), never the
data; where it needs a defect, it corrupts a throwaway COPY purely to prove
the QC gate refuses it — corrupted copies never feed a backtest.

  T1  Fee accounting is exact ON REAL TRADES: for every trade,
      net = gross - round_trip, to machine precision; totals reconcile.
  T2  No look-ahead: signals computed on real data truncated at T are
      identical to signals computed on the full series, for every bar < T.
  T3  QC gates refuse corrupted copies of the real data
      (negative price, duplicate timestamp, high<low).
  T4  Purged CV on real events: no training trade's lifespan overlaps the
      test fold's calendar span (plus a 30-day embargo) in any split.
  T5  Fees provably flow to P&L: with fees inflated 10x, every real trade's
      net worsens by exactly 9x the base round trip; expectancy drops by it.
  T6  Luck rejection on real prices: an information-free placebo (random
      entry dates, coin-flip sides, real prices) must FAIL the Deflated
      Sharpe gate. A 16-config momentum sweep is then judged by DSR with
      the trial count charged — the verdict is printed either way.
  T7  Broker adapters verified offline against the installed SDKs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BarrierSpec, FeeModel, Trade, equity_curve_real, simulate_trades
from .data import DataQualityError, load_real, quality_check, real_daily_universe
from .metrics import deflated_sharpe, sharpe, trade_stats, verdict
from .select import purged_kfold_splits
from .signals import Event, feature_frame, momentum_events

COMMODITY_FEES = FeeModel(commission_bps=1, half_spread_bps=2, slippage_bps=2)  # 10 bps RT


def _universe():
    return real_daily_universe(("WTI", "BRENT", "NATGAS"))


def _all_momentum_trades(uni, fees=COMMODITY_FEES):
    out = []
    for name, df in uni.items():
        out += simulate_trades(df, momentum_events(df, name), BarrierSpec(),
                               fees, feature_frame(df))
    return out


def t1_fee_accounting():
    uni = _universe()
    trades = _all_momentum_trades(uni)
    assert len(trades) > 100
    for t in trades:
        assert abs(t.net_ret - (t.gross_ret - COMMODITY_FEES.round_trip)) < 1e-12
    tot_net = sum(t.net_ret for t in trades)
    tot_gross = sum(t.gross_ret for t in trades)
    assert abs(tot_net - (tot_gross - len(trades) * COMMODITY_FEES.round_trip)) < 1e-9
    print(f"T1 PASS  fee accounting exact on {len(trades)} REAL trades "
          f"(WTI/Brent/NatGas): net = gross - {COMMODITY_FEES.round_trip:.2%} RT, "
          "to machine precision, totals reconcile")


def t2_no_lookahead():
    df = load_real("WTI")
    full = momentum_events(df, "WTI")
    cut = int(len(df) * 0.6)
    trunc = momentum_events(df.iloc[:cut], "WTI")
    f_early = [(e.i_signal, e.side) for e in full if e.i_signal < cut - 1]
    t_early = [(e.i_signal, e.side) for e in trunc if e.i_signal < cut - 1]
    assert f_early == t_early, "signal changed when future data was removed"
    ff, ft = feature_frame(df), feature_frame(df.iloc[:cut])
    pd.testing.assert_frame_equal(ff.iloc[: cut - 1], ft.iloc[: cut - 1])
    print(f"T2 PASS  no look-ahead on real WTI (1986->now): removing the future "
          f"leaves all {len(t_early)} earlier signals and every feature unchanged")


def t3_qc_gates():
    base = load_real("VIX")            # real OHLC series
    caught = 0
    bad1 = base.copy(); bad1.iloc[100, bad1.columns.get_loc("close")] = -5.0
    bad2 = pd.concat([base, base.iloc[[500]]]).sort_index()
    bad3 = base.copy(); bad3["high"] = bad3["low"] - 1.0
    for broken in (bad1, bad2, bad3):
        try:
            quality_check(broken)
        except DataQualityError:
            caught += 1
    quality_check(base)                # the untouched real data passes
    assert caught == 3
    print("T3 PASS  QC gates refuse all 3 corrupted copies of real VIX data "
          "(negative price, duplicate timestamp, high<low); clean original passes")


def t4_purged_cv():
    trades = _all_momentum_trades(_universe())
    emb = pd.Timedelta(days=30)
    checked = 0
    for tr_idx, te_idx in purged_kfold_splits(trades, 5, 30):
        t0 = min(trades[i].ts_entry for i in te_idx)
        t1 = max(trades[i].ts_exit for i in te_idx)
        for i in tr_idx:
            assert trades[i].ts_exit < t0 or trades[i].ts_entry > t1 + emb
            checked += 1
    print(f"T4 PASS  purged CV on {len(trades)} real events across 3 calendars: "
          f"{checked} train/test pairs checked, zero lifespan overlaps, "
          "30-day embargo respected")


def t5_fees_flow_through():
    uni = _universe()
    fat = FeeModel(commission_bps=10, half_spread_bps=20, slippage_bps=20)  # 100 bps RT
    base_tr = _all_momentum_trades(uni, COMMODITY_FEES)
    fat_tr = _all_momentum_trades(uni, fat)
    assert len(base_tr) == len(fat_tr)
    delta = fat.round_trip - COMMODITY_FEES.round_trip
    for a, b in zip(base_tr, fat_tr):
        assert abs((a.net_ret - b.net_ret) - delta) < 1e-12
    sa, sb = trade_stats(np.array([t.net_ret for t in base_tr])), \
             trade_stats(np.array([t.net_ret for t in fat_tr]))
    print(f"T5 PASS  fees flow to P&L on real trades: 10x fees worsen every "
          f"trade by exactly {delta:.2%}; expectancy {sa.expectancy:+.3%} -> "
          f"{sb.expectancy:+.3%} per trade")


def t6_luck_rejection():
    uni = _universe()
    # Placebo design notes (both matter):
    #  * barriers must be SYMMETRIC (tp = sl): asymmetric barriers let a
    #    coin-flip-side placebo harvest drift — a bias of the TEST, not edge;
    #    with symmetric barriers, short P&L = -(long P&L) per entry exactly
    #    (verified to machine precision), so gross expectation is zero.
    #  * a stochastic null is judged as an ENSEMBLE, never a single draw —
    #    single seeds land in the lucky tail ~3-5% of the time, which is
    #    precisely what a lucky backtest is. So: many seeds, and the gate's
    #    pass-rate must be small.
    sym = BarrierSpec(tp_mult=1.5, sl_mult=1.5, max_hold=15)
    n_seeds, exps, passes = 100, [], 0
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        placebo: list[Trade] = []
        for name, df in uni.items():
            n = len(df)
            idx = np.sort(rng.choice(np.arange(150, n - 30), size=120, replace=False))
            ev = [Event(name, int(i), int(rng.choice([-1, 1])), "placebo") for i in idx]
            placebo += simulate_trades(df, ev, sym, COMMODITY_FEES)
        nets = pd.Series([t.net_ret for t in placebo])
        exps.append(trade_stats(nets.to_numpy()).expectancy)
        # gate on TRADE-LEVEL returns: one observation per trade, honestly
        # independent — daily curves of multi-day trades are autocorrelated
        # and overstate PSR/DSR evidence
        if deflated_sharpe(nets, n_trials=1) >= 0.95:
            passes += 1
    exps = np.array(exps)
    se = exps.std(ddof=1) / np.sqrt(n_seeds)
    print(f"T6a      placebo ensemble on REAL prices ({n_seeds} seeds, random "
          f"dates, coin-flip sides): mean expectancy {exps.mean():+.3%} "
          f"(theory: -fees = {-COMMODITY_FEES.round_trip:.3%}), "
          f"DSR-gate pass rate {passes}/{n_seeds}")
    assert abs(exps.mean() - (-COMMODITY_FEES.round_trip)) < 3 * se, \
        "placebo ensemble mean must equal minus fees"
    assert passes <= 0.10 * n_seeds, "gate must reject nearly all zero-skill runs"
    print("T6a PASS placebo ensemble centred on minus-fees; gate rejects "
          f"{n_seeds - passes}/{n_seeds} zero-skill runs")

    configs = [(f, s) for f in (10, 15, 20, 30) for s in (60, 100, 150, 200)]
    srs, curves = [], []
    for fast, slow in configs:
        tr = []
        for name, df in uni.items():
            tr += simulate_trades(df, momentum_events(df, name, fast, slow),
                                  BarrierSpec(), COMMODITY_FEES)
        c = equity_curve_real(tr, uni)
        curves.append(c); srs.append(sharpe(c))
    best = int(np.argmax(srs))
    var_trials = float(np.var([s / np.sqrt(252) for s in srs], ddof=1))
    dsr_b = deflated_sharpe(curves[best], n_trials=len(configs),
                            var_trials_sr=var_trials)
    print(f"T6b      real-data sweep: best of {len(configs)} momentum configs "
          f"Sharpe {srs[best]:.2f}; DSR (trial count charged) {dsr_b:.3f} -> "
          f"{verdict(dsr_b)}")
    print("T6  PASS luck-rejection machinery verified on real data only")


def t7_adapters():
    from .validate_adapters import run as run_t7
    run_t7()


def main():
    print("=" * 78)
    print("QTSYS VALIDATION SUITE — real data only, $0 cost")
    print("=" * 78)
    t1_fee_accounting()
    t2_no_lookahead()
    t3_qc_gates()
    t4_purged_cv()
    t5_fees_flow_through()
    t6_luck_rejection()
    t7_adapters()
    print("-" * 78)
    print("ALL TESTS PASSED — fee-exact, leak-free, luck-rejecting, on real data.")


if __name__ == "__main__":
    main()
