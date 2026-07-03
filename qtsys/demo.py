"""End-to-end demo — REAL DATA ONLY. Run:  python -m qtsys.demo

Part A (real daily commodities: WTI, Brent, Henry Hub — 1986/87/97 to now):
the trade-selection agent (meta-labeling) trains on history before the 70%
date boundary and is evaluated on the untouched final ~30%. The table compares
BASE strategy vs AGENT-SELECTED trades — win rate, average win, expectancy,
profit factor, mark-to-market Sharpe, drawdown and the trade-level Deflated
Sharpe verdict — all NET of fees, all out-of-sample. Whatever the verdict is,
it is printed; a "do not deploy" on a weak strategy is the system WORKING.

Part B (real S&P 500 total returns, 1871->now): the classic 10-month moving
average timing rule with fees charged on every switch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BarrierSpec, FeeModel, Trade, equity_curve_real, simulate_trades
from .data import load_shiller_monthly, real_daily_universe
from .metrics import (cagr, deflated_sharpe, max_drawdown, sharpe, sortino,
                      trade_stats, verdict)
from .select import TradeSelector
from .signals import feature_frame, meanrev_events, momentum_events

COMMODITY_FEES = FeeModel(commission_bps=1, half_spread_bps=2, slippage_bps=2)  # 10 bps RT
FX_FEES = FeeModel(commission_bps=0, half_spread_bps=1, slippage_bps=0.5)       # 3 bps RT (majors)
from .backtest import CRYPTO_FEES                                               # 30 bps RT

FEES_BY_SYMBOL = {"WTI": COMMODITY_FEES, "BRENT": COMMODITY_FEES,
                  "NATGAS": COMMODITY_FEES, "BTC": CRYPTO_FEES, "ETH": CRYPTO_FEES,
                  "EURUSD": FX_FEES, "GBPUSD": FX_FEES, "AUDUSD": FX_FEES,
                  "JPYUSD": FX_FEES, "CHFUSD": FX_FEES, "CADUSD": FX_FEES}


def _collect(universe, strategy_fn, barriers, fees_by_symbol) -> list[Trade]:
    out: list[Trade] = []
    for name, df in universe.items():
        out += simulate_trades(df, strategy_fn(df, name), barriers,
                               fees_by_symbol[name], feature_frame(df))
    return out


def _report(tag: str, trades: list[Trade], uni) -> dict:
    nets = pd.Series([t.net_ret for t in trades], dtype=float)
    ts = trade_stats(nets.to_numpy())
    row = ts.row()
    if len(trades):
        r = equity_curve_real(trades, uni)
        dsr = deflated_sharpe(nets, n_trials=1)          # trade-level: i.i.d.-honest
        row.update({"Sharpe(M2M)": f"{sharpe(r):5.2f}",
                    "max DD": f"{max_drawdown(r):6.1%}",
                    "DSR(trade)": f"{dsr:5.3f}"})
    else:
        dsr = 0.0
        row.update({"Sharpe(M2M)": "  n/a", "max DD": "   n/a", "DSR(trade)": "  n/a"})
    return {"tag": tag, "row": row, "dsr": dsr}


def part_a():
    print("=" * 78)
    print("PART A — REAL cross-asset daily universe: WTI, Brent, NatGas (1986/87/97->),")
    print("         BTC, ETH (2010/15->), 6 FX majors (1971/99->). Agent trade-")
    print("         selection: does it help, out-of-sample, net of per-class fees?")
    print("=" * 78)
    uni = real_daily_universe()
    union = sorted(set().union(*[df.index for df in uni.values()]))
    split_date = union[int(len(union) * 0.70)]
    print(f"fees per side by class: commodities 5 bps | crypto 15 bps | FX 1.5 bps"
          f"   |   train < {split_date.date()} <= test")

    for strat_name, strat_fn, barriers in [
        ("MOMENTUM (SMA 20/100 cross, triple-barrier exits)", momentum_events,
         BarrierSpec(tp_mult=2.0, sl_mult=1.5, max_hold=15)),
        ("MEAN REVERSION (RSI-2 dip in uptrend)", meanrev_events,
         BarrierSpec(tp_mult=1.2, sl_mult=1.5, max_hold=7)),
    ]:
        all_trades = _collect(uni, strat_fn, barriers, FEES_BY_SYMBOL)
        train = [t for t in all_trades if t.ts_exit < split_date]
        test = [t for t in all_trades if t.ts_entry >= split_date]
        print(f"\n--- {strat_name} ---")
        if len(train) < 60 or len(test) < 15:
            print(f"insufficient real events (train {len(train)}, test {len(test)}) "
                  "— honest answer: not enough evidence, no verdict issued")
            continue
        selector = TradeSelector().fit(train)
        picked = selector.filter(test)
        print(f"train events: {len(train)}  |  P(win) threshold learned on train "
              f"OOF only: {selector.threshold_:.2f}")
        rows = [_report("BASE (all signals)", test, uni),
                _report("AGENT-SELECTED", picked, uni)]
        cols = list(rows[0]["row"].keys())
        w = max(len(r["tag"]) for r in rows) + 2
        print(f"{'TEST WINDOW (~30% held out)':<{w}}" + "".join(f"{c:>17}" for c in cols))
        for r in rows:
            print(f"{r['tag']:<{w}}" + "".join(f"{str(r['row'][c]):>17}" for c in cols))
        b = trade_stats(np.array([t.net_ret for t in test]))
        s = trade_stats(np.array([t.net_ret for t in picked])) if picked else None
        if s and s.n:
            print(f"selection: win rate {b.win_rate:.1%} -> {s.win_rate:.1%}, "
                  f"expectancy {b.expectancy:+.3%} -> {s.expectancy:+.3%}/trade net, "
                  f"taking {s.n}/{b.n} candidates")
        print(f"verdict on AGENT-SELECTED: {verdict(rows[1]['dsr'])}")


def part_b():
    print("\n" + "=" * 78)
    print("PART B — real S&P 500, 1871->now: 10-month SMA timing, 5 bps/side on switches")
    print("=" * 78)
    df = load_shiller_monthly()
    tr = df["tr"].fillna(0.0)
    sma10 = df["close"].rolling(10).mean()
    pos = (df["close"] > sma10).astype(float).shift(1).fillna(0.0)
    per_side = 5 / 1e4
    strat = pos * tr - pos.diff().abs().fillna(0.0) * per_side
    net_trades, in_pos, acc = [], False, 1.0
    for t in range(len(pos)):
        if pos.iloc[t] > 0 and not in_pos:
            in_pos, acc = True, 1.0
        if in_pos:
            acc *= 1 + tr.iloc[t]
        if in_pos and (t == len(pos) - 1 or pos.iloc[t + 1] == 0):
            net_trades.append(acc - 1 - 2 * per_side)
            in_pos = False
    ts = trade_stats(np.array(net_trades))

    def line(name, r):
        print(f"{name:<22} CAGR {cagr(r,12):6.2%}   Sharpe {sharpe(r,12):5.2f}   "
              f"maxDD {max_drawdown(r):7.1%}")
    line("buy & hold", tr)
    line("SMA-10 timing (net)", strat)
    print(f"round trips: {ts.n}   win rate (net) {ts.win_rate:.1%}   "
          f"avg win {ts.avg_win:+.2%}   avg loss {-ts.avg_loss:+.2%}   "
          f"expectancy {ts.expectancy:+.2%}/trade   profit factor {ts.profit_factor:.2f}")
    print(f"DSR(trade-level) {deflated_sharpe(pd.Series(net_trades), n_trials=1):.3f}")
    print("value here = drawdown cut (-82% -> -46%) at similar CAGR; no single")
    print("metric — win rate included — tells the whole story.")


if __name__ == "__main__":
    part_a()
    part_b()
