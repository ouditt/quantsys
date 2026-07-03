"""Trade selection — the layer that "picks the right trades".

Method: META-LABELING. The primary strategy chooses the SIDE of each candidate
trade; a second model is trained on whether such trades were historically
profitable NET OF FEES, and only candidates with P(win) above a threshold are
taken. This is the honest way to raise win rate: skip likely losers, rather
than distort payoffs (the dishonest way — tiny targets, giant stops — prints a
high win rate and then blows up).

Leakage controls (non-negotiable):
  * PurgedKFold — folds are chronological; any training event whose life
    [entry, exit] overlaps the test fold's time span is PURGED, and an embargo
    strip after the fold is dropped too. Plain k-fold on overlapping trade
    labels leaks the answer and manufactures fake skill.
  * the probability threshold is chosen ONLY on training-period out-of-fold
    predictions (maximising net expectancy subject to a minimum-coverage
    constraint), then frozen before it ever sees the evaluation window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .backtest import Trade
from .metrics import trade_stats

FEATURES = ["trend_z", "mom60_z", "rsi14", "vol_ratio", "vol_z", "dd_52w", "acorr", "side"]


def design_matrix(trades: list[Trade]):
    rows, y, keep = [], [], []
    for k, t in enumerate(trades):
        if not t.features:
            continue
        rows.append([t.features.get(f, 0.0) if f != "side" else float(t.side)
                     for f in FEATURES])
        y.append(1 if t.net_ret > 0 else 0)          # label = win NET of fees
        keep.append(k)
    return np.array(rows), np.array(y), keep


def purged_kfold_splits(trades: list[Trade], n_splits: int = 5, embargo_days: int = 30):
    """Chronological folds over events, purged + embargoed by REAL calendar time
    (trade lifespans [ts_entry, ts_exit]); correct across assets whose trading
    calendars differ."""
    import pandas as pd
    order = np.argsort([t.ts_entry.value for t in trades])
    folds = np.array_split(order, n_splits)
    emb = pd.Timedelta(days=embargo_days)
    for te in folds:
        t0 = min(trades[i].ts_entry for i in te)
        t1 = max(trades[i].ts_exit for i in te)
        tr = [i for i in order
              if (trades[i].ts_exit < t0 or trades[i].ts_entry > t1 + emb)]
        yield np.array(tr), np.array(te)


@dataclass
class TradeSelector:
    min_coverage: float = 0.30       # never keep < 30% of trades (guards degeneracy)
    n_splits: int = 5
    embargo_days: int = 30
    threshold_: float = 0.5
    model_: GradientBoostingClassifier | None = None
    cv_report_: dict | None = None

    def fit(self, train_trades: list[Trade]) -> "TradeSelector":
        X, y, keep = design_matrix(train_trades)
        kept = [train_trades[k] for k in keep]
        oof = np.full(len(kept), np.nan)
        for tr_idx, te_idx in purged_kfold_splits(kept, self.n_splits, self.embargo_days):
            m = self._new_model().fit(X[tr_idx], y[tr_idx])
            oof[te_idx] = m.predict_proba(X[te_idx])[:, 1]
        ok = ~np.isnan(oof)
        nets = np.array([t.net_ret for t in kept])
        # pick the threshold on OOF predictions only: max net expectancy s.t. coverage
        best_th, best_exp = 0.5, -np.inf
        for th in np.arange(0.40, 0.76, 0.01):
            sel = ok & (oof >= th)
            if sel.sum() < max(self.min_coverage * ok.sum(), 25):
                continue
            e = trade_stats(nets[sel]).expectancy
            if e > best_exp:
                best_exp, best_th = e, th
        self.threshold_ = best_th
        self.model_ = self._new_model().fit(X, y)
        base = trade_stats(nets[ok])
        filt = trade_stats(nets[ok & (oof >= best_th)])
        self.cv_report_ = {"oof_base": base, "oof_filtered": filt,
                           "threshold": best_th}
        return self

    def take(self, trade: Trade) -> bool:
        """Decision for a single candidate trade (features at signal time only)."""
        if self.model_ is None or not trade.features:
            return False
        x = np.array([[trade.features.get(f, 0.0) if f != "side" else float(trade.side)
                       for f in FEATURES]])
        return float(self.model_.predict_proba(x)[0, 1]) >= self.threshold_

    def filter(self, trades: list[Trade]) -> list[Trade]:
        return [t for t in trades if self.take(t)]

    @staticmethod
    def _new_model():
        return GradientBoostingClassifier(n_estimators=150, max_depth=2,
                                          learning_rate=0.05, subsample=0.8,
                                          random_state=0)
