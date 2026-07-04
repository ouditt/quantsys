"""selector.py — ML that narrows the morning scan over time.

Learns which instruments are worth the full strategy grid, so after a warm-up the
scan runs on a high-value shortlist and gets ~10x faster.

Phased protocol (the discipline that earns the narrowing):
  warmup  : scan the FULL universe daily, logging per-instrument features + forward
            outcomes. No narrowing. (until WARMUP_DAYS distinct days logged)
  shadow  : train the model, run it alongside the full scan, measure recall@K on
            held-out days. Narrowing NOT yet applied.
  active  : once the gate passes (recall@K >= GATE_RECALL), the scan runs the grid
            only on the model's shortlist + a rotating exploration quota.

Auto-demotes back to warmup if live recall decays. Everything is logged and
inspectable (feature importances) so the ML's focus is auditable.

Model: gradient-boosted trees (same family as the trade meta-labeler in select.py).
Label: did the instrument's fresh setup realise a positive forward move (net of a
fee floor) over the next K bars? Forward-looking; time-ordered split (production
upgrade: purged/embargoed CV like select.py).
"""
from __future__ import annotations

import json
import os
import sqlite3
import time

import numpy as np

HERE = os.path.dirname(__file__)
DB = os.path.join(HERE, "universe_selector.db")

WARMUP_DAYS = 180          # distinct scan-days before we may leave warmup
FWD_K = 10                 # bars ahead to score the setup
FEE_FLOOR = 0.001          # a move must clear this to count as "worthwhile"
GATE_RECALL = 0.90         # top-K must capture this share of good setups
SHORTLIST_K = 500
EXPLORE_FRAC = 0.10        # always scan this fraction at random (never go blind)

_FEATS = ["dollar_vol", "rvol_20", "rvol_ratio", "mom_63", "mom_252",
          "dist_52w_high", "n_signals_total", "bars"]


def _db():
    d = sqlite3.connect(DB)
    d.execute("""CREATE TABLE IF NOT EXISTS feat(
        date TEXT, asset TEXT, feats TEXT, side INTEGER, had_setup INTEGER,
        label INTEGER, PRIMARY KEY(date, asset))""")
    d.execute("CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT)")
    return d


def _get(d, k, default=None):
    r = d.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    return json.loads(r[0]) if r else default


def _set(d, k, v):
    d.execute("INSERT OR REPLACE INTO kv VALUES(?,?)", (k, json.dumps(v)))


# ------------------------------------------------------------------- logging
def log_scan(date: str, features: list[dict], setups: list[dict]) -> None:
    """Record one scan-day: per-instrument features + whether it produced a fresh
    setup (and the dominant side). Labels are filled in later by update_labels()."""
    side_of: dict[str, int] = {}
    for s in setups:
        side_of[s["asset"]] = side_of.get(s["asset"], 0) + (1 if s["side"] == "LONG" else -1)
    d = _db()
    for f in features:
        a = f["asset"]
        had = 1 if a in side_of else 0
        side = int(np.sign(side_of.get(a, 0)))
        d.execute("INSERT OR REPLACE INTO feat(date,asset,feats,side,had_setup,label)"
                  " VALUES(?,?,?,?,?,NULL)",
                  (date, a, json.dumps({k: f.get(k, 0.0) for k in _FEATS}), side, had))
    days = d.execute("SELECT COUNT(DISTINCT date) FROM feat").fetchone()[0]
    _set(d, "days_logged", days)
    d.commit(); d.close()


def update_labels(bars_by_asset: dict) -> int:
    """Fill labels for logged rows now old enough to score: label=1 if the
    instrument had a fresh setup whose forward K-bar move (in the setup's
    direction) cleared the fee floor."""
    import pandas as pd
    d = _db()
    rows = d.execute("SELECT date,asset,side,had_setup FROM feat WHERE label IS NULL").fetchall()
    n = 0
    for date, asset, side, had in rows:
        df = bars_by_asset.get(asset)
        if df is None or asset not in bars_by_asset:
            continue
        try:
            idx = df.index.get_indexer([pd.Timestamp(date)], method="nearest")[0]
        except Exception:
            continue
        if idx < 0 or idx + FWD_K >= len(df):
            continue                      # not enough forward bars yet
        c = df["close"]
        fwd = c.iloc[idx + FWD_K] / c.iloc[idx] - 1.0
        good = 1 if (had and side != 0 and side * fwd > FEE_FLOOR) else 0
        d.execute("UPDATE feat SET label=? WHERE date=? AND asset=?", (good, date, asset))
        n += 1
    d.commit(); d.close()
    return n


# ------------------------------------------------------------------- training
def train() -> dict:
    """Train the selector on labelled rows; time-ordered holdout for recall@K.
    Advances the phase when the gate passes. Returns status."""
    import pandas as pd
    d = _db()
    rows = d.execute("SELECT date,asset,feats,label FROM feat WHERE label IS NOT NULL").fetchall()
    if len(rows) < 500 or len({r[0] for r in rows}) < 8:
        d.close()
        return status()                 # not enough labelled history yet
    df = pd.DataFrame([{"date": r[0], "asset": r[1], **json.loads(r[2]),
                        "label": r[3]} for r in rows])
    dates = sorted(df["date"].unique())
    cut = dates[int(len(dates) * 0.7)]
    tr, te = df[df.date <= cut], df[df.date > cut]
    if tr["label"].nunique() < 2 or len(te) < 50:
        d.close()
        return status()
    from sklearn.ensemble import GradientBoostingClassifier
    X = tr[_FEATS].to_numpy(); y = tr["label"].to_numpy()
    clf = GradientBoostingClassifier(max_depth=3, n_estimators=150, subsample=0.8)
    clf.fit(X, y)
    # recall@K on the held-out days: does the top-K by score capture the goods?
    te = te.copy(); te["score"] = clf.predict_proba(te[_FEATS].to_numpy())[:, 1]
    recalls = []
    for dt, g in te.groupby("date"):
        goods = g[g.label == 1]
        if not len(goods):
            continue
        k = min(SHORTLIST_K, len(g))
        top = g.nlargest(k, "score")
        recalls.append(len(top[top.label == 1]) / len(goods))
    recall = float(np.mean(recalls)) if recalls else 0.0
    import pickle
    with open(os.path.join(HERE, "universe_selector.pkl"), "wb") as f:
        pickle.dump({"clf": clf, "feats": _FEATS}, f)
    imp = dict(sorted(zip(_FEATS, clf.feature_importances_.round(3).tolist()),
                      key=lambda x: -x[1]))
    _set(d, "recall_at_k", recall)
    _set(d, "importances", imp)
    _set(d, "trained_at", time.time())
    phase = _get(d, "phase", "warmup")
    days = _get(d, "days_logged", 0)
    if days >= WARMUP_DAYS and phase == "warmup":
        _set(d, "phase", "shadow")
    if phase in ("shadow", "active"):
        _set(d, "phase", "active" if recall >= GATE_RECALL else "shadow")
    d.commit(); d.close()
    return status()


def shortlist(features: list[dict]) -> tuple[list[str], str]:
    """Which instruments to actually scan today. In 'active' phase: model top-K +
    exploration quota. Otherwise: everything (warmup/shadow scan the full set)."""
    d = _db(); phase = _get(d, "phase", "warmup"); d.close()
    allsyms = [f["asset"] for f in features]
    if phase != "active":
        return allsyms, phase
    try:
        import pickle
        with open(os.path.join(HERE, "universe_selector.pkl"), "rb") as f:
            m = pickle.load(f)
        X = np.array([[f.get(k, 0.0) for k in m["feats"]] for f in features])
        scores = m["clf"].predict_proba(X)[:, 1]
        order = np.argsort(scores)[::-1]
        top = [allsyms[i] for i in order[:SHORTLIST_K]]
        rest = [allsyms[i] for i in order[SHORTLIST_K:]]
        n_ex = int(len(allsyms) * EXPLORE_FRAC)
        explore = list(np.random.default_rng().choice(rest, min(n_ex, len(rest)),
                                                      replace=False)) if rest else []
        return top + explore, phase
    except Exception:
        return allsyms, "shadow"          # any failure → don't narrow


def last_features(symbols) -> dict:
    """Most recent logged feature row per symbol (for active-phase narrowing
    without a fresh fetch)."""
    d = _db()
    want = set(symbols)
    out: dict[str, dict] = {}
    for date, asset, feats in d.execute(
            "SELECT date,asset,feats FROM feat ORDER BY date ASC"):
        if asset in want:
            out[asset] = json.loads(feats)
    d.close()
    return out


def status() -> dict:
    d = _db()
    days = _get(d, "days_logged", 0)
    labelled = d.execute("SELECT COUNT(*) FROM feat WHERE label IS NOT NULL").fetchone()[0]
    total = d.execute("SELECT COUNT(*) FROM feat").fetchone()[0]
    out = {"phase": _get(d, "phase", "warmup"), "days_logged": days,
           "warmup_days": WARMUP_DAYS, "rows_total": total, "rows_labelled": labelled,
           "recall_at_k": _get(d, "recall_at_k"), "gate_recall": GATE_RECALL,
           "shortlist_k": SHORTLIST_K, "importances": _get(d, "importances"),
           "trained_at": _get(d, "trained_at")}
    d.close()
    return out
