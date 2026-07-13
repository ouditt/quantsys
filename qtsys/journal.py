"""Trading journal (card 9) — every trade recorded before, during, and after;
weekly analysis that surfaces patterns in wins AND losses; and the one
question that exposes the real reason behind a losing trade better than any
indicator: "WAS THIS LOSS INSIDE THE PLAN?" A loss inside the plan is the
paid-for cost of the edge (variance). A loss outside the plan — wrong size,
late entry, skipped stop, setup not on the scan — is the only kind that
predicts account failure, so breach-rate is the journal's headline metric.
For an agentic book "psychology" = rule adherence, and it is measurable.

Run:  python -m qtsys.journal          (seeds from the real flagship trades)
"""
from __future__ import annotations

import json
import os
import sqlite3
import time

import pandas as pd

HERE = os.path.dirname(__file__)
FIELDS_BEFORE = ["setup_id", "asset", "side", "signal_date", "planned_risk_pct",
                 "planned_stop_frac", "regime_trend", "regime_vol", "scan_rank", "tier"]
FIELDS_DURING = ["entry_px", "size_units", "slippage_bps_vs_plan", "mfe", "mae"]
FIELDS_AFTER = ["exit_px", "net_ret", "hold_bars", "exit_reason",
                "loss_within_plan", "breach_code", "note"]
BREACH_CODES = {"NONE": "no breach", "SIZE": "size differed from sizing.py output",
                "ENTRY": "entered without a fresh scan signal",
                "STOP": "stop moved/skipped", "LIMITS": "traded through a risk limit",
                "DATA": "stale or failed data at decision time"}


class Journal:
    def __init__(self, db_path: str | None = None):
        self.db = sqlite3.connect(db_path or os.path.join(HERE, "journal.db"))
        self.db.execute("PRAGMA busy_timeout=5000")   # wait out brief contention
        self.db.execute("PRAGMA journal_mode=WAL")     # concurrent read/write
        cols = ", ".join(f"{c} TEXT" for c in FIELDS_BEFORE + FIELDS_DURING + FIELDS_AFTER)
        self.db.execute(f"CREATE TABLE IF NOT EXISTS trades (ts REAL, {cols})")
        for col in ("mfe", "mae"):                 # migrate older journals
            try:
                self.db.execute(f"ALTER TABLE trades ADD COLUMN {col} TEXT")
            except Exception:
                pass
        self.db.commit()

    def log(self, **kw):
        row = {c: kw.get(c) for c in FIELDS_BEFORE + FIELDS_DURING + FIELDS_AFTER}
        self.db.execute(
            f"INSERT INTO trades VALUES ({','.join('?' * (len(row) + 1))})",
            [time.time()] + [json.dumps(v) if isinstance(v, (dict, list)) else v
                             for v in row.values()])
        self.db.commit()

    def frame(self) -> pd.DataFrame:
        df = pd.read_sql("SELECT * FROM trades", self.db)
        for c in ("net_ret", "planned_risk_pct", "hold_bars", "slippage_bps_vs_plan",
                  "mfe", "mae"):
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def weekly_review(self) -> str:
        df = self.frame()
        if df.empty:
            return "journal empty — nothing traded, nothing learned"
        g = (df.groupby(["setup_id", "regime_trend"])["net_ret"]
               .agg(n="count", wr=lambda x: (x > 0).mean(), exp="mean"))
        g = g[g["n"] >= 5].sort_values("exp", ascending=False)
        breaches = df[df["breach_code"].fillna("NONE") != "NONE"]
        br = len(breaches) / len(df)
        worst = (breaches.groupby("breach_code")["net_ret"].agg(["count", "mean"])
                 if len(breaches) else None)
        lines = ["WEEKLY REVIEW — patterns in real logged trades",
                 f"trades {len(df)} | breach rate {br:.1%} "
                 f"({'ACCEPTABLE' if br < 0.05 else 'FIX THE PROCESS BEFORE THE STRATEGY'})",
                 "", "expectancy by (setup, regime) — feed the best back into scan ranking:"]
        for (sid, reg), r in g.iterrows():
            lines.append(f"  {sid:16s} {reg:4s}  n={int(r['n']):4d}  wr {r['wr']:.0%}  "
                         f"exp {r['exp']:+.2%}")
        if worst is not None:
            lines.append("\nbreaches by code (count, avg P&L):")
            for code, r in worst.iterrows():
                lines.append(f"  {code}: {int(r['count'])}  {r['mean']:+.2%}  — {BREACH_CODES.get(code, '')}")
        lines.append("\nTHE question, answered per losing trade: inside the plan? "
                     f"{(df.loc[df['net_ret'] < 0, 'loss_within_plan'] == 'True').mean():.0%} yes "
                     "— those are tuition; the rest are leaks.")
        return "\n".join(lines)


def _seed_demo():
    import numpy as np
    path = os.path.join(HERE, "journal_demo.db")
    if os.path.exists(path):
        os.remove(path)
    j = Journal(path)
    r = pd.read_csv(os.path.join(HERE, "flagship_returns.csv"))["net_ret"]
    rng = np.random.default_rng(3)                      # assigns demo regime labels only
    for i, x in enumerate(r):
        j.log(setup_id="meanrev_rsi2+sel", asset="mixed", side="LONG",
              regime_trend=("UP" if rng.random() < 0.7 else "DOWN"),
              regime_vol="NORMAL", tier="FLAGSHIP",
              planned_risk_pct=0.015, net_ret=float(x),
              hold_bars=int(rng.integers(2, 8)),
              exit_reason="barrier", loss_within_plan=str(x > -0.12),
              breach_code="NONE" if rng.random() > 0.03 else "SIZE")
    print(j.weekly_review())
    os.remove(path)


if __name__ == "__main__":
    _seed_demo()
