"""pit.py — point-in-time fundamentals store (design + implementation).

THE PROBLEM: qtsys.intel serves LIVE fundamentals (today's P/E, today's
margins). Feeding those into a backtest is look-ahead poison — you'd be
trading 2024 prices on 2026 knowledge, and restated financials make it worse
(vendors silently revise history; what you'd have SEEN then differs from
what databases say now).

THE DESIGN (implemented here):
  - Free sources offer NO honest backfill, so the store accumulates its own
    vintages: one snapshot per (symbol, day) of the normalised metrics dict,
    written by the Fundamental Analyst's daily cycle from TODAY FORWARD.
  - A vintage is immutable: what we knew on day D stays exactly as recorded
    on day D — no revisions, ever (INSERT OR IGNORE).
  - The ONLY sanctioned backtest read is `asof(symbol, date)`: the latest
    vintage with snapshot-date <= date. If none exists, the answer is None —
    a backtest before the store's first day has NO fundamentals, honestly.
  - After ~2 quarters of accumulation the store supports real fundamental
    factor research (e.g. did cheap-quality outperform, measured with the
    numbers as they were actually known).

SQLite (pit_fundamentals.db, gitignored like every runtime .db).
Run `python -m qtsys.pit` for self-tests.
"""
from __future__ import annotations

import datetime
import json
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "pit_fundamentals.db")


class PITStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.execute("""CREATE TABLE IF NOT EXISTS vintage(
            asof TEXT, symbol TEXT, metrics TEXT,
            PRIMARY KEY(asof, symbol))""")
        self.db.commit()

    # ---------------------------------------------------------------- write
    def snapshot(self, symbols, fundamentals_fn, asof: str | None = None) -> int:
        """Record today's vintage for each symbol. Idempotent per (day,
        symbol); existing vintages are NEVER overwritten. Returns # written."""
        asof = asof or str(datetime.date.today())
        n = 0
        for s in symbols:
            try:
                m = (fundamentals_fn(s) or {}).get("metrics") or {}
            except Exception:
                continue
            m = {k: v for k, v in m.items()
                 if isinstance(v, (int, float, str)) and v is not None}
            if not m:
                continue
            cur = self.db.execute(
                "INSERT OR IGNORE INTO vintage VALUES (?,?,?)",
                (asof, s, json.dumps(m)))
            n += cur.rowcount
        self.db.commit()
        return n

    # ----------------------------------------------------------------- read
    def asof(self, symbol: str, date: str) -> dict | None:
        """Point-in-time read: the latest vintage recorded ON OR BEFORE
        `date`. None if the store hadn't started — no silent backfill."""
        row = self.db.execute(
            "SELECT asof, metrics FROM vintage WHERE symbol=? AND asof<=? "
            "ORDER BY asof DESC LIMIT 1", (symbol, date)).fetchone()
        if not row:
            return None
        return {"asof": row[0], "metrics": json.loads(row[1])}

    def history(self, symbol: str, key: str) -> list[tuple[str, float]]:
        """Vintage series of one metric — for factor research."""
        out = []
        for asof, m in self.db.execute(
                "SELECT asof, metrics FROM vintage WHERE symbol=? ORDER BY asof",
                (symbol,)):
            v = json.loads(m).get(key)
            if isinstance(v, (int, float)):
                out.append((asof, float(v)))
        return out

    def coverage(self) -> dict:
        row = self.db.execute("SELECT COUNT(DISTINCT asof), "
                              "COUNT(DISTINCT symbol), MIN(asof), MAX(asof), "
                              "COUNT(*) FROM vintage").fetchone()
        return {"days": row[0], "symbols": row[1], "first": row[2],
                "last": row[3], "vintages": row[4]}


# ------------------------------------------------------------------ self-test
def _selftest():
    import tempfile
    p = tempfile.mktemp(suffix=".db")
    st = PITStore(p)
    fn1 = lambda s: {"metrics": {"pe": 20.0, "margin": 10.0}}
    fn2 = lambda s: {"metrics": {"pe": 25.0, "margin": 11.0}}
    assert st.snapshot(["AAA"], fn1, asof="2026-01-10") == 1
    assert st.snapshot(["AAA"], fn2, asof="2026-02-10") == 1
    # immutability: same-day re-snapshot with different numbers is IGNORED
    assert st.snapshot(["AAA"], fn2, asof="2026-01-10") == 0
    assert st.asof("AAA", "2026-01-10")["metrics"]["pe"] == 20.0
    # PIT correctness: mid-period read sees the OLD vintage, not the new one
    assert st.asof("AAA", "2026-02-09")["metrics"]["pe"] == 20.0
    assert st.asof("AAA", "2026-03-01")["metrics"]["pe"] == 25.0
    # no-lookahead: before the store existed -> None, never a backfill
    assert st.asof("AAA", "2026-01-09") is None
    assert st.history("AAA", "pe") == [("2026-01-10", 20.0), ("2026-02-10", 25.0)]
    cov = st.coverage()
    assert cov["days"] == 2 and cov["symbols"] == 1 and cov["vintages"] == 2
    os.remove(p)
    print("pit self-test ✓  vintages immutable, as-of reads point-in-time, "
          "pre-store reads honestly None")


if __name__ == "__main__":
    _selftest()
