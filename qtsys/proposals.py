"""proposals.py — the agent → action inbox.

Agents PROPOSE (the blueprint's bright line); only the risk-checked gateway
DISPOSES. Until now proposals lived only in the scrolling log and evaporated.
This is a durable, de-duplicated queue of actionable proposals the operator
can review and promote into a (still gateway-checked, still confirmed) order.

A proposal is (agent, kind, symbol, side, summary, payload). It is
de-duplicated on a caller-supplied `dedup` key within a TTL window, so an
agent firing every cycle doesn't flood the inbox — it refreshes the existing
open proposal instead. Status flows open -> promoted | dismissed | expired.

SQLite (proposals.db, gitignored). Promotion itself is a UI action: the
terminal opens a prefilled, confirm-gated order ticket — nothing here sends
orders.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time

DB_PATH = os.path.join(os.path.dirname(__file__), "proposals.db")
DEFAULT_TTL = 12 * 3600


class ProposalStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.execute("""CREATE TABLE IF NOT EXISTS proposal(
            id INTEGER PRIMARY KEY, ts REAL, agent TEXT, kind TEXT,
            symbol TEXT, side TEXT, qty REAL, summary TEXT, payload TEXT,
            dedup TEXT, status TEXT, ttl REAL)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS ix_prop_status "
                        "ON proposal(status, ts)")
        self.db.commit()

    def propose(self, agent: str, kind: str, summary: str, symbol: str = "",
                side: str = "", qty: float = 0.0, payload: dict | None = None,
                dedup: str | None = None, ttl: float = DEFAULT_TTL) -> int:
        """Add or refresh an open proposal. If an OPEN proposal with the same
        dedup key exists, its timestamp/summary are refreshed rather than a
        duplicate created. Returns the proposal id."""
        now = time.time()
        dedup = dedup or f"{agent}:{kind}:{symbol}:{side}"
        row = self.db.execute(
            "SELECT id FROM proposal WHERE dedup=? AND status='open' "
            "AND ts+ttl>?", (dedup, now)).fetchone()
        pj = json.dumps(payload or {})
        if row:
            self.db.execute(
                "UPDATE proposal SET ts=?, summary=?, qty=?, payload=?, ttl=? "
                "WHERE id=?", (now, summary, qty, pj, ttl, row[0]))
            self.db.commit()
            return row[0]
        cur = self.db.execute(
            "INSERT INTO proposal(ts,agent,kind,symbol,side,qty,summary,"
            "payload,dedup,status,ttl) VALUES (?,?,?,?,?,?,?,?,?, 'open', ?)",
            (now, agent, kind, symbol, side, qty, summary, pj, dedup, ttl))
        self.db.commit()
        return cur.lastrowid

    def open(self, limit: int = 50) -> list[dict]:
        now = time.time()
        self.db.execute("UPDATE proposal SET status='expired' WHERE "
                        "status='open' AND ts+ttl<?", (now,))
        self.db.commit()
        rows = self.db.execute(
            "SELECT id,ts,agent,kind,symbol,side,qty,summary,payload FROM "
            "proposal WHERE status='open' ORDER BY ts DESC LIMIT ?",
            (limit,)).fetchall()
        return [{"id": r[0], "ts": r[1], "agent": r[2], "kind": r[3],
                 "symbol": r[4], "side": r[5], "qty": r[6], "summary": r[7],
                 "payload": json.loads(r[8] or "{}")} for r in rows]

    def set_status(self, pid: int, status: str) -> bool:
        cur = self.db.execute("UPDATE proposal SET status=? WHERE id=?",
                              (status, pid))
        self.db.commit()
        return cur.rowcount > 0

    def count_open(self) -> int:
        return self.db.execute(
            "SELECT COUNT(*) FROM proposal WHERE status='open' AND ts+ttl>?",
            (time.time(),)).fetchone()[0]


def _selftest():
    import tempfile
    p = tempfile.mktemp(suffix=".db")
    st = ProposalStore(p)
    a = st.propose("Arb Strategist", "triangular", "BTC-ETH nets +40bps",
                   symbol="BTC/USD", side="buy", qty=0.01, dedup="tri:BTC-ETH")
    b = st.propose("Arb Strategist", "triangular", "BTC-ETH nets +55bps",
                   symbol="BTC/USD", side="buy", qty=0.01, dedup="tri:BTC-ETH")
    assert a == b, "dedup refreshes, not duplicates"
    assert st.count_open() == 1
    assert st.open()[0]["summary"].endswith("+55bps"), "refreshed to latest"
    c = st.propose("Fundamental Analyst", "pick", "META cheap growth",
                   symbol="META", side="buy")
    assert st.count_open() == 2
    assert st.set_status(a, "dismissed")
    assert st.count_open() == 1
    # ttl expiry
    st.propose("X", "k", "old", dedup="old", ttl=-1)
    assert st.count_open() == 1, "expired proposals drop out of open()"
    os.remove(p)
    print("proposals self-test ✓  dedup-refresh, dismiss, ttl-expire")


if __name__ == "__main__":
    _selftest()
