"""l2lab.py — the crypto-L2 benefit experiment.

Runs a live, honest A/B: every cycle it snapshots the free Alpaca crypto L2
book, records the microstructure metrics an L2-aware policy would use, and on
the NEXT snapshot labels each row with the realised forward return. That lets
us quantify — weekly — what L2 buys you that a Level-1 (top-of-book) feed
simply cannot see:

  1. slippage/ depth risk  — expected cost to fill a reference order by walking
     the book, and how often that order would EXHAUST visible depth (an L1-only
     policy is blind to both and would over-trade into thin books);
  2. order-book-imbalance signal — does the sign of OBI / microprice-tilt
     predict the next interval's move better than a coin flip (which is all L1
     has)?
  3. spread capture — the passive-fill opportunity L1 can't size.

After 180 days it emits a one-shot upgrade-decision brief that recaps the L2/SIP
economics discussed with the operator and asks — armed with the measured
numbers — whether to buy the paid equity SIP / L2 feeds.

Self-contained SQLite (l2lab.db); every call is best-effort.
"""
from __future__ import annotations

import os
import sqlite3
import time

from . import orderbook

CRYPTO = ("BTC/USD", "ETH/USD")
REMIND_AFTER_DAYS = 180


class L2Lab:
    def __init__(self, db_path: str, ob_fn):
        """ob_fn: symbol -> book dict {'bids':[(px,sz)..],'asks':[..]}."""
        self.ob_fn = ob_fn
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("PRAGMA busy_timeout=5000")
        self._init()

    def _init(self):
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS snap(
          id INTEGER PRIMARY KEY, ts REAL, sym TEXT, mid REAL, spread_bps REAL,
          imbalance REAL, micro_tilt_bps REAL, slip_buy_bps REAL, slip_sell_bps REAL,
          depth_exhausted INTEGER, fwd_ret REAL);
        CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v REAL);
        CREATE INDEX IF NOT EXISTS ix_snap_sym_ts ON snap(sym, ts);
        """)
        self.db.commit()

    def _kv(self, k, v=None):
        if v is None:
            r = self.db.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
            return r[0] if r else 0.0
        self.db.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (k, v))
        self.db.commit()

    # -------------------------------------------------------------- snapshot
    def snapshot(self) -> int:
        """Take one L2 snapshot per crypto symbol; label the previous row with the
        realised forward return. Returns the number of symbols captured."""
        now = time.time()
        n = 0
        for sym in CRYPTO:
            try:
                m = orderbook.metrics(self.ob_fn(sym))
            except Exception:
                m = {}
            if not m:
                continue
            # label the last unlabelled row for this symbol
            prev = self.db.execute(
                "SELECT id, mid FROM snap WHERE sym=? AND fwd_ret IS NULL "
                "ORDER BY ts DESC LIMIT 1", (sym,)).fetchone()
            if prev and prev[1]:
                self.db.execute("UPDATE snap SET fwd_ret=? WHERE id=?",
                                (m["mid"] / prev[1] - 1.0, prev[0]))
            self.db.execute(
                "INSERT INTO snap(ts,sym,mid,spread_bps,imbalance,micro_tilt_bps,"
                "slip_buy_bps,slip_sell_bps,depth_exhausted,fwd_ret) "
                "VALUES (?,?,?,?,?,?,?,?,?,NULL)",
                (now, sym, m["mid"], m["spread_bps"], m["imbalance"],
                 m["micro_tilt_bps"], m["slip_buy_bps"], m["slip_sell_bps"],
                 1 if m["depth_exhausted"] else 0))
            n += 1
        if n and not self._kv("first_ts"):
            self._kv("first_ts", now)
        self.db.commit()
        return n

    # ---------------------------------------------------------------- stats
    def days_active(self) -> float:
        f = self._kv("first_ts")
        return (time.time() - f) / 86400.0 if f else 0.0

    def _hit_rate(self, rows, col_idx):
        """Fraction of labelled rows where sign(signal) == sign(fwd_ret)."""
        hit = tot = 0
        for r in rows:
            sig, fwd = r[col_idx], r[-1]
            if fwd is None or sig is None or sig == 0 or fwd == 0:
                continue
            tot += 1
            if (sig > 0) == (fwd > 0):
                hit += 1
        return (hit / tot, tot) if tot else (None, 0)

    def _sym_stats(self, sym, since):
        rows = self.db.execute(
            "SELECT mid,spread_bps,imbalance,micro_tilt_bps,slip_buy_bps,"
            "slip_sell_bps,depth_exhausted,fwd_ret FROM snap "
            "WHERE sym=? AND ts>=? ORDER BY ts", (sym, since)).fetchall()
        if not rows:
            return None
        n = len(rows)
        avg = lambda i: (sum(r[i] for r in rows if r[i] is not None)
                         / max(1, sum(1 for r in rows if r[i] is not None)))
        imb_hit, imb_n = self._hit_rate(rows, 2)      # imbalance vs fwd_ret
        mt_hit, mt_n = self._hit_rate(rows, 3)        # micro_tilt vs fwd_ret
        exh = sum(r[6] for r in rows) / n
        return {"sym": sym, "n": n, "spread_bps": avg(1), "slip_buy": avg(4),
                "slip_sell": avg(5), "imb_hit": imb_hit, "imb_n": imb_n,
                "mt_hit": mt_hit, "mt_n": mt_n, "exhausted_pct": exh * 100}

    # -------------------------------------------------------------- reports
    def weekly_report(self) -> str:
        since = time.time() - 7 * 86400
        L = [f"CRYPTO L2 BENEFIT — trailing 7 days (day {self.days_active():.0f} of the trial)",
             "Measures what the free L2 depth feed sees that a Level-1 (top-of-book) "
             "feed cannot. Baseline = L1-only (no depth, no imbalance).", ""]
        any_sym = False
        best_edge = 0.0
        for sym in CRYPTO:
            s = self._sym_stats(sym, since)
            if not s:
                continue
            any_sym = True
            L.append(f"## {sym}  ({s['n']} snapshots)")
            L.append(f"  avg spread: {s['spread_bps']:.2f} bps  "
                     f"(passive-fill capture an L1 policy can't size)")
            if s["slip_buy"] is not None:
                L.append(f"  expected slippage on a $5k order: "
                         f"buy {s['slip_buy']:.2f} bps / sell {s['slip_sell']:.2f} bps  "
                         f"(L1 is blind to this — assumes a touch fill)")
            L.append(f"  book too thin for $5k: {s['exhausted_pct']:.1f}% of the time  "
                     f"(L1 would over-trade into these; L2 declines/splits)")
            if s["imb_hit"] is not None:
                edge = (s["imb_hit"] - 0.5) * 100
                best_edge = max(best_edge, edge)
                L.append(f"  order-book imbalance predicted next-interval direction "
                         f"{s['imb_hit']*100:.1f}% (n={s['imb_n']}, vs 50% for L1) "
                         f"→ {edge:+.1f} pts of directional edge")
            if s["mt_hit"] is not None:
                L.append(f"  microprice tilt predicted direction {s['mt_hit']*100:.1f}% "
                         f"(n={s['mt_n']})")
            L.append("")
        if not any_sym:
            return ("CRYPTO L2 BENEFIT — no snapshots in the last 7 days yet "
                    "(the experiment records on each agent cycle once L2 is live).")
        verdict = ("edge visible — L2 is adding signal + execution safety the L1 feed can't"
                   if best_edge >= 2.0 else
                   "marginal so far — mostly execution-safety value; keep collecting")
        L.append(f"VERDICT: {verdict}. All of the above is $0 (crypto L2 is free); "
                 "it also directly sharpens the triangular-arb skill (depth-validated legs).")
        return "\n".join(L)

    def upgrade_due(self) -> bool:
        return (self.days_active() >= REMIND_AFTER_DAYS
                and not self._kv("upgrade_reminded"))

    def upgrade_reminder(self) -> str:
        """One-shot 180-day brief: recap the L2/SIP economics we discussed, attach
        the measured crypto-L2 benefit, and ask whether to buy the paid feeds."""
        self._kv("upgrade_reminded", 1.0)
        wk = self.weekly_report()
        return "\n".join([
            "═══ L2 / SIP UPGRADE DECISION — 180-DAY CHECK-IN ═══",
            "",
            f"You've now run the FREE crypto L2 depth feed for "
            f"{self.days_active():.0f} days. As agreed, here's the recap of what "
            "we discussed, plus the measured results, so you can decide on the PAID "
            "equity feeds.",
            "",
            "WHAT THE PAID FEEDS ADD (recap):",
            "  • SIP (consolidated tape / NBBO): the official all-exchange best "
            "bid/offer + every trade. Fixes today's IEX-only equity gap; it's the "
            "reference price fills are measured against.",
            "  • L2 depth-of-book (equities): the full ladder of resting size at "
            "each price — liquidity-aware sizing, slippage estimation, order-flow "
            "imbalance signals. (You've been getting exactly this for CRYPTO, free.)",
            "",
            "PRICING (approx., verify current):",
            "  • Crypto L2 — FREE (what you're running now).",
            "  • Alpaca 'Algo Trader Plus' — ~$99/mo = full equity SIP real-time "
            "(+OPRA options). Cheapest way to close the consolidated-tape gap.",
            "  • Equity L2 depth — ~$199/mo floor (Polygon 'Advanced' or Databento). "
            "Professional/redistribution status adds exchange license fees.",
            "",
            "ARB FEASIBILITY (recap): L2 doesn't create arbitrage on one retail "
            "broker. It sharpens crypto TRIANGULAR arb (depth-validated legs) and "
            "feeds order-book-imbalance signals to the stat-arb agent. Latency and "
            "spatial arb remain infeasible without co-location / multiple venues.",
            "",
            "── MEASURED CRYPTO-L2 BENEFIT OVER THE TRIAL ──",
            wk,
            "",
            "DECISION REQUESTED: based on the crypto-L2 performance above, do you "
            "want to implement the paid equity data + SIP?",
            "  [A] Alpaca SIP only (~$99/mo)   [B] add equity L2 depth (~$199/mo)   "
            "[C] stay crypto-only (free)",
            "Reply with your choice and I'll wire the adapter (already scaffoldable "
            "behind a feature flag).",
        ])
