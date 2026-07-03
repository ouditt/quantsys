"""High-frequency stack (card 13) — REAL DATA ONLY, by recording it yourself.

Honesty first: true HFT (microseconds, colocation, rebate tiers) is not
winnable from a laptop, and this module does not pretend otherwise. What IS
realistically attemptable at retail: 100ms–1s reaction trading on crypto,
where venues hand out full order-book and trade feeds for free. So the
pipeline is:

  1) RECORD real microstructure locally:   python -m qtsys.hft record binance BTC/USDT 600
  2) BACKTEST on the recording:            python -m qtsys.hft backtest recordings/binance_BTCUSDT.csv mm
  3) PAPER trade the same logic via the ExecutionGateway; live only after that.

Nothing synthetic anywhere: the backtester replays your recorded books and
trades. Fees and latency are COST MODELS (deterministic arithmetic), not data.
The unit fixtures at the bottom verify order-matching arithmetic only and are
never a source of performance claims.

Two starter strategies:
  mm        passive spread-capture with inventory bands — quotes only when the
            spread pays > 2× maker fees + buffer; skews quotes against inventory
  imbalance top-of-book depth imbalance momentum (taker) with tick TP/SL

Cross-venue spread scan (card 15) is `scan_arb` — run it live locally; it
reports NET spreads after both venues' taker fees, which is why most "free
money" prints vanish on contact.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass, field

MAKER_BPS, TAKER_BPS = 1.0, 5.0          # typical crypto tier-0; override per venue
LATENCY_MS = 250                         # honest home-connection assumption


# ------------------------------------------------------------------ recording
def record(exchange_id: str, symbol: str, seconds: int = 600, depth: int = 5,
           out_dir: str = "recordings") -> str:
    """Poll a real venue's order book + trades ~1/s and append to CSV.
    Requires network + ccxt: run on YOUR machine, not the sandbox."""
    import ccxt                                             # local install
    ex = getattr(ccxt, exchange_id)()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{exchange_id}_{symbol.replace('/', '')}.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            hdr = ["ts"] + [f"{s}{i}{k}" for s in ("bid", "ask")
                            for i in range(depth) for k in ("px", "sz")] + ["last_trade_px", "last_trade_side"]
            w.writerow(hdr)
        t_end, last_id = time.time() + seconds, None
        while time.time() < t_end:
            ob = ex.fetch_order_book(symbol, depth)
            row = [time.time()]
            for side in ("bids", "asks"):
                lv = ob[side][:depth] + [[None, None]] * (depth - len(ob[side]))
                for px, sz in lv:
                    row += [px, sz]
            tr = ex.fetch_trades(symbol, limit=1)
            row += [tr[-1]["price"], tr[-1]["side"]] if tr else [None, None]
            w.writerow(row); f.flush()
            time.sleep(1.0)
    return path


def load_recording(path: str, depth: int = 5) -> list[dict]:
    books = []
    with open(path) as f:
        for r in csv.DictReader(f):
            b = {"ts": float(r["ts"]),
                 "bids": [(float(r[f"bid{i}px"]), float(r[f"bid{i}sz"]))
                          for i in range(depth) if r.get(f"bid{i}px")],
                 "asks": [(float(r[f"ask{i}px"]), float(r[f"ask{i}sz"]))
                          for i in range(depth) if r.get(f"ask{i}px")],
                 "trade_px": float(r["last_trade_px"]) if r.get("last_trade_px") else None,
                 "trade_side": r.get("last_trade_side") or None}
            if b["bids"] and b["asks"]:
                books.append(b)
    return books


# ------------------------------------------------------------------ backtester
@dataclass
class HOrder:
    side: str; px: float; qty: float; maker: bool; ts: float; oid: int


@dataclass
class HState:
    cash: float = 0.0
    inv: float = 0.0
    fees: float = 0.0
    fills: int = 0
    orders: list = field(default_factory=list)
    _oid: int = 0


class TickBacktester:
    """Event-driven replay of REAL recorded books.

    Conservative fill model (biases results DOWN, never up):
      taker: fills at the touch of the NEXT snapshot (latency), size-capped by
             displayed depth, plus taker fee
      maker: rests; fills only if a later snapshot's opposite touch STRICTLY
             crosses the quote (price trades through it) — i.e. assume the
             whole visible queue was ahead of you
    """

    def __init__(self, books, maker_bps=MAKER_BPS, taker_bps=TAKER_BPS,
                 latency_ms=LATENCY_MS, max_inv=1.0, kill_dd=0.02):
        self.books, self.mbps, self.tbps = books, maker_bps / 1e4, taker_bps / 1e4
        self.lat = latency_ms / 1000.0
        self.max_inv, self.kill_dd = max_inv, kill_dd

    def run(self, strategy) -> dict:
        st, eq_peak, killed = HState(), 0.0, False
        pending: list[tuple[float, str, float]] = []          # (exec_ts, side, qty)
        mid0 = (self.books[0]["bids"][0][0] + self.books[0]["asks"][0][0]) / 2
        for k, b in enumerate(self.books):
            bid, ask = b["bids"][0], b["asks"][0]
            # 1) resting maker orders: strict trade-through
            for o in list(st.orders):
                if o.maker and ((o.side == "buy" and ask[0] < o.px) or
                                (o.side == "sell" and bid[0] > o.px)):
                    self._fill(st, o.side, o.px, o.qty, maker=True)
                    st.orders.remove(o)
            # 2) latency-delayed taker executions at current touch
            for ts, side, qty in list(pending):
                if b["ts"] >= ts:
                    px, cap = (ask if side == "buy" else bid)
                    self._fill(st, side, px, min(qty, cap), maker=False)
                    pending.remove((ts, side, qty))
            # 3) kill switch on inventory-marked drawdown
            eq = st.cash + st.inv * (bid[0] + ask[0]) / 2 - st.fees
            eq_peak = max(eq_peak, eq)
            if eq_peak - eq > self.kill_dd * mid0 * self.max_inv:
                for o in list(st.orders):
                    st.orders.remove(o)
                if abs(st.inv) > 1e-12:                       # flatten at touch
                    side = "sell" if st.inv > 0 else "buy"
                    px = bid[0] if side == "sell" else ask[0]
                    self._fill(st, side, px, abs(st.inv), maker=False)
                killed = True
                break
            # 4) let the strategy act on the REAL book
            for act in strategy.on_book(b, st):
                if act["type"] == "cancel_all":
                    st.orders.clear()
                elif act["type"] == "taker":
                    if abs(st.inv + (act["qty"] if act["side"] == "buy" else -act["qty"])) <= self.max_inv:
                        pending.append((b["ts"] + self.lat, act["side"], act["qty"]))
                elif act["type"] == "maker":
                    st._oid += 1
                    st.orders.append(HOrder(act["side"], act["px"], act["qty"],
                                            True, b["ts"], st._oid))
        last = self.books[min(k, len(self.books) - 1)]
        mid = (last["bids"][0][0] + last["asks"][0][0]) / 2
        pnl = st.cash + st.inv * mid - st.fees
        return {"pnl": pnl, "fills": st.fills, "fees": st.fees,
                "end_inventory": st.inv, "killed": killed,
                "pnl_bps_of_mid": 1e4 * pnl / mid if mid else 0.0}

    def _fill(self, st: HState, side: str, px: float, qty: float, maker: bool):
        if qty <= 0:
            return
        st.cash += -px * qty if side == "buy" else px * qty
        st.inv += qty if side == "buy" else -qty
        st.fees += px * qty * (self.mbps if maker else self.tbps)
        st.fills += 1


# ------------------------------------------------------------------ strategies
class MMSpread:
    """Quote both sides around mid only when the spread pays. Inventory skew
    pushes quotes to unload risk; stops quoting the loaded side at the band."""

    def __init__(self, min_edge_bps=None, qty=0.001, band=0.005):
        self.edge = (min_edge_bps or (2 * MAKER_BPS + 2)) / 1e4
        self.qty, self.band = qty, band

    def on_book(self, b, st):
        bid, ask = b["bids"][0][0], b["asks"][0][0]
        mid, spread = (bid + ask) / 2, (ask - bid)
        if spread / mid < self.edge:
            return [{"type": "cancel_all"}]
        skew = -st.inv / max(self.band, 1e-12) * spread / 2
        acts = [{"type": "cancel_all"}]
        if st.inv < self.band:
            acts.append({"type": "maker", "side": "buy",
                         "px": round(mid - spread / 2 + skew, 8), "qty": self.qty})
        if st.inv > -self.band:
            acts.append({"type": "maker", "side": "sell",
                         "px": round(mid + spread / 2 + skew, 8), "qty": self.qty})
        return acts


class ImbalanceTaker:
    """Depth imbalance momentum: when displayed size is lopsided beyond theta,
    take in that direction; flat otherwise (time-based exit via re-imbalance)."""

    def __init__(self, theta=0.75, qty=0.001, levels=3):
        self.th, self.qty, self.lv = theta, qty, levels

    def on_book(self, b, st):
        bsz = sum(s for _, s in b["bids"][:self.lv])
        asz = sum(s for _, s in b["asks"][:self.lv])
        tot = bsz + asz
        if tot <= 0:
            return []
        imb = bsz / tot
        if imb > self.th and st.inv <= 0:
            return [{"type": "taker", "side": "buy", "qty": self.qty}]
        if imb < 1 - self.th and st.inv >= 0:
            return [{"type": "taker", "side": "sell", "qty": self.qty}]
        return []


# ------------------------------------------------------------------ arb scan
def scan_arb(symbol="BTC/USDT", venues=("binance", "kraken", "coinbase"),
             taker_bps=(10, 16, 25)) -> list[dict]:
    """Card 15 — cross-venue NET spread after both taker fees. Run locally."""
    import ccxt
    quotes = {}
    for v in venues:
        try:
            ob = getattr(ccxt, v)().fetch_order_book(symbol, 5)
            quotes[v] = (ob["bids"][0][0], ob["asks"][0][0])
        except Exception as e:                                # venue down/geo-blocked
            quotes[v] = None
    fees = dict(zip(venues, taker_bps))
    out = []
    for a in venues:
        for b in venues:
            if a == b or not quotes.get(a) or not quotes.get(b):
                continue
            buy_px, sell_px = quotes[a][1], quotes[b][0]
            gross = (sell_px - buy_px) / buy_px * 1e4
            net = gross - fees[a] - fees[b]
            out.append({"buy_on": a, "sell_on": b, "gross_bps": round(gross, 2),
                        "net_bps_after_fees": round(net, 2),
                        "actionable": net > 5})
    return sorted(out, key=lambda x: -x["net_bps_after_fees"])


# ------------------------------------------------------------------ fixtures
def _selftest():
    """Order-matching ARITHMETIC checks on a hand-written 6-snapshot fixture.
    Labeled test-only: never a source of performance numbers."""
    mk = lambda ts, b, a: {"ts": ts, "bids": [(b, 1.0)], "asks": [(a, 1.0)],
                           "trade_px": None, "trade_side": None}
    books = [mk(0, 100.0, 100.2), mk(1, 100.0, 100.2), mk(2, 100.3, 100.5),
             mk(3, 100.3, 100.5), mk(4, 99.8, 100.0), mk(5, 99.8, 100.0)]

    class OneQuote:                                            # posts once at t=0
        def __init__(self):
            self.done = False
        def on_book(self, b, st):
            if not self.done:
                self.done = True
                return [{"type": "maker", "side": "sell", "px": 100.25, "qty": 1.0}]
            return []

    bt = TickBacktester(books, maker_bps=1.0, taker_bps=5.0, latency_ms=0,
                        max_inv=2.0, kill_dd=1.0)
    r = bt.run(OneQuote())
    # bid at t=2 is 100.3 > 100.25 -> maker sell fills at 100.25; fee 1bp
    assert r["fills"] == 1 and abs(r["end_inventory"] + 1.0) < 1e-9
    exp_fee = 100.25 * 1.0 * 1e-4
    exp_pnl = 100.25 - 99.9 - exp_fee                          # marked at last mid 99.9
    assert abs(r["fees"] - exp_fee) < 1e-9 and abs(r["pnl"] - exp_pnl) < 1e-9

    class TakeOnce:
        def __init__(self):
            self.done = False
        def on_book(self, b, st):
            if not self.done:
                self.done = True
                return [{"type": "taker", "side": "buy", "qty": 0.5}]
            return []

    r2 = TickBacktester(books, taker_bps=5.0, latency_ms=1500, max_inv=2.0,
                        kill_dd=1.0).run(TakeOnce())
    # latency 1.5s from t=0 -> executes on the t=2 snapshot at ask 100.5
    exp_fee2 = 100.5 * 0.5 * 5e-4
    exp_pnl2 = (99.9 - 100.5) * 0.5 - exp_fee2
    assert r2["fills"] == 1 and abs(r2["pnl"] - exp_pnl2) < 1e-9
    print("hft self-test: maker trade-through, taker latency, fees — exact ✓")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "record":
        print(record(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 600))
    elif len(sys.argv) > 1 and sys.argv[1] == "backtest":
        books = load_recording(sys.argv[2])
        strat = MMSpread() if (len(sys.argv) < 4 or sys.argv[3] == "mm") else ImbalanceTaker()
        print(TickBacktester(books).run(strat))
    elif len(sys.argv) > 1 and sys.argv[1] == "arb":
        for row in scan_arb():
            print(row)
    else:
        _selftest()
