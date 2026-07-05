"""orderbook.py — L2 depth-of-book microstructure metrics (pure functions).

Turns a raw crypto order book (from AlpacaBroker.crypto_orderbook — free L2)
into the numbers an execution/signal policy actually reasons over:

  - mid / spread (bps)
  - top-of-book sizes and depth within a bps band of mid
  - order-book imbalance (OBI, -1..+1) and the size-weighted microprice
  - expected slippage in bps to fill a reference notional by WALKING the book

All of this is invisible to a Level-1 (top-of-book only) feed — which is the
whole point of the L2-benefit experiment in l2lab.py. No network here; feed it
a book dict {'bids': [(price,size),...], 'asks': [...]}.
"""
from __future__ import annotations


def _walk(levels, notional: float):
    """VWAP + fill fraction to consume `notional` USD across price levels
    (already ordered best-first). Returns (vwap, filled_notional, exhausted)."""
    spent = qty = 0.0
    for price, size in levels:
        if price <= 0 or size <= 0:
            continue
        lvl_notional = price * size
        take = min(lvl_notional, notional - spent)
        qty += take / price
        spent += take
        if spent >= notional - 1e-9:
            return (spent / qty if qty else None), spent, False
    return (spent / qty if qty else None), spent, True


def metrics(book: dict, ref_notional: float = 5000.0, band_bps: float = 25.0) -> dict:
    """Microstructure metrics for one L2 book. `ref_notional` is the order size
    (USD) used for the slippage estimate; `band_bps` the depth-measurement band."""
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    if not bids or not asks:
        return {}
    bb, bb_sz = bids[0]
    ba, ba_sz = asks[0]
    if bb <= 0 or ba <= 0:
        return {}
    mid = (bb + ba) / 2.0
    spread_bps = (ba - bb) / mid * 1e4
    band = mid * band_bps / 1e4
    depth_bid = sum(sz for p, sz in bids if p >= mid - band)     # base-units
    depth_ask = sum(sz for p, sz in asks if p <= mid + band)
    db_usd, da_usd = depth_bid * mid, depth_ask * mid
    imb = ((db_usd - da_usd) / (db_usd + da_usd)) if (db_usd + da_usd) else 0.0
    micro = (bb * ba_sz + ba * bb_sz) / (bb_sz + ba_sz) if (bb_sz + ba_sz) else mid
    buy_vwap, _, buy_exh = _walk(asks, ref_notional)
    sell_vwap, _, sell_exh = _walk(bids, ref_notional)
    slip_buy = (buy_vwap / mid - 1) * 1e4 if buy_vwap else None
    slip_sell = (1 - sell_vwap / mid) * 1e4 if sell_vwap else None
    return {
        "mid": mid, "spread_bps": round(spread_bps, 3),
        "bid": bb, "ask": ba, "bid_sz": bb_sz, "ask_sz": ba_sz,
        "depth_bid_usd": round(db_usd, 1), "depth_ask_usd": round(da_usd, 1),
        "imbalance": round(imb, 4), "microprice": micro,
        "micro_tilt_bps": round((micro / mid - 1) * 1e4, 3),
        "slip_buy_bps": round(slip_buy, 3) if slip_buy is not None else None,
        "slip_sell_bps": round(slip_sell, 3) if slip_sell is not None else None,
        "depth_exhausted": bool(buy_exh or sell_exh),
        "ref_notional": ref_notional, "levels_bid": len(bids), "levels_ask": len(asks),
    }
