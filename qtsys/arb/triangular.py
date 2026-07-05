"""triangular.py — crypto triangular-arbitrage monitor (execution-honest).

Computes the round-trip edge of a currency loop (e.g. USD→BTC→ETH→USD) by
WALKING the real L2 depth on every leg — not touch prices — and netting the
per-leg taker fee. Both loop directions are evaluated; a signal exists only
when the worse of depth-exhaustion/fees still leaves positive edge.

This is the honest version of the classic "EUR/USD × USD/JPY ≠ EUR/JPY"
textbook trade: on one retail venue the edge is almost always inside
fees+depth, and the monitor proves (and logs) exactly when it isn't.
Agents PROPOSE on positive net edge; nothing here auto-executes.

Fee default: Alpaca crypto taker 25 bps/leg (QTSYS_CRYPTO_FEE_BPS overrides).
"""
from __future__ import annotations

import os

TRIANGLES = {                      # loop name -> (leg pairs in USD-out order)
    "BTC-ETH": ("BTC/USD", "ETH/BTC", "ETH/USD"),
    "BTC-LTC": ("BTC/USD", "LTC/BTC", "LTC/USD"),
}


def _fee() -> float:
    return float(os.environ.get("QTSYS_CRYPTO_FEE_BPS", "25")) / 1e4


def _buy(levels, spend):
    """Spend `spend` units of QUOTE ccy walking asks -> (base received, ok)."""
    got = 0.0
    left = spend
    for px, sz in levels:
        if px <= 0 or sz <= 0:
            continue
        cost = px * sz
        take = min(cost, left)
        got += take / px
        left -= take
        if left <= 1e-12:
            return got, True
    return got, False


def _sell(levels, qty):
    """Sell `qty` units of BASE ccy walking bids -> (quote received, ok)."""
    out = 0.0
    left = qty
    for px, sz in levels:
        if px <= 0 or sz <= 0:
            continue
        take = min(sz, left)
        out += take * px
        left -= take
        if left <= 1e-12:
            return out, True
    return out, False


def loop_edge(get_book, triangle: str = "BTC-ETH",
              notional: float = 1000.0, fee_bps: float | None = None) -> dict:
    """Net round-trip edge for both directions of a triangle.

    get_book: pair -> {'bids': [(px,sz)...], 'asks': [...]} (L2, best first).
    Returns {triangle, notional, fee_bps, fwd: {...}, rev: {...}, best}. Each
    direction: edge_bps (net of fees, depth-walked), ok (depth sufficient).
    """
    fee = (_fee() if fee_bps is None else fee_bps / 1e4)
    p_usd_a, p_cross, p_usd_b = TRIANGLES[triangle]
    books = {p: get_book(p) for p in (p_usd_a, p_cross, p_usd_b)}
    if any(not b or not b.get("bids") or not b.get("asks")
           for b in books.values()):
        return {"triangle": triangle, "error": "missing book"}

    # forward: USD -> A (buy p_usd_a) -> B via cross (buy p_cross, quote=A)
    #          -> USD (sell p_usd_b)
    a_qty, ok1 = _buy(books[p_usd_a]["asks"], notional)
    a_qty *= (1 - fee)
    b_qty, ok2 = _buy(books[p_cross]["asks"], a_qty)
    b_qty *= (1 - fee)
    usd_f, ok3 = _sell(books[p_usd_b]["bids"], b_qty)
    usd_f *= (1 - fee)
    fwd = {"path": f"USD→{p_usd_a.split('/')[0]}→{p_usd_b.split('/')[0]}→USD",
           "edge_bps": round((usd_f / notional - 1) * 1e4, 2),
           "ok": bool(ok1 and ok2 and ok3)}

    # reverse: USD -> B (buy p_usd_b) -> A via cross (sell p_cross, base=B)
    #          -> USD (sell p_usd_a)
    b2, rk1 = _buy(books[p_usd_b]["asks"], notional)
    b2 *= (1 - fee)
    a2, rk2 = _sell(books[p_cross]["bids"], b2)
    a2 *= (1 - fee)
    usd_r, rk3 = _sell(books[p_usd_a]["bids"], a2)
    usd_r *= (1 - fee)
    rev = {"path": f"USD→{p_usd_b.split('/')[0]}→{p_usd_a.split('/')[0]}→USD",
           "edge_bps": round((usd_r / notional - 1) * 1e4, 2),
           "ok": bool(rk1 and rk2 and rk3)}

    best = max((d for d in (fwd, rev) if d["ok"]),
               key=lambda d: d["edge_bps"], default=None)
    return {"triangle": triangle, "notional": notional,
            "fee_bps_leg": round((fee) * 1e4, 1), "fwd": fwd, "rev": rev,
            "best": best, "signal": bool(best and best["edge_bps"] > 0)}


# ------------------------------------------------------------------ self-test
def _mkbook(bid, ask, depth=50.0):
    return {"bids": [(bid, depth), (bid * 0.999, depth)],
            "asks": [(ask, depth), (ask * 1.001, depth)]}


def _selftest():
    # consistent books: BTC=50k, ETH=2.5k, cross fair at 0.05 -> no free lunch
    fair = {"BTC/USD": _mkbook(49990, 50010, 10),
            "ETH/BTC": _mkbook(0.04999, 0.05001, 200),
            "ETH/USD": _mkbook(2499.5, 2500.5, 200),
            "LTC/BTC": _mkbook(0.00072, 0.000722, 500),
            "LTC/USD": _mkbook(35.9, 36.1, 500)}
    r = loop_edge(fair.get, "BTC-ETH", 1000, fee_bps=25)
    assert not r["signal"] and r["fwd"]["edge_bps"] < 0 and r["rev"]["edge_bps"] < 0, r
    assert r["fwd"]["ok"] and r["rev"]["ok"]
    # plant a 200bps cross mispricing (ETH cheap in BTC terms): fwd loop wins
    rich = dict(fair)
    rich["ETH/BTC"] = _mkbook(0.04899, 0.04901, 200)
    r2 = loop_edge(rich.get, "BTC-ETH", 1000, fee_bps=25)
    assert r2["signal"] and r2["best"]["edge_bps"] > 50, r2
    assert r2["best"]["path"].startswith("USD→BTC"), r2["best"]
    # depth exhaustion: tiny books can't fill $1k -> ok=False, no signal
    thin = {k: _mkbook(*(v["bids"][0][0], v["asks"][0][0]), depth=0.001)
            for k, v in rich.items()}
    r3 = loop_edge(thin.get, "BTC-ETH", 1000, fee_bps=25)
    assert not r3["signal"] and not r3["fwd"]["ok"], r3
    print(f"triangular self-test ✓  fair->no signal ({r['fwd']['edge_bps']}bps), "
          f"planted 200bps->signal ({r2['best']['edge_bps']}bps net), "
          f"thin depth->blocked")


if __name__ == "__main__":
    _selftest()
