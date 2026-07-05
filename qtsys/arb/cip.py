"""cip.py — covered-interest-parity monitor (ANALYSIS ONLY, by design).

From FRED policy/bill rates this computes, per FX pair: the 3-month rate
differential and the CIP-theoretical forward points off the live spot,
F = S · e^{(r_dom − r_for)·T} (USD legs quoted as XXX/USD).

Two honest limitations, stated up front:
  - the true CIP *deviation* (cross-currency basis) needs live FX FORWARD
    quotes, which no free feed provides — so this reports the theoretical
    fair forward, not a tradable mispricing;
  - Alpaca carries no FX forwards/swaps anyway, so nothing here could be
    hedged or executed. The skill exists to inform the desk (carry regime,
    rate-differential shifts), not to trade.
"""
from __future__ import annotations

import math

# pair -> (foreign-rate FRED series, note); domestic (USD) side is DGS3MO
PAIRS = {
    "EURUSD": ("ECBDFR", "ECB deposit facility"),
    "GBPUSD": ("IUDSOIA", "SONIA o/n"),
}
US_SERIES = "DGS3MO"
T_YEARS = 0.25                       # 3-month tenor


def theoretical_forward(spot: float, r_dom_pct: float, r_for_pct: float,
                        T: float = T_YEARS) -> dict:
    F = spot * math.exp((r_dom_pct - r_for_pct) / 100.0 * T)
    return {"fwd": round(F, 6), "points": round((F - spot) * 1e4, 1),
            "carry_bps_ann": round((r_dom_pct - r_for_pct) * 100, 1)}


def snapshot(fred_latest, spots: dict[str, float]) -> list[dict]:
    """fred_latest: series_id -> latest value (pct) or None (intel._fred_latest).
    spots: pair -> live spot. Best-effort; skips pairs with missing inputs."""
    out = []
    try:
        r_us = fred_latest(US_SERIES)
    except Exception:
        r_us = None
    if r_us is None:
        return out
    for pair, (series, note) in PAIRS.items():
        spot = spots.get(pair)
        try:
            r_f = fred_latest(series)
        except Exception:
            r_f = None
        if not spot or r_f is None:
            continue
        # pair is XXX/USD: USD is the DOMESTIC (quote) currency
        th = theoretical_forward(spot, float(r_us), float(r_f))
        out.append({"pair": pair, "spot": spot, "r_usd_pct": float(r_us),
                    "r_for_pct": float(r_f), "r_for_src": note, **th,
                    "tenor": "3M",
                    "note": "theoretical CIP forward — deviation needs live "
                            "forward quotes; not tradable on this venue"})
    return out


def _selftest():
    # deterministic: 5% USD vs 3% EUR, spot 1.10, 3M
    th = theoretical_forward(1.10, 5.0, 3.0)
    expect = 1.10 * math.exp(0.02 * 0.25)
    assert abs(th["fwd"] - round(expect, 6)) < 1e-9, th
    assert th["points"] > 0 and th["carry_bps_ann"] == 200.0, th
    # inverted differential -> negative points
    th2 = theoretical_forward(1.10, 2.0, 4.0)
    assert th2["points"] < 0, th2
    rows = snapshot(lambda s: {"DGS3MO": 5.0, "ECBDFR": 3.0,
                               "IUDSOIA": 4.5}.get(s),
                    {"EURUSD": 1.10, "GBPUSD": 1.30})
    assert len(rows) == 2 and rows[0]["fwd"] > 0
    missing = snapshot(lambda s: None, {"EURUSD": 1.10})
    assert missing == []
    print(f"cip self-test ✓  fwd {th['fwd']} (+{th['points']}pts), "
          f"inverted -> {th2['points']}pts, missing-rate -> skip")


if __name__ == "__main__":
    _selftest()
