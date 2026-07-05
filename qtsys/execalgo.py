"""execalgo.py — execution-algorithm core: TWAP / VWAP / IS slice planning.

Turns "buy 5,000 X over 2 hours" into a concrete slice schedule plus an
honest cost estimate against the arrival price:

  twap        equal slices at equal intervals — the neutral baseline.
  vwap        slices weighted by the intraday volume profile (real minute
              bars when a fetcher is supplied, classic U-curve fallback),
              minimising participation-rate spikes.
  is_schedule Almgren-Chriss implementation-shortfall trajectory
              x(t) = X·sinh(κ(T−t))/sinh(κT): urgency (risk aversion)
              front-loads to cut timing risk, patience approaches TWAP.
              κ = sqrt(λσ²/η) — λ risk aversion, σ per-interval vol,
              η temporary-impact coefficient.

Cost model per slice: half-spread + temporary impact. For crypto the
impact is read from the REAL L2 book (walk the depth for the slice size —
same math the Microstructure Analyst records); otherwise a square-root
participation model η·σ·√(slice/ADV-interval).

This is a PLANNER: it emits the schedule and the cost expectation. Orders
still go one-by-one through the ExecutionGateway's risk checks — nothing
here bypasses them.

Run `python -m qtsys.execalgo` for self-tests.
"""
from __future__ import annotations

import math

# classic equity intraday U-curve (half-hour buckets, 6.5h session) — used
# when no real minute-volume fetcher is supplied
U_CURVE = (0.115, 0.085, 0.070, 0.062, 0.058, 0.055, 0.055, 0.058,
           0.062, 0.068, 0.075, 0.090, 0.147)

# urgency -> κT, the Almgren-Chriss trajectory-shape parameter (κ=sqrt(λσ²/η)
# × horizon). Deriving it from λ/σ/η needs impact coefficients calibrated to
# OUR OWN fills — data we don't have yet — so the practical control is the
# curve shape itself: 0 = risk-neutral (TWAP), ~4 = strongly front-loaded.
URGENCY = {"passive": 0.3, "neutral": 1.5, "urgent": 4.0}


def twap(qty: float, minutes: int, n_slices: int | None = None) -> list[dict]:
    n = n_slices or max(min(minutes // 5, 24), 2)
    per = qty / n
    step = minutes / n
    return [{"t_min": round(i * step, 1), "qty": round(per, 6),
             "pct": round(100 / n, 2)} for i in range(n)]


def vwap(qty: float, minutes: int, profile: list[float] | None = None,
         n_slices: int | None = None) -> list[dict]:
    n = n_slices or max(min(minutes // 5, 24), 2)
    prof = profile or list(U_CURVE)
    # resample the profile onto n slices
    w = []
    for i in range(n):
        lo, hi = i / n * len(prof), (i + 1) / n * len(prof)
        acc, j = 0.0, int(lo)
        while j < hi and j < len(prof):
            frac = min(hi, j + 1) - max(lo, j)
            acc += prof[j] * max(frac, 0)
            j += 1
        w.append(acc)
    tot = sum(w) or 1.0
    step = minutes / n
    return [{"t_min": round(i * step, 1), "qty": round(qty * w[i] / tot, 6),
             "pct": round(w[i] / tot * 100, 2)} for i in range(n)]


def is_schedule(qty: float, minutes: int, sigma_interval: float = 0.002,
                kappa_T: float | None = None, urgency: str = "neutral",
                n_slices: int | None = None) -> list[dict]:
    """Almgren-Chriss optimal liquidation trajectory, discretized.
    kappa_T overrides the urgency preset when given."""
    n = n_slices or max(min(minutes // 5, 24), 2)
    kappa = kappa_T if kappa_T is not None else URGENCY.get(urgency,
                                                            URGENCY["neutral"])
    step = minutes / n
    out = []
    if kappa < 1e-6:                          # λ→0: risk-neutral == TWAP
        return twap(qty, minutes, n)
    for i in range(n):
        t0, t1 = i / n, (i + 1) / n
        x0 = math.sinh(kappa * (1 - t0)) / math.sinh(kappa)
        x1 = math.sinh(kappa * (1 - t1)) / math.sinh(kappa)
        sl = qty * (x0 - x1)
        out.append({"t_min": round(i * step, 1), "qty": round(sl, 6),
                    "pct": round((x0 - x1) * 100, 2)})
    return out


def cost_estimate(slices: list[dict], price: float, spread: float,
                  sigma_interval: float = 0.002, adv_interval: float = 0.0,
                  eta: float = 0.1, l2_slip_fn=None) -> dict:
    """Expected cost vs arrival, in bps of notional.

    half-spread paid per slice + temporary impact: from the REAL L2 book via
    l2_slip_fn(notional)->bps when provided (crypto), else the square-root
    model eta·sigma·sqrt(slice_notional/adv_interval_notional)."""
    if not slices or price <= 0:
        return {"total_bps": None}
    tot_qty = sum(s["qty"] for s in slices)
    if tot_qty <= 0:
        return {"total_bps": None}
    half_spread_bps = spread / 2 / price * 1e4 if spread else 0.0
    impact_bps_w = 0.0
    for s in slices:
        notional = s["qty"] * price
        if l2_slip_fn is not None:
            try:
                slip = l2_slip_fn(notional) or 0.0
            except Exception:
                slip = 0.0
        elif adv_interval > 0:
            slip = eta * sigma_interval * math.sqrt(
                notional / max(adv_interval * price, 1e-9)) * 1e4
        else:
            slip = 0.0
        impact_bps_w += slip * (s["qty"] / tot_qty)
    return {"half_spread_bps": round(half_spread_bps, 2),
            "impact_bps": round(impact_bps_w, 2),
            "total_bps": round(half_spread_bps + impact_bps_w, 2),
            "benchmark": "arrival price (implementation shortfall)"}


def plan(qty: float, minutes: int, algo: str = "twap", price: float = 0.0,
         spread: float = 0.0, sigma_interval: float = 0.002,
         adv_interval: float = 0.0, urgency: str = "neutral",
         profile: list[float] | None = None, l2_slip_fn=None) -> dict:
    """One-call planner: schedule + cost, for /api/exec/plan and the agents."""
    algo = algo.lower()
    if algo == "vwap":
        slices = vwap(qty, minutes, profile)
    elif algo in ("is", "shortfall"):
        slices = is_schedule(qty, minutes, sigma_interval, urgency=urgency)
    else:
        algo, slices = "twap", twap(qty, minutes)
    cost = cost_estimate(slices, price, spread, sigma_interval,
                         adv_interval, l2_slip_fn=l2_slip_fn)
    return {"algo": algo, "minutes": minutes, "n_slices": len(slices),
            "slices": slices, "cost": cost, "urgency": urgency,
            "note": "planner output — every slice still passes the "
                    "ExecutionGateway risk checks individually"}


# ------------------------------------------------------------------ self-test
def _selftest():
    q, m = 10_000.0, 120
    tw = twap(q, m)
    assert abs(sum(s["qty"] for s in tw) - q) < 1e-3
    assert len({s["qty"] for s in tw}) == 1, "TWAP slices equal"
    vw = vwap(q, m)
    assert abs(sum(s["qty"] for s in vw) - q) < 1e-3
    assert vw[0]["qty"] > vw[len(vw) // 2]["qty"], "U-curve: open > midday"
    assert vw[-1]["qty"] > vw[len(vw) // 2]["qty"], "U-curve: close > midday"
    urgent = is_schedule(q, m, urgency="urgent")
    passive = is_schedule(q, m, urgency="passive")
    assert abs(sum(s["qty"] for s in urgent) - q) < 1e-3
    assert urgent[0]["qty"] > urgent[-1]["qty"] * 3, "urgent front-loads"
    spread_tw = max(s["qty"] for s in passive) - min(s["qty"] for s in passive)
    assert spread_tw < q * 0.02, "passive ≈ TWAP"
    c_small = cost_estimate(tw, 100.0, 0.02, adv_interval=1e6)
    c_big = cost_estimate(twap(q * 50, m), 100.0, 0.02, adv_interval=1e6)
    assert c_big["impact_bps"] > c_small["impact_bps"], "impact grows with size"
    # L2-driven cost: linear slip fn -> impact reflects book walk
    c_l2 = cost_estimate(tw, 100.0, 0.02, l2_slip_fn=lambda n: n / 1e4)
    assert c_l2["impact_bps"] > 0
    p = plan(q, m, "is", price=100.0, spread=0.02, urgency="urgent")
    assert p["n_slices"] == len(p["slices"]) and p["cost"]["total_bps"] is not None
    print(f"execalgo self-test ✓  TWAP equal, VWAP U-curve, IS front-load "
          f"(urgent first/last = {urgent[0]['qty']:.0f}/{urgent[-1]['qty']:.0f}), "
          f"impact monotone ({c_small['impact_bps']} -> {c_big['impact_bps']} bps)")


if __name__ == "__main__":
    _selftest()
