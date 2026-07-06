"""optstrat.py — options strategy builder: multi-leg payoff, greeks, risk.

Given a live chain (strikes carrying call/put mid + surface greeks) and spot,
assemble a named structure around the money and return everything a desk needs
to judge it: the legs, net debit/credit, max profit / max loss, breakevens,
and net position greeks — plus a sampled expiry-payoff curve for plotting.

Payoff is at expiry (intrinsic) per $1 of underlying × 100-multiplier per
contract; greeks are the current surface greeks summed across legs (per 1
contract). Pure and testable — no market data here.
"""
from __future__ import annotations

MULT = 100.0   # US equity option contract multiplier

# preset -> list of (right, strike-picker, qty). strike-picker walks OTM steps
# from ATM: 0 = nearest ATM, +k = k strikes above, -k = k below.
PRESETS = {
    "long_call":   [("call", 0, +1)],
    "long_put":    [("put", 0, +1)],
    "straddle":    [("call", 0, +1), ("put", 0, +1)],
    "strangle":    [("call", +1, +1), ("put", -1, +1)],
    "bull_call":   [("call", 0, +1), ("call", +2, -1)],
    "bear_put":    [("put", 0, +1), ("put", -2, -1)],
    "iron_condor": [("put", -1, -1), ("put", -3, +1),
                    ("call", +1, -1), ("call", +3, +1)],
}
LABEL = {"long_call": "Long call", "long_put": "Long put",
         "straddle": "Straddle", "strangle": "Strangle",
         "bull_call": "Bull call spread", "bear_put": "Bear put spread",
         "iron_condor": "Iron condor"}


def _strikes(chain):
    return sorted(r["strike"] for r in chain)


def _pick(chain, spot, right, step):
    ks = _strikes(chain)
    if not ks:
        return None
    atm = min(range(len(ks)), key=lambda i: abs(ks[i] - spot))
    j = min(max(atm + step, 0), len(ks) - 1)
    k = ks[j]
    row = next((r for r in chain if r["strike"] == k), None)
    side = row.get(right) if row else None
    if not side or side.get("mid") in (None, 0):
        return None
    return {"right": right, "strike": k, **side}


def _leg_payoff(leg, s_t):
    k, prem, qty = leg["strike"], leg["mid"], leg["qty"]
    intr = max(s_t - k, 0.0) if leg["right"] == "call" else max(k - s_t, 0.0)
    return qty * (intr - prem) * MULT


def build(chain, spot, preset="straddle"):
    """Assemble a preset. Returns None if the needed strikes aren't quotable."""
    if preset not in PRESETS or not chain or not spot:
        return None
    legs = []
    for right, step, qty in PRESETS[preset]:
        pk = _pick(chain, spot, right, step)
        if not pk:
            return None
        pk["qty"] = qty
        legs.append(pk)
    net_cost = sum(l["qty"] * l["mid"] * MULT for l in legs)     # +debit / -credit
    g = {k: round(sum(l["qty"] * (l.get(k) or 0) for l in legs), 4)
         for k in ("delta", "gamma", "theta", "vega")}
    # payoff curve over a wide strike grid
    lo, hi = spot * 0.6, spot * 1.4
    n = 121
    xs = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    ys = [round(sum(_leg_payoff(l, s) for l in legs), 2) for s in xs]
    max_p, max_l = max(ys), min(ys)
    # breakevens: sign changes of the payoff curve
    bes = []
    for i in range(1, n):
        a, b = ys[i - 1], ys[i]
        if a == b or not ((a <= 0 <= b) or (a >= 0 >= b)):
            continue
        be = round(xs[i - 1] + (-a / (b - a)) * (xs[i] - xs[i - 1]), 2)
        if not bes or abs(be - bes[-1]) > 1e-6:      # dedup exact-node crossings
            bes.append(be)
    return {
        "preset": preset, "label": LABEL[preset], "spot": spot,
        "legs": [{"right": l["right"], "strike": l["strike"], "qty": l["qty"],
                  "mid": l["mid"], "symbol": l.get("symbol")} for l in legs],
        "net_cost": round(net_cost, 2),
        "kind": "debit" if net_cost > 0 else "credit",
        "max_profit": round(max_p, 2), "max_loss": round(max_l, 2),
        "breakevens": bes, "greeks": g,
        "payoff": [[round(x, 2), y] for x, y in zip(xs, ys)],
    }


def _selftest():
    # synthetic chain around spot 100, flat-ish mids
    import math
    def mk(k):
        tv = 5.0 * math.exp(-((k - 100) / 15.0) ** 2)   # time value peaks ATM
        c = max(100 - k, 0) + tv                        # (spot=100)
        p = max(k - 100, 0) + tv
        return {"strike": float(k),
                "call": {"mid": c, "delta": 0.5, "gamma": 0.02, "theta": -0.05,
                         "vega": 0.1, "symbol": f"C{k}"},
                "put": {"mid": p, "delta": -0.5, "gamma": 0.02, "theta": -0.05,
                        "vega": 0.1, "symbol": f"P{k}"}}
    chain = [mk(k) for k in range(80, 121, 5)]
    st = build(chain, 100.0, "straddle")
    assert st and len(st["legs"]) == 2 and st["kind"] == "debit"
    assert st["max_loss"] < 0 and st["max_profit"] > 0
    assert len(st["breakevens"]) == 2, st["breakevens"]      # straddle: two BEs
    assert abs(st["greeks"]["vega"] - 0.2) < 1e-9            # both legs long vega
    bc = build(chain, 100.0, "bull_call")
    assert bc["max_profit"] > 0 and bc["max_loss"] < 0
    # bull call: capped max profit at the short strike, defined risk
    assert bc["max_profit"] <= (bc["legs"][1]["strike"]
                                - bc["legs"][0]["strike"]) * MULT + 1
    ic = build(chain, 100.0, "iron_condor")
    assert ic["kind"] in ("credit", "debit") and len(ic["legs"]) == 4
    assert len(ic["breakevens"]) == 2, ic["breakevens"]
    miss = build([{"strike": 100.0, "call": {"mid": None}, "put": {"mid": None}}],
                 100.0, "straddle")
    assert miss is None, "unquotable -> None, not a crash"
    print("optstrat self-test ✓  straddle 2 BEs + long vega, bull-call capped "
          "risk, iron-condor 4 legs, unquotable->None")


if __name__ == "__main__":
    _selftest()
