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
# Spread WIDTH (strike steps between the long and short leg) is parametric:
# width 2 is the default desk shape; width 1 is the narrowest the chain allows,
# which shrinks max loss per contract — how a SMALL account affords structures.
PRESETS = {
    "long_call":   [("call", 0, +1)],
    "long_put":    [("put", 0, +1)],
    "straddle":    [("call", 0, +1), ("put", 0, +1)],
    "strangle":    [("call", +1, +1), ("put", -1, +1)],
    "bull_call":   [("call", 0, +1), ("call", +2, -1)],
    "bear_put":    [("put", 0, +1), ("put", -2, -1)],
    # credit verticals — sell the near-OTM leg, buy a further-OTM wing. Favoured
    # when IV is rich: collect premium, still fully defined risk.
    "bull_put":    [("put", -1, -1), ("put", -3, +1)],
    "bear_call":   [("call", +1, -1), ("call", +3, +1)],
    "iron_condor": [("put", -1, -1), ("put", -3, +1),
                    ("call", +1, -1), ("call", +3, +1)],
}


def _preset_legs(preset: str, width: int | None) -> list | None:
    """Leg spec for a preset at a given spread width (strike steps between the
    bought and sold leg). None width = the PRESETS default shape."""
    if width is None:
        return PRESETS.get(preset)
    w = max(1, int(width))
    shapes = {
        "bull_call":   [("call", 0, +1), ("call", +w, -1)],
        "bear_put":    [("put", 0, +1), ("put", -w, -1)],
        "bull_put":    [("put", -1, -1), ("put", -1 - w, +1)],
        "bear_call":   [("call", +1, -1), ("call", +1 + w, +1)],
        "iron_condor": [("put", -1, -1), ("put", -1 - w, +1),
                        ("call", +1, -1), ("call", +1 + w, +1)],
    }
    return shapes.get(preset, PRESETS.get(preset))   # width-less presets unchanged
LABEL = {"long_call": "Long call", "long_put": "Long put",
         "straddle": "Straddle", "strangle": "Strangle",
         "bull_call": "Bull call spread", "bear_put": "Bear put spread",
         "bull_put": "Bull put spread (credit)",
         "bear_call": "Bear call spread (credit)",
         "iron_condor": "Iron condor"}

# view + IV regime -> the structure that expresses it. Mirrors the standard
# desk playbook: buy premium when it's cheap, sell it when it's rich, always
# through a defined-risk structure.
_SELECT = {
    ("bullish", "rich"): "bull_put",   ("bullish", "cheap"): "bull_call",
    ("bearish", "rich"): "bear_call",  ("bearish", "cheap"): "bear_put",
    ("neutral", "rich"): "iron_condor",("neutral", "cheap"): "strangle",
    ("big_move", "cheap"): "straddle", ("big_move", "rich"): "strangle",
}


def select(view: str, vol_regime: str) -> str:
    """Map a directional/vol view ('bullish'|'bearish'|'neutral'|'big_move') and
    an IV regime ('rich'|'cheap'|'normal') to the right defined-risk structure."""
    reg = "rich" if vol_regime == "rich" else "cheap"     # normal -> lean debit
    return _SELECT.get((view, reg), "bull_call" if view == "bullish"
                       else "bear_put" if view == "bearish"
                       else "iron_condor" if reg == "rich" else "straddle")


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


def build(chain, spot, preset="straddle", width=None):
    """Assemble a preset. Returns None if the needed strikes aren't quotable.
    width (verticals/condors only): strike steps between the bought and sold
    leg — width 1 is the narrowest structure the chain offers (smallest max
    loss per contract, the small-account shape); None = default (2)."""
    if preset not in PRESETS or not chain or not spot:
        return None
    legs = []
    picked = set()
    for right, step, qty in _preset_legs(preset, width):
        pk = _pick(chain, spot, right, step)
        if not pk:
            return None
        if (right, pk["strike"]) in picked:      # clamped onto the same strike
            return None                          # -> degenerate spread, refuse
        picked.add((right, pk["strike"]))
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
    # credit verticals: net credit (negative net_cost), defined risk
    bp = build(chain, 100.0, "bull_put")
    assert bp and len(bp["legs"]) == 2 and bp["kind"] == "credit", bp["net_cost"]
    assert bp["max_loss"] < 0 and bp["max_profit"] > 0
    bc2 = build(chain, 100.0, "bear_call")
    assert bc2 and bc2["kind"] == "credit"
    # regime selection playbook
    assert select("bullish", "rich") == "bull_put"
    assert select("bullish", "cheap") == "bull_call"
    assert select("neutral", "rich") == "iron_condor"
    assert select("big_move", "cheap") == "straddle"
    # width fitting: a width-1 vertical is strictly narrower -> smaller max loss
    bc_w2 = build(chain, 100.0, "bull_call")            # default width 2
    bc_w1 = build(chain, 100.0, "bull_call", width=1)
    assert bc_w1 and (bc_w1["legs"][1]["strike"] - bc_w1["legs"][0]["strike"]
                      < bc_w2["legs"][1]["strike"] - bc_w2["legs"][0]["strike"])
    assert abs(bc_w1["max_loss"]) < abs(bc_w2["max_loss"]), "narrower = cheaper risk"
    ic_w1 = build(chain, 100.0, "iron_condor", width=1)
    assert ic_w1 and len(ic_w1["legs"]) == 4
    # degenerate (both legs clamp to the same strike) -> refused, not nonsense
    tiny_chain = [mk(100)]
    assert build(tiny_chain, 100.0, "bull_call", width=1) is None
    miss = build([{"strike": 100.0, "call": {"mid": None}, "put": {"mid": None}}],
                 100.0, "straddle")
    assert miss is None, "unquotable -> None, not a crash"
    print("optstrat self-test ✓  straddle 2 BEs + long vega, bull-call capped "
          "risk, iron-condor 4 legs, unquotable->None")


if __name__ == "__main__":
    _selftest()
