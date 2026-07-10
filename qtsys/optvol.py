"""optvol.py — the options VOLATILITY skill (idea generation).

The rest of the options stack expresses a *directional* view through a
defined-risk structure. This skill adds the missing leg: trading VOLATILITY
itself, independent of any directional signal, using the variance risk premium —
implied vol (from the live chain) versus realized vol (from price bars):

  IV rich  (implied >> realized)  -> SELL premium: iron condor (neutral, range).
  IV cheap (implied << realized)  -> BUY premium:  long straddle (a big move is
                                     under-priced).
  a directional idea in a rich-IV name -> a CREDIT vertical instead of a debit
                                     one (collect the premium, same direction).

Everything it emits is a fully defined-risk structure sized off its known max
loss, tagged asset_class "Option", so it flows through the SAME desk
deliberation (committee) and the SAME auto-trader as equities and crypto.

Pure functions + a thin `ideas()` orchestrator that takes injected accessors so
it stays testable off-server. Self-test: python -m qtsys.optvol
"""
from __future__ import annotations

import math

# variance-risk-premium thresholds on the IV/realized ratio
RICH = 1.25       # implied >= 1.25x realized -> vol is rich, sell it
CHEAP = 0.90      # implied <= 0.90x realized -> vol is cheap, buy it


def realized_vol(closes: list[float], window: int = 20) -> float | None:
    """Annualised close-to-close realized volatility from the last `window`
    daily bars. None if there isn't enough clean data."""
    cs = [c for c in (closes or []) if c and c > 0][-(window + 1):]
    if len(cs) < max(5, window // 2):
        return None
    rets = [math.log(cs[i] / cs[i - 1]) for i in range(1, len(cs))]
    n = len(rets)
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / max(n - 1, 1)
    return math.sqrt(var) * math.sqrt(252)


def atm_iv(chain_rows: list[dict], spot: float) -> float | None:
    """ATM implied vol: average of the nearest-strike call & put IVs."""
    rows = [r for r in (chain_rows or []) if r.get("strike")]
    if not rows or not spot:
        return None
    atm = min(rows, key=lambda r: abs(r["strike"] - spot))
    ivs = [leg.get("iv") for leg in (atm.get("call"), atm.get("put"))
           if leg and leg.get("iv") and leg["iv"] > 0]
    return (sum(ivs) / len(ivs)) if ivs else None


def vol_regime(iv: float | None, rvol: float | None) -> tuple[str, float | None]:
    """('rich'|'cheap'|'normal', iv/rvol ratio). Needs both vols."""
    if not iv or not rvol or rvol <= 0:
        return "normal", None
    ratio = iv / rvol
    if ratio >= RICH:
        return "rich", ratio
    if ratio <= CHEAP:
        return "cheap", ratio
    return "normal", ratio


def _idea(symbol, structure, view, iv, rvol, ratio, note):
    return {"symbol": symbol, "kind": "option_structure", "structure": structure,
            "view": view, "asset_class": "Option", "side": view.upper(),
            "iv": round(iv, 4) if iv else None,
            "rvol": round(rvol, 4) if rvol else None,
            "iv_rv": round(ratio, 2) if ratio else None,
            "strategy": f"vol_{structure}", "source": "optvol",
            "rationale": note}


def ideas(underlyings: list[str], chain_of, bars_of, *, directional: dict = None,
          max_ideas: int = 3) -> list[dict]:
    """Generate volatility option ideas.

      underlyings : symbols to scan (liquid optionable names).
      chain_of(sym) -> {"spot":.., "chain":[rows], "expiration":".."} or None.
      bars_of(sym)  -> [closes]  (recent daily closes).
      directional   -> {sym: "bullish"|"bearish"} views from the equity scan; a
                       rich-IV name with a directional view becomes a CREDIT
                       vertical rather than a neutral condor.

    Returns defined-risk structure ideas (unsized — the executor sizes to the
    risk budget), newest-conviction first, capped at max_ideas.
    """
    from .optstrat import select
    directional = directional or {}
    out: list[dict] = []
    for sym in underlyings:
        try:
            ch = chain_of(sym)
        except Exception:
            ch = None
        if not ch or not ch.get("chain") or not ch.get("spot"):
            continue
        iv = atm_iv(ch["chain"], ch["spot"])
        rvol = realized_vol(bars_of(sym) or [])
        regime, ratio = vol_regime(iv, rvol)
        if regime == "normal":
            continue                              # no vol edge -> no trade
        view = directional.get(sym)
        if view in ("bullish", "bearish"):
            structure = select(view, regime)      # directional: debit or credit by regime
            v = view
        elif regime == "rich":
            structure, v = "iron_condor", "neutral"     # sell rich vol, range
        else:                                     # cheap
            structure, v = "straddle", "big_move"       # buy cheap vol, big move
        note = (f"IV {iv:.0%} vs realized {rvol:.0%} (x{ratio:.2f}) — "
                f"{'rich, sell premium' if regime == 'rich' else 'cheap, buy premium'} "
                f"via {structure.replace('_', ' ')}")
        out.append({**_idea(sym, structure, v, iv, rvol, ratio, note),
                    "expiration": ch.get("expiration", ""),
                    "_edge": abs((ratio or 1) - 1)})
    out.sort(key=lambda i: i["_edge"], reverse=True)
    for i in out:
        i.pop("_edge", None)
    return out[:max_ideas]


# ------------------------------------------------------------------ self-test
def _selftest():
    import datetime

    def mk_chain(atm_iv_val):
        def row(k):
            tv = 5.0 * math.exp(-((k - 100) / 15.0) ** 2)
            return {"strike": float(k),
                    "call": {"mid": max(100 - k, 0) + tv, "iv": atm_iv_val,
                             "delta": .5, "vega": .1, "symbol": f"C{k}"},
                    "put": {"mid": max(k - 100, 0) + tv, "iv": atm_iv_val,
                            "delta": -.5, "vega": .1, "symbol": f"P{k}"}}
        return {"spot": 100.0, "chain": [row(k) for k in range(80, 121, 5)],
                "expiration": str(datetime.date.today() + datetime.timedelta(days=21))}

    # realized ~ 16% annualised from a gently trending series
    import random
    random.seed(1)
    closes = [100.0]
    for _ in range(40):
        closes.append(closes[-1] * math.exp(random.gauss(0, 0.01)))
    rv = realized_vol(closes)
    assert rv and 0.05 < rv < 0.40, rv

    # IV set RELATIVE to the actual realized vol so regimes are deterministic:
    # rich (2x rv) -> sell; cheap (0.5x rv) -> buy; matched (1x rv) -> no trade.
    chains = {"RICH": mk_chain(rv * 2.0), "CHEAP": mk_chain(rv * 0.5),
              "MEH": mk_chain(rv)}
    ii = ideas(["RICH", "CHEAP", "MEH"], chains.get, lambda s: closes)
    by = {i["symbol"]: i for i in ii}
    assert by["RICH"]["structure"] == "iron_condor" and by["RICH"]["view"] == "neutral"
    assert by["CHEAP"]["structure"] == "straddle" and by["CHEAP"]["view"] == "big_move"
    assert "MEH" not in by, "normal IV -> no vol trade"
    # a directional view flips a rich-IV name to a CREDIT vertical
    dd = ideas(["RICH"], chains.get, lambda s: closes,
               directional={"RICH": "bullish"})
    assert dd[0]["structure"] == "bull_put", dd[0]["structure"]
    # regime math
    assert vol_regime(0.6, 0.16)[0] == "rich"
    assert vol_regime(0.05, 0.16)[0] == "cheap"
    assert vol_regime(0.16, 0.16)[0] == "normal"
    assert vol_regime(None, 0.16)[0] == "normal"
    print("optvol self-test ✓  realized vol, ATM IV, regime (rich/cheap/normal), "
          "condor on rich + straddle on cheap + credit-vertical on directional")


if __name__ == "__main__":
    _selftest()
