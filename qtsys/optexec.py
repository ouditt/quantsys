"""optexec.py — options execution skill: defined-risk spreads for plan ideas.

Turns a directional, DSR-verified plan idea into a DEFINED-RISK vertical
spread from the live surface-priced chain:

  LONG underlying  -> bull call spread (buy ATM call, sell OTM call)
  SHORT underlying -> bear put spread  (buy ATM put,  sell OTM put)

Why spreads and never naked options: max loss is EXACTLY the debit paid, known
at entry — so sizing is honest ("contracts = risk budget // max loss") and no
tail can exceed the plan's per-idea risk. The engine will not construct any
structure whose worst case isn't fully paid for up front.

Exit rules (managed by the auto-trader's monitor):
  target : spread value >= debit + PT x max-profit   (default PT=60%)
  stop   : spread value <= (1-SL) x debit            (default SL=50% of debit)
  time   : <= TIME_EXIT_DAYS to expiry — theta decay and pin risk aren't
           worth the last day; close and move on.

Pure functions — chain in, structure out. Self-test: python -m qtsys.optexec
"""
from __future__ import annotations

import datetime

from .optstrat import MULT, build

PROFIT_TARGET = 0.60      # close at 60% of max profit captured
STOP_FRAC = 0.50          # close after losing 50% of the debit
TIME_EXIT_DAYS = 1        # close with <=1 day to expiry
MAX_CONTRACTS = 10


def pick_spread(chain: list[dict], spot: float, side: str, risk_amt: float,
                expiration: str = "", max_contracts: int = MAX_CONTRACTS) -> dict | None:
    """Build + size the defined-risk vertical for a plan idea. Returns None
    when the chain can't support a clean two-sided structure or the risk
    budget doesn't cover even one contract's max loss."""
    preset = "bull_call" if side in ("LONG", "buy") else "bear_put"
    st = build(chain, spot, preset)
    if not st or st["net_cost"] <= 0:            # only DEBIT verticals: the
        return None                              # worst case must be prepaid
    max_loss_per = abs(st["max_loss"])           # $ per contract, known
    if max_loss_per <= 0 or risk_amt < max_loss_per:
        return None
    contracts = min(int(risk_amt // max_loss_per), max_contracts)
    debit = st["net_cost"]
    return {
        "kind": "ospread", "preset": preset, "side": side,
        "expiration": expiration, "contracts": contracts,
        "legs": [{"symbol": l["symbol"], "qty": l["qty"], "right": l["right"],
                  "strike": l["strike"], "mid": l["mid"]} for l in st["legs"]],
        "debit_per": round(debit, 2),                    # $ per contract
        "max_loss_per": round(-max_loss_per, 2),
        "max_profit_per": round(st["max_profit"], 2),
        "breakevens": st["breakevens"],
        "total_debit": round(debit * contracts, 2),
        "total_max_loss": round(-max_loss_per * contracts, 2),
        "exit": {"target_value": round(debit + PROFIT_TARGET * st["max_profit"], 2),
                 "stop_value": round((1 - STOP_FRAC) * debit, 2),
                 "time_exit_days": TIME_EXIT_DAYS},
    }


# structures the auto-trader may enter unattended: all DEFINED-RISK (worst case
# known and, for debits, prepaid). Cash-secured puts / covered calls need share
# collateral or assignment handling, so they are library-only (human/INBOX).
AUTO_STRUCTURES = {"long_call", "long_put", "bull_call", "bear_put",
                   "bull_put", "bear_call", "iron_condor", "straddle", "strangle"}


def pick_structure(chain: list[dict], spot: float, preset: str, risk_amt: float,
                   expiration: str = "", max_contracts: int = MAX_CONTRACTS,
                   view: str = "") -> dict | None:
    """Build + size ANY defined-risk structure (debit or credit) for a given
    risk budget. Generalises pick_spread beyond debit verticals so the vol skill
    can trade straddles/strangles (buy vol), iron condors and credit verticals
    (sell vol). Sizing is off the KNOWN max loss per contract; exits are the
    same P&L-vs-entry rule the monitor already applies, expressed in the signed
    per-contract value space so debit and credit share one code path."""
    from .optstrat import build
    st = build(chain, spot, preset)
    if not st:
        return None
    entry = st["net_cost"]                       # signed $/contract: +debit / -credit
    max_loss_per = abs(st["max_loss"])           # known worst case $/contract
    max_profit_per = st["max_profit"]
    if max_loss_per <= 0 or max_profit_per <= 0:
        return None
    # sign sanity: a debit structure must actually cost money and a credit one
    # must actually pay it. A mismatch means stale/garbage quotes (common on
    # illiquid chains) — refuse it rather than trade a nonsense structure.
    _DEBIT = {"long_call", "long_put", "bull_call", "bear_put", "straddle", "strangle"}
    _CREDIT = {"bull_put", "bear_call", "iron_condor"}
    if (preset in _DEBIT and entry <= 0) or (preset in _CREDIT and entry >= 0):
        return None
    if risk_amt < max_loss_per:                  # can't afford one contract's risk
        return None
    contracts = min(int(risk_amt // max_loss_per), max_contracts)
    if contracts < 1:
        return None
    return {
        "kind": "ospread", "preset": preset, "structure": preset,
        "side": view or ("LONG" if entry > 0 else "NEUTRAL"),
        "expiration": expiration, "contracts": contracts,
        "legs": [{"symbol": l["symbol"], "qty": l["qty"], "right": l["right"],
                  "strike": l["strike"], "mid": l["mid"]} for l in st["legs"]],
        "debit_per": round(entry, 2),                    # signed entry value $/contract
        "flow": "debit" if entry > 0 else "credit",
        "max_loss_per": round(-max_loss_per, 2),
        "max_profit_per": round(max_profit_per, 2),
        "breakevens": st["breakevens"], "greeks": st.get("greeks", {}),
        "total_debit": round(entry * contracts, 2),
        "total_max_loss": round(-max_loss_per * contracts, 2),
        "total_max_profit": round(max_profit_per * contracts, 2),
        # exits in the SAME signed value space spread_value() returns: take
        # PROFIT_TARGET of the max gain, stop after STOP_FRAC of the max loss.
        "exit": {"target_value": round(entry + PROFIT_TARGET * max_profit_per, 2),
                 "stop_value": round(entry + STOP_FRAC * st["max_loss"], 2),
                 "time_exit_days": TIME_EXIT_DAYS},
    }


def spread_value(legs: list[dict], quote_fn) -> float | None:
    """Current $ value of the spread per contract: sum of signed leg quotes
    x multiplier. None if any leg can't be priced."""
    total = 0.0
    for l in legs:
        try:
            px = quote_fn(l["symbol"])
        except Exception:
            return None
        if not px or px != px:
            return None
        total += (1 if l["qty"] > 0 else -1) * px
    return round(total * MULT, 2)


def days_to_expiry(expiration: str) -> int | None:
    try:
        return (datetime.date.fromisoformat(expiration)
                - datetime.date.today()).days
    except Exception:
        return None


def exit_check(value_now: float | None, spread: dict) -> str | None:
    """'target' | 'stop' | 'expiry' | None — the monitor's decision rule."""
    ex = spread["exit"]
    dte = days_to_expiry(spread.get("expiration", ""))
    if dte is not None and dte <= ex["time_exit_days"]:
        return "expiry"
    if value_now is None:
        return None
    if value_now >= ex["target_value"]:
        return "target"
    if value_now <= ex["stop_value"]:
        return "stop"
    return None


# ------------------------------------------------------------------ self-test
def _selftest():
    import math
    def mk(k):                                   # synthetic chain around 100
        tv = 5.0 * math.exp(-((k - 100) / 15.0) ** 2)
        return {"strike": float(k),
                "call": {"mid": max(100 - k, 0) + tv, "delta": .5, "gamma": .02,
                         "theta": -.05, "vega": .1, "symbol": f"C{int(k)}"},
                "put": {"mid": max(k - 100, 0) + tv, "delta": -.5, "gamma": .02,
                        "theta": -.05, "vega": .1, "symbol": f"P{int(k)}"}}
    chain = [mk(k) for k in range(80, 121, 5)]
    exp = str(datetime.date.today() + datetime.timedelta(days=14))

    sp = pick_spread(chain, 100.0, "LONG", risk_amt=1500.0, expiration=exp)
    assert sp and sp["preset"] == "bull_call" and sp["contracts"] >= 1
    assert sp["total_max_loss"] >= -1500.0, "sizing never exceeds risk budget"
    assert sp["legs"][0]["qty"] > 0 > sp["legs"][1]["qty"], "buy low, sell high strike"
    assert sp["exit"]["target_value"] > sp["debit_per"] > sp["exit"]["stop_value"]

    bp = pick_spread(chain, 100.0, "SHORT", risk_amt=1500.0, expiration=exp)
    assert bp and bp["preset"] == "bear_put"

    tiny = pick_spread(chain, 100.0, "LONG", risk_amt=5.0, expiration=exp)
    assert tiny is None, "budget below one contract's max loss -> no trade"

    # exit rules: value path -> hold, target, stop, and time exit
    q = {l["symbol"]: l["mid"] for l in sp["legs"]}
    v0 = spread_value(sp["legs"], q.get)
    assert v0 is not None and abs(v0 - sp["debit_per"]) < 1e-6, "entry value = debit"
    assert exit_check(v0, sp) is None, "at entry -> hold"
    assert exit_check(sp["exit"]["target_value"] + 1, sp) == "target"
    assert exit_check(sp["exit"]["stop_value"] - 1, sp) == "stop"
    near = dict(sp, expiration=str(datetime.date.today()))
    assert exit_check(v0, near) == "expiry", "T-0 -> time exit"
    assert spread_value(sp["legs"], lambda s: None) is None, "unquotable -> None"

    # ---- pick_structure: credit + long-vol, defined risk, unified exits ----
    straddle = pick_structure(chain, 100.0, "straddle", risk_amt=2000.0, expiration=exp)
    assert straddle and straddle["preset"] == "straddle" and straddle["flow"] == "debit"
    assert straddle["total_max_loss"] >= -2000.0, "vol buy sized within budget"
    assert len(straddle["legs"]) == 2 and straddle["side"] == "LONG"
    condor = pick_structure(chain, 100.0, "iron_condor", risk_amt=2000.0,
                            expiration=exp, view="NEUTRAL")
    assert condor and len(condor["legs"]) == 4 and condor["flow"] == "credit"
    assert condor["debit_per"] < 0, "credit structure: negative entry value"
    assert condor["total_max_loss"] < 0 and condor["total_max_profit"] > 0
    # exits straddle the entry value on both sides (take profit above, stop below)
    assert condor["exit"]["stop_value"] < condor["debit_per"] < condor["exit"]["target_value"]
    bp = pick_structure(chain, 100.0, "bull_put", risk_amt=2000.0, expiration=exp)
    assert bp and bp["flow"] == "credit" and bp["contracts"] >= 1
    # the monitor's realized math works on the signed entry for a credit too:
    # value decaying toward 0 is a PROFIT vs the negative entry
    entry_val = spread_value(condor["legs"], {l["symbol"]: l["mid"] for l in condor["legs"]}.get)
    assert abs(entry_val - condor["debit_per"]) < 1e-6, "entry value = net credit"
    assert AUTO_STRUCTURES == {"long_call", "long_put", "bull_call", "bear_put",
                               "bull_put", "bear_call", "iron_condor",
                               "straddle", "strangle"}
    print(f"optexec self-test ✓  bull-call sized {sp['contracts']}x within "
          f"risk (maxL {sp['total_max_loss']}), bear-put, budget floor, "
          f"target/stop/time exits; pick_structure straddle/condor/credit "
          f"defined-risk + unified signed exits")


if __name__ == "__main__":
    _selftest()
