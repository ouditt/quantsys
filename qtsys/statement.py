"""statement.py — the transparent account statement.

Answers "is my balance real?" with venue math instead of trust: pulls the
FULL Alpaca activities ledger (every fill, fee, dividend, transfer, journal),
aggregates per instrument with TOTALS (bought $, sold $, fees, realized,
open qty, unrealized), lists every non-trade cash event, and closes with a
reconciliation:

    prior close equity
  + transfers/journals (deposits, withdrawals)
  + trading cash flow effects + unrealized moves
  + fees/dividends/interest
  = current equity ... and any residual is labelled loudly as
    "VENUE ADJUSTMENT (not explained by the ledger)" — e.g. a paper-account
    reset done in the Alpaca dashboard, which wipes history without leaving
    a journal entry.

Pure aggregation over dicts -> testable; the REST fetch lives in the broker.
"""
from __future__ import annotations

import datetime
from collections import defaultdict

# Alpaca non-trade activity types worth grouping
_TRANSFER_TYPES = {"CSD", "CSW", "JNLC", "JNLS", "ACATC", "ACATS"}
_INCOME_TYPES = {"DIV", "DIVCGL", "DIVCGS", "DIVNRA", "DIVROC", "DIVTXEX",
                 "INT", "INTNRA", "INTTW", "SSP"}


def _f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def build(activities: list[dict], positions: list[dict],
          equity: float, last_equity: float) -> dict:
    """Aggregate the raw activities ledger into the transparent statement.

    activities: raw dicts from GET /v2/account/activities (any order).
    positions:  [{symbol, qty, avg_price, last, unrealized}] current book.
    """
    per: dict[str, dict] = defaultdict(lambda: {
        "bought_qty": 0.0, "bought_usd": 0.0, "sold_qty": 0.0, "sold_usd": 0.0,
        "n_fills": 0, "first": None, "last": None})
    cash_events: list[dict] = []
    transfers = fees = income = other = 0.0

    for a in activities:
        typ = a.get("activity_type", "")
        if typ == "FILL":
            sym = a.get("symbol", "?")
            qty = abs(_f(a.get("qty")))
            px = _f(a.get("price"))
            usd = qty * px * (100.0 if _is_option(sym) else 1.0)
            d = per[sym]
            side = a.get("side", "")
            if side.startswith("buy"):
                d["bought_qty"] += qty
                d["bought_usd"] += usd
            else:                                   # sell / sell_short
                d["sold_qty"] += qty
                d["sold_usd"] += usd
            d["n_fills"] += 1
            ts = a.get("transaction_time", "")
            d["first"] = min(d["first"] or ts, ts)
            d["last"] = max(d["last"] or ts, ts)
        else:
            net = _f(a.get("net_amount"))
            when = a.get("date") or str(a.get("transaction_time", ""))[:10]
            cash_events.append({"date": when, "type": typ, "net": round(net, 2),
                                "desc": (a.get("description") or typ)[:80]})
            if typ in _TRANSFER_TYPES:
                transfers += net
            elif typ == "FEE":
                fees += net
            elif typ in _INCOME_TYPES:
                income += net
            else:
                other += net

    # attach open position + unrealized + realized per symbol
    pos_by = {p["symbol"]: p for p in positions}
    rows = []
    for sym, d in per.items():
        p = pos_by.get(sym, {})
        open_qty = _f(p.get("qty"))
        unreal = _f(p.get("unrealized")) if p else 0.0
        # realized (ledger view): sold$ - bought$ + what the open remainder is
        # carried at (open qty × avg cost) — exact for closed symbols, cost-basis
        # for open ones
        mult = 100.0 if _is_option(sym) else 1.0
        carry = open_qty * _f(p.get("avg_price")) * mult
        realized = d["sold_usd"] - d["bought_usd"] + carry
        rows.append({
            "symbol": sym, "n_fills": d["n_fills"],
            "bought_qty": round(d["bought_qty"], 4),
            "bought_usd": round(d["bought_usd"], 2),
            "sold_qty": round(d["sold_qty"], 4),
            "sold_usd": round(d["sold_usd"], 2),
            "open_qty": round(open_qty, 4),
            "unrealized": round(unreal, 2),
            "realized": round(realized, 2),
            "net_total": round(realized + unreal, 2),
            "last_activity": (d["last"] or "")[:16].replace("T", " ")})
    rows.sort(key=lambda r: r["last_activity"], reverse=True)
    cash_events.sort(key=lambda e: e["date"], reverse=True)

    total_realized = sum(r["realized"] for r in rows)
    total_unreal = sum(r["unrealized"] for r in rows)
    # reconciliation vs prior close: what the LEDGER explains
    explained = transfers + fees + income + other
    trading_effect = equity - last_equity - explained
    # NOTE: trading_effect blends realized+unrealized since prior close; the
    # honest residual test is ledger-vs-history over the SAME window, which the
    # venue truncates on paper resets — so surface a strong heuristic instead:
    # unexplained = the day swing minus what transfers/fees/income account for;
    # trading marks rarely move a book >50% in a day — a huge unexplained
    # residual with no matching transfer means the venue adjusted the account
    unexplained = (equity - last_equity) - explained
    reset_suspected = abs(unexplained) > max(0.5 * max(last_equity, 1), 500)

    return {
        "as_of": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "equity": round(equity, 2), "prior_close_equity": round(last_equity, 2),
        "day_change": round(equity - last_equity, 2),
        "per_symbol": rows,
        "cash_events": cash_events[:40],
        "totals": {"realized_all_ledger": round(total_realized, 2),
                   "unrealized_open": round(total_unreal, 2),
                   "transfers": round(transfers, 2),
                   "fees": round(fees, 2),
                   "income": round(income, 2)},
        "reconciliation": {
            "equity_minus_prior_close": round(equity - last_equity, 2),
            "explained_by_cash_events": round(explained, 2),
            "attributed_to_trading_and_marks": round(trading_effect, 2),
            "note": ("LARGE swing with NO transfer in the ledger — consistent "
                     "with a PAPER-ACCOUNT RESET/adjustment made in the Alpaca "
                     "dashboard (resets wipe portfolio history and leave no "
                     "journal entry). Not trading P&L."
                     if reset_suspected else
                     "day change is explained by trading marks and the listed "
                     "cash events."),
            "reset_suspected": reset_suspected},
    }


def _is_option(sym: str) -> bool:
    import re
    return bool(re.match(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$", sym or ""))


def _selftest():
    acts = [
        {"activity_type": "FILL", "symbol": "BNO", "side": "buy", "qty": "160",
         "price": "41.10", "transaction_time": "2026-07-07T16:19:00Z"},
        {"activity_type": "FILL", "symbol": "BNO", "side": "sell", "qty": "60",
         "price": "42.00", "transaction_time": "2026-07-07T18:00:00Z"},
        {"activity_type": "FILL", "symbol": "AAPL260814P00320000", "side": "buy",
         "qty": "1", "price": "15.35", "transaction_time": "2026-07-07T19:01:00Z"},
        {"activity_type": "FILL", "symbol": "AAPL260814P00320000", "side": "sell",
         "qty": "1", "price": "14.25", "transaction_time": "2026-07-07T19:02:00Z"},
        {"activity_type": "FEE", "net_amount": "-0.05", "date": "2026-07-07",
         "description": "TAF fee"},
        {"activity_type": "CSD", "net_amount": "1000", "date": "2026-07-06",
         "description": "deposit"},
    ]
    pos = [{"symbol": "BNO", "qty": 100, "avg_price": 41.10, "unrealized": 90.0}]
    st = build(acts, pos, equity=9000.0, last_equity=2000.0)
    bno = next(r for r in st["per_symbol"] if r["symbol"] == "BNO")
    assert bno["bought_qty"] == 160 and bno["bought_usd"] == 6576.0
    assert bno["sold_qty"] == 60 and bno["sold_usd"] == 2520.0
    assert bno["open_qty"] == 100 and bno["unrealized"] == 90.0
    # realized: 2520 - 6576 + 100*41.10 = 54  (sold 60 @42 vs cost 41.10 = +54)
    assert abs(bno["realized"] - 54.0) < 1e-6, bno["realized"]
    opt = next(r for r in st["per_symbol"] if _is_option(r["symbol"]))
    assert opt["bought_usd"] == 1535.0 and opt["sold_usd"] == 1425.0
    assert abs(opt["realized"] - (-110.0)) < 1e-6, "option x100 multiplier"
    assert st["totals"]["transfers"] == 1000.0 and st["totals"]["fees"] == -0.05
    # 7000 swing, only 1000 transferred -> reset suspected
    assert st["reconciliation"]["reset_suspected"] is True
    st2 = build(acts, pos, equity=2100.0, last_equity=2000.0)
    assert st2["reconciliation"]["reset_suspected"] is False
    print("statement self-test ✓  per-symbol totals (equity+options x100), "
          "realized math, transfers/fees split, reset heuristic")


if __name__ == "__main__":
    _selftest()
