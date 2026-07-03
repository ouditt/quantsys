"""Broker & execution layer — one interface, many venues.

Everything trades through the same `Broker` contract so strategy code never
changes when the venue does. `PaperBroker` is a complete, stateful simulator
(market + limit orders, fills on ticks, positions, P&L, flatten-all) — it is
how the whole stack is tested at $0. The real adapters (Alpaca, IBKR, ccxt,
Oanda, Tradier) contain the actual client mappings behind guarded imports, so
they activate the moment the free SDK is installed and keys are supplied —
Alpaca and most crypto testnets offer PAPER endpoints, which keeps even the
live-API step at $0.

Every order passes the ExecutionGateway's pre-trade checks. An order that
fails a check is never sent. The kill switch halts and flattens.
"""
from __future__ import annotations

import itertools
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ------------------------------------------------------------------ order model
@dataclass
class Order:
    symbol: str
    side: str                    # "buy" | "sell"
    qty: float
    type: str = "market"         # "market" | "limit" | "stop"
    limit_price: float | None = None
    stop_price: float | None = None
    id: str = ""
    status: str = "pending"      # pending -> working -> filled | cancelled | rejected
    reason: str = ""
    submitted_at: float = 0.0
    filled_at: float | None = None
    fill_price: float | None = None

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in
                ("id", "symbol", "side", "qty", "type", "limit_price", "stop_price",
                 "status", "reason", "submitted_at", "filled_at", "fill_price")}


@dataclass
class Position:
    symbol: str
    qty: float                   # signed: + long, - short
    avg_price: float

    def to_dict(self, last: float) -> dict:
        mv = self.qty * last
        upnl = (last - self.avg_price) * self.qty
        base = abs(self.qty * self.avg_price)
        return {"symbol": self.symbol, "qty": self.qty, "avg_price": self.avg_price,
                "last": last, "mkt_value": mv, "unrealized": upnl,
                "unrealized_pct": (upnl / base) if base else 0.0,
                "side": "long" if self.qty >= 0 else "short"}


# ---------------------------------------------------------------- broker contract
class Broker(ABC):
    @abstractmethod
    def get_account(self) -> dict: ...
    @abstractmethod
    def get_positions(self) -> list[Position]: ...
    @abstractmethod
    def get_orders(self, open_only: bool = True) -> list[Order]: ...
    @abstractmethod
    def submit(self, order: Order) -> Order: ...
    @abstractmethod
    def cancel(self, order_id: str) -> bool: ...
    @abstractmethod
    def get_quote(self, symbol: str) -> float: ...
    def flatten_all(self) -> None:
        for p in self.get_positions():
            if p.qty:
                self.submit(Order(p.symbol, "sell" if p.qty > 0 else "buy",
                                  abs(p.qty), "market"))


# ------------------------------------------------------------------- paper broker
class PaperBroker(Broker):
    """Full offline venue: immediate market fills at quote +/- slippage, resting
    limit orders matched on process_tick(), realized/unrealized P&L, day P&L."""

    def __init__(self, cash: float = 100_000.0, fee_per_side: float = 0.0003,
                 slippage_bps: float = 1.0):
        self.cash = cash
        self.start_equity = cash
        self.day_open_equity = cash
        self.fee_per_side = fee_per_side
        self.slip = slippage_bps / 1e4
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self.fills: list[dict] = []
        self.quotes: dict[str, float] = {}
        self._seq = itertools.count(1)

    # --- market data in ----------------------------------------------------
    def process_tick(self, symbol: str, price: float) -> None:
        self.quotes[symbol] = price
        for o in self.orders:
            if o.symbol != symbol or o.status != "working":
                continue
            hit = (o.type == "limit" and
                   ((o.side == "buy" and price <= o.limit_price) or
                    (o.side == "sell" and price >= o.limit_price))) or \
                  (o.type == "stop" and
                   ((o.side == "buy" and price >= o.stop_price) or
                    (o.side == "sell" and price <= o.stop_price)))
            if hit:
                self._fill(o, o.limit_price if o.type == "limit" else price)

    # --- trading -----------------------------------------------------------
    def submit(self, order: Order) -> Order:
        order.id = f"P{next(self._seq):05d}"
        order.submitted_at = time.time()
        px = self.quotes.get(order.symbol)
        if px is None:
            order.status, order.reason = "rejected", "no market data for symbol"
        elif order.type == "market":
            fill = px * (1 + self.slip) if order.side == "buy" else px * (1 - self.slip)
            self._fill(order, fill)
        else:
            order.status = "working"
        self.orders.append(order)
        return order

    def _fill(self, o: Order, price: float) -> None:
        signed = o.qty if o.side == "buy" else -o.qty
        pos = self.positions.get(o.symbol, Position(o.symbol, 0.0, 0.0))
        new_qty = pos.qty + signed
        if pos.qty * signed >= 0:                                   # add/open
            tot = abs(pos.qty) + abs(signed)
            pos.avg_price = (abs(pos.qty) * pos.avg_price + abs(signed) * price) / tot
        elif abs(signed) > abs(pos.qty):                            # flip through 0
            pos.avg_price = price
        pos.qty = new_qty
        self.positions[o.symbol] = pos
        fee = abs(signed) * price * self.fee_per_side
        self.cash -= signed * price + fee
        o.status, o.fill_price, o.filled_at = "filled", price, time.time()
        self.fills.append({"ts": o.filled_at, "symbol": o.symbol, "side": o.side,
                           "qty": o.qty, "price": price, "fee": fee, "order_id": o.id})
        if abs(pos.qty) < 1e-12:
            del self.positions[o.symbol]

    def cancel(self, order_id: str) -> bool:
        for o in self.orders:
            if o.id == order_id and o.status == "working":
                o.status = "cancelled"
                return True
        return False

    # --- state out ----------------------------------------------------------
    def get_positions(self) -> list[Position]:
        return list(self.positions.values())

    def get_orders(self, open_only: bool = True) -> list[Order]:
        return [o for o in self.orders
                if (o.status in ("working", "pending")) or not open_only]

    def get_quote(self, symbol: str) -> float:
        return self.quotes.get(symbol, float("nan"))

    def equity(self) -> float:
        return self.cash + sum(p.qty * self.quotes.get(p.symbol, p.avg_price)
                               for p in self.positions.values())

    def get_account(self) -> dict:
        eq = self.equity()
        gross = sum(abs(p.qty) * self.quotes.get(p.symbol, p.avg_price)
                    for p in self.positions.values())
        net = sum(p.qty * self.quotes.get(p.symbol, p.avg_price)
                  for p in self.positions.values())
        return {"equity": eq, "cash": self.cash,
                "buying_power": max(eq * 2 - gross, 0.0),
                "day_pnl": eq - self.day_open_equity,
                "total_pnl": eq - self.start_equity,
                "gross_exposure": gross, "net_exposure": net,
                "leverage": gross / eq if eq > 0 else 0.0,
                "drawdown": min(eq / max(self.start_equity, eq) - 1, 0.0)}


# ------------------------------------------------------------ pre-trade gateway
@dataclass
class RiskLimits:
    max_order_notional: float = 25_000.0
    max_position_notional: float = 60_000.0
    max_gross_leverage: float = 2.0
    allow_short: bool = True


class ExecutionGateway:
    """Every order passes here. Fails a check -> never sent. halt() = kill switch."""

    def __init__(self, broker: Broker, limits: RiskLimits, live: bool = False):
        self.broker, self.limits, self.live = broker, limits, live
        self.halted = False
        self.halt_reason = ""

    def pretrade_check(self, order: Order) -> str | None:
        if self.halted:
            return f"halted: {self.halt_reason}"
        px = order.limit_price or self.broker.get_quote(order.symbol)
        if not px or px != px:
            return "no market data for symbol"
        notional = order.qty * px
        if notional > self.limits.max_order_notional:
            return (f"order notional {notional:,.0f} exceeds per-order cap "
                    f"{self.limits.max_order_notional:,.0f}")
        pos = {p.symbol: p for p in self.broker.get_positions()}.get(order.symbol)
        cur = (pos.qty if pos else 0.0) * px
        new = cur + (notional if order.side == "buy" else -notional)
        if abs(new) > self.limits.max_position_notional:
            return "would exceed per-symbol position cap"
        if not self.limits.allow_short and new < 0:
            return "short selling not permitted"
        acct = self.broker.get_account()
        if (acct["gross_exposure"] + notional) / max(acct["equity"], 1) \
                > self.limits.max_gross_leverage:
            return "would exceed gross leverage cap"
        return None

    def submit(self, order: Order) -> Order:
        why = self.pretrade_check(order)
        if why:
            order.status, order.reason = "rejected", why
            return order
        return self.broker.submit(order)

    def halt(self, reason: str) -> None:
        self.halted, self.halt_reason = True, reason
        for o in self.broker.get_orders(open_only=True):
            self.broker.cancel(o.id)
        self.broker.flatten_all()

    def resume(self) -> None:
        self.halted, self.halt_reason = False, ""


# ------------------------------------------------------- real venue adapters
# Each adapter maps the same contract onto a real (free-to-install) SDK. Imports
# are guarded: the class explains exactly what to `pip install` when used.

class AlpacaBroker(Broker):
    """US equities/ETFs/crypto. FREE paper endpoint. `pip install alpaca-py`."""

    def __init__(self, api_key: str, secret: str, paper: bool = True):
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import (LimitOrderRequest,
                                             MarketOrderRequest)
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.data.historical import (CryptoHistoricalDataClient,
                                            StockHistoricalDataClient)
        from alpaca.data.requests import (CryptoLatestTradeRequest,
                                          StockLatestTradeRequest)
        self._req = {"mkt": MarketOrderRequest, "lim": LimitOrderRequest,
                     "side": OrderSide, "tif": TimeInForce,
                     "last": StockLatestTradeRequest,
                     "last_crypto": CryptoLatestTradeRequest}
        self.c = TradingClient(api_key, secret, paper=paper)
        self.d = StockHistoricalDataClient(api_key, secret)
        self.dc = CryptoHistoricalDataClient(api_key, secret)

    def get_account(self) -> dict:
        a = self.c.get_account()
        return {"equity": float(a.equity), "cash": float(a.cash),
                "buying_power": float(a.buying_power),
                "day_pnl": float(a.equity) - float(a.last_equity),
                "gross_exposure": float(a.long_market_value) - float(a.short_market_value),
                "net_exposure": float(a.long_market_value) + float(a.short_market_value),
                "leverage": float(a.multiplier), "drawdown": 0.0,
                "total_pnl": float(a.equity) - float(a.last_equity)}

    def get_positions(self) -> list[Position]:
        return [Position(p.symbol, float(p.qty) * (1 if p.side.value == "long" else -1),
                         float(p.avg_entry_price)) for p in self.c.get_all_positions()]

    def get_orders(self, open_only: bool = True) -> list[Order]:
        out = []
        for o in self.c.get_orders():
            out.append(Order(o.symbol, o.side.value, float(o.qty or 0),
                             o.order_type.value,
                             float(o.limit_price) if o.limit_price else None,
                             float(o.stop_price) if o.stop_price else None,
                             id=str(o.id), status=o.status.value))
        return [o for o in out if not open_only or o.status in
                ("new", "accepted", "partially_filled", "pending_new")]

    def submit(self, order: Order) -> Order:
        R, side = self._req, self._req["side"]
        s = side.BUY if order.side == "buy" else side.SELL
        req = (R["mkt"](symbol=order.symbol, qty=order.qty, side=s,
                        time_in_force=R["tif"].DAY) if order.type == "market"
               else R["lim"](symbol=order.symbol, qty=order.qty, side=s,
                             time_in_force=R["tif"].DAY,
                             limit_price=order.limit_price))
        r = self.c.submit_order(req)
        order.id, order.status = str(r.id), r.status.value
        return order

    def cancel(self, order_id: str) -> bool:
        self.c.cancel_order_by_id(order_id); return True

    def get_quote(self, symbol: str) -> float:
        if "/" in symbol:   # crypto pair, e.g. "BTC/USD"
            t = self.dc.get_crypto_latest_trade(
                self._req["last_crypto"](symbol_or_symbols=symbol))
        else:
            t = self.d.get_stock_latest_trade(
                self._req["last"](symbol_or_symbols=symbol))
        return float(t[symbol].price)


class IBKRBroker(Broker):
    """Multi-asset (stocks/options/futures/FX/bonds) via TWS or IB Gateway.
    `pip install ib_insync` and run TWS/Gateway with API enabled (paper: port 7497)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        from ib_insync import IB, Stock, MarketOrder, LimitOrder
        self._ib_mod = {"Stock": Stock, "MarketOrder": MarketOrder,
                        "LimitOrder": LimitOrder}
        self.ib = IB(); self.ib.connect(host, port, clientId=client_id)

    def _contract(self, symbol: str):
        return self._ib_mod["Stock"](symbol, "SMART", "USD")

    def get_account(self) -> dict:
        v = {r.tag: float(r.value) for r in self.ib.accountSummary()
             if r.tag in ("NetLiquidation", "TotalCashValue", "BuyingPower",
                          "GrossPositionValue")}
        return {"equity": v.get("NetLiquidation", 0.0),
                "cash": v.get("TotalCashValue", 0.0),
                "buying_power": v.get("BuyingPower", 0.0),
                "gross_exposure": v.get("GrossPositionValue", 0.0),
                "net_exposure": v.get("GrossPositionValue", 0.0),
                "day_pnl": 0.0, "total_pnl": 0.0,
                "leverage": (v.get("GrossPositionValue", 0.0)
                             / max(v.get("NetLiquidation", 1.0), 1.0)),
                "drawdown": 0.0}

    def get_positions(self) -> list[Position]:
        return [Position(p.contract.symbol, p.position, p.avgCost)
                for p in self.ib.positions()]

    def get_orders(self, open_only: bool = True) -> list[Order]:
        out = []
        for t in self.ib.openTrades():
            o = t.order
            out.append(Order(t.contract.symbol, o.action.lower(), o.totalQuantity,
                             "limit" if o.orderType == "LMT" else "market",
                             getattr(o, "lmtPrice", None), id=str(o.orderId),
                             status=t.orderStatus.status.lower()))
        return out

    def submit(self, order: Order) -> Order:
        M = self._ib_mod
        ib_order = (M["MarketOrder"](order.side.upper(), order.qty)
                    if order.type == "market"
                    else M["LimitOrder"](order.side.upper(), order.qty,
                                         order.limit_price))
        tr = self.ib.placeOrder(self._contract(order.symbol), ib_order)
        order.id, order.status = str(tr.order.orderId), "working"
        return order

    def cancel(self, order_id: str) -> bool:
        for t in self.ib.openTrades():
            if str(t.order.orderId) == order_id:
                self.ib.cancelOrder(t.order); return True
        return False

    def get_quote(self, symbol: str) -> float:
        [tk] = self.ib.reqTickers(self._contract(symbol))
        return float(tk.marketPrice())


class CCXTBroker(Broker):
    """Crypto on 100+ exchanges. `pip install ccxt`. Most venues offer testnets
    (set sandbox mode) so this too can be exercised at $0."""

    def __init__(self, exchange: str, api_key: str = "", secret: str = "",
                 sandbox: bool = True):
        import ccxt
        self.x = getattr(ccxt, exchange)({"apiKey": api_key, "secret": secret})
        if sandbox and self.x.has.get("sandbox"):
            self.x.set_sandbox_mode(True)

    def get_account(self) -> dict:
        b = self.x.fetch_balance()
        eq = float(b.get("total", {}).get("USDT", 0.0))
        return {"equity": eq, "cash": float(b.get("free", {}).get("USDT", 0.0)),
                "buying_power": eq, "day_pnl": 0.0, "total_pnl": 0.0,
                "gross_exposure": 0.0, "net_exposure": 0.0,
                "leverage": 1.0, "drawdown": 0.0}

    def get_positions(self) -> list[Position]:
        if not self.x.has.get("fetchPositions"):
            return []
        return [Position(p["symbol"], float(p["contracts"] or 0)
                         * (1 if p["side"] == "long" else -1),
                         float(p["entryPrice"] or 0))
                for p in self.x.fetch_positions() if p.get("contracts")]

    def get_orders(self, open_only: bool = True) -> list[Order]:
        return [Order(o["symbol"], o["side"], float(o["amount"]),
                      o["type"], o.get("price"), id=str(o["id"]),
                      status=o["status"]) for o in self.x.fetch_open_orders()]

    def submit(self, order: Order) -> Order:
        r = self.x.create_order(order.symbol, order.type, order.side,
                                order.qty, order.limit_price)
        order.id, order.status = str(r["id"]), r.get("status", "working")
        return order

    def cancel(self, order_id: str) -> bool:
        self.x.cancel_order(order_id); return True

    def get_quote(self, symbol: str) -> float:
        return float(self.x.fetch_ticker(symbol)["last"])


class OandaBroker(Broker):
    """FX. `pip install oandapyV20`; free practice accounts exist."""

    def __init__(self, account_id: str, token: str, practice: bool = True):
        import oandapyV20
        import oandapyV20.endpoints.accounts as accounts
        import oandapyV20.endpoints.orders as orders_ep
        import oandapyV20.endpoints.positions as positions_ep
        import oandapyV20.endpoints.pricing as pricing
        env = "practice" if practice else "live"
        self.api = oandapyV20.API(access_token=token, environment=env)
        self.acc, self._ep = account_id, {"accounts": accounts, "orders": orders_ep,
                                          "positions": positions_ep,
                                          "pricing": pricing}

    def get_account(self) -> dict:
        r = self._ep["accounts"].AccountSummary(self.acc)
        a = self.api.request(r)["account"]
        return {"equity": float(a["NAV"]), "cash": float(a["balance"]),
                "buying_power": float(a["marginAvailable"]),
                "day_pnl": float(a.get("unrealizedPL", 0)),
                "total_pnl": float(a.get("pl", 0)),
                "gross_exposure": 0.0, "net_exposure": 0.0,
                "leverage": 0.0, "drawdown": 0.0}

    def get_positions(self) -> list[Position]:
        r = self._ep["positions"].OpenPositions(self.acc)
        out = []
        for p in self.api.request(r)["positions"]:
            for side, sign in (("long", 1), ("short", -1)):
                units = float(p[side]["units"])
                if units:
                    out.append(Position(p["instrument"], units,
                                        float(p[side]["averagePrice"])))
        return out

    def get_orders(self, open_only: bool = True) -> list[Order]:
        r = self._ep["orders"].OrdersPending(self.acc)
        return [Order(o["instrument"], "buy" if float(o["units"]) > 0 else "sell",
                      abs(float(o["units"])), o["type"].lower(),
                      float(o.get("price", 0)) or None, id=o["id"],
                      status="working") for o in self.api.request(r)["orders"]]

    def submit(self, order: Order) -> Order:
        units = order.qty if order.side == "buy" else -order.qty
        data = {"order": {"instrument": order.symbol, "units": str(units),
                          "type": order.type.upper(),
                          **({"price": str(order.limit_price)}
                             if order.type == "limit" else {})}}
        r = self._ep["orders"].OrderCreate(self.acc, data=data)
        resp = self.api.request(r)
        order.id = resp.get("orderCreateTransaction", {}).get("id", "")
        order.status = "working" if order.type == "limit" else "filled"
        return order

    def cancel(self, order_id: str) -> bool:
        r = self._ep["orders"].OrderCancel(self.acc, orderID=order_id)
        self.api.request(r); return True

    def get_quote(self, symbol: str) -> float:
        r = self._ep["pricing"].PricingInfo(self.acc,
                                            params={"instruments": symbol})
        p = self.api.request(r)["prices"][0]
        return (float(p["bids"][0]["price"]) + float(p["asks"][0]["price"])) / 2


class TradierBroker(Broker):
    """US equities & OPTIONS. `pip install requests`; free paper sandbox exists."""

    BASE = {"paper": "https://sandbox.tradier.com/v1",
            "live": "https://api.tradier.com/v1"}

    def __init__(self, account_id: str, token: str, paper: bool = True):
        import requests
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}",
                               "Accept": "application/json"})
        self.base = self.BASE["paper" if paper else "live"]
        self.acc = account_id

    def get_account(self) -> dict:
        b = self.s.get(f"{self.base}/accounts/{self.acc}/balances").json()["balances"]
        return {"equity": float(b["total_equity"]), "cash": float(b["total_cash"]),
                "buying_power": float(b.get("cash", {}).get("cash_available",
                                                            b["total_cash"])),
                "day_pnl": 0.0, "total_pnl": 0.0, "gross_exposure": 0.0,
                "net_exposure": 0.0, "leverage": 1.0, "drawdown": 0.0}

    def get_positions(self) -> list[Position]:
        r = self.s.get(f"{self.base}/accounts/{self.acc}/positions").json()
        pos = (r.get("positions") or {}).get("position") or []
        pos = pos if isinstance(pos, list) else [pos]
        return [Position(p["symbol"], float(p["quantity"]),
                         float(p["cost_basis"]) / max(float(p["quantity"]), 1))
                for p in pos]

    def get_orders(self, open_only: bool = True) -> list[Order]:
        r = self.s.get(f"{self.base}/accounts/{self.acc}/orders").json()
        os_ = (r.get("orders") or {}).get("order") or []
        os_ = os_ if isinstance(os_, list) else [os_]
        out = [Order(o["symbol"], o["side"], float(o["quantity"]), o["type"],
                     o.get("price"), id=str(o["id"]), status=o["status"])
               for o in os_]
        return [o for o in out if not open_only or o.status in ("open", "pending")]

    def submit(self, order: Order) -> Order:
        data = {"class": "equity", "symbol": order.symbol, "side": order.side,
                "quantity": str(int(order.qty)), "type": order.type,
                "duration": "day"}
        if order.type == "limit":
            data["price"] = str(order.limit_price)
        r = self.s.post(f"{self.base}/accounts/{self.acc}/orders", data=data).json()
        order.id = str(r.get("order", {}).get("id", ""))
        order.status = r.get("order", {}).get("status", "working")
        return order

    def cancel(self, order_id: str) -> bool:
        self.s.delete(f"{self.base}/accounts/{self.acc}/orders/{order_id}")
        return True

    def get_quote(self, symbol: str) -> float:
        r = self.s.get(f"{self.base}/markets/quotes",
                       params={"symbols": symbol}).json()
        return float(r["quotes"]["quote"]["last"])


def make_broker(venue: str, **kw) -> Broker:
    """Factory: paper | alpaca | ibkr | ccxt | oanda | tradier."""
    table = {"paper": PaperBroker, "alpaca": AlpacaBroker, "ibkr": IBKRBroker,
             "ccxt": CCXTBroker, "oanda": OandaBroker, "tradier": TradierBroker}
    if venue not in table:
        raise ValueError(f"unknown venue '{venue}'; options: {sorted(table)}")
    return table[venue](**kw)
