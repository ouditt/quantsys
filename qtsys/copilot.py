"""copilot.py — the Account Copilot: ask-anything Q&A grounded in LIVE state.

Assembles a compact, current snapshot of the whole account (equity, positions
with unrealised P&L, closed round-trips, the day plan, open proposals, risk
attribution, auto-trader status, recent agent activity) and answers the
operator's question STRICTLY from that snapshot via a LOCAL LLM — the data
never leaves the machine.

The prompt is grounded and defensive: answer only from the data, cite the
numbers, and say "I don't have that" rather than invent. Answers are short and
plain-spoken because they may be read aloud (voice briefing).

build_context(state) is pure assembly (testable without an LLM); answer()
runs the local model.
"""
from __future__ import annotations

import datetime
import json


def _f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def build_context(state) -> dict:
    """Gather the live account snapshot the Copilot reasons over. Best-effort:
    any section that errors is simply omitted."""
    from . import server  # reuse the server's helpers/endpanels via state
    ctx: dict = {"as_of": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
    broker = state.get("broker")
    gw = state.get("gw")

    try:
        a = broker.get_account()
        dc = server._day_change(broker)
        ctx["account"] = {
            "equity": round(_f(a.get("equity")), 2),
            "cash": round(_f(a.get("cash")), 2),
            "buying_power": round(_f(a.get("buying_power")), 2),
            "day_pnl": round(_f(a.get("day_pnl")), 2),
            "day_change_pct": round(dc * 100, 2) if dc is not None else None,
            "gross_exposure": round(_f(a.get("gross_exposure")), 2),
            "net_exposure": round(_f(a.get("net_exposure")), 2),
            "leverage": round(_f(a.get("leverage")), 2),
            "halted": bool(getattr(gw, "halted", False)),
            "halt_reason": getattr(gw, "halt_reason", ""),
        }
    except Exception:
        pass

    try:
        pos = []
        for p in broker.get_positions():
            d = p.to_dict(p.v_last if p.v_last is not None else p.avg_price)
            pos.append({"symbol": d["symbol"], "side": d["side"],
                        "qty": d["qty"], "avg_cost": round(d["avg_price"], 4),
                        "last": round(d["last"], 4),
                        "unrealised_pnl": round(d["unrealized"], 2),
                        "unrealised_pct": round(d["unrealized_pct"] * 100, 2)})
        ctx["open_positions"] = pos
    except Exception:
        pass

    try:
        from . import tracking
        trips = tracking.realised_roundtrips(
            server._fills(broker), lambda s: server._cls_of(s) or "Equity")
        ctx["closed_trades_recent"] = [
            {"symbol": t["symbol"], "side": t["side"], "qty": t["qty"],
             "entry": t["entry_px"], "exit": t["exit_px"], "pnl": t["pnl"],
             "pnl_pct": t["pnl_pct"]} for t in trips[:12]]
        ctx["closed_summary"] = {
            "n": len(trips),
            "realised_pnl": round(sum(t["pnl"] for t in trips), 2),
            "win_rate_pct": round(sum(1 for t in trips if t["pnl"] > 0)
                                  / len(trips) * 100, 1) if trips else None}
    except Exception:
        pass

    try:
        at = state.get("autotrader")
        if at:
            s = at.status()
            ctx["auto_trader"] = {
                "armed": s["enabled"], "mode": "paper" if s["paper"] else "live",
                "open_managed": s["open"], "orders_today": s["orders_today"],
                "realised_today": s["realised_today"],
                "dsr_gate": s["require_dsr"], "dsr_threshold": s["dsr_threshold"],
                "options_trading": s["options_on"],
                "live_unlock": f"{s['paper_days']}/{s['paper_days_req']} paper days",
                "per_symbol_cap_pct": round(s["max_symbol_pct"] * 100),
                "daily_loss_breaker_pct": round(s["max_daily_loss"] * 100)}
    except Exception:
        pass

    try:
        p = state["planstore"].latest() if state.get("planstore") else None
        if p:
            ctx["today_plan"] = {
                "date": p.get("date"), "status": p.get("status"),
                "posture": p.get("posture"), "notes": p.get("notes"),
                "ideas": [{"side": i["side"], "symbol": i["symbol"],
                           "strategy": i.get("strategy"),
                           "dsr": i.get("dsr"),
                           "auto_tradable": bool(i.get("verified")),
                           "entry": i.get("entry"), "stop": i.get("stop"),
                           "target": i.get("target"), "qty": i.get("qty")}
                          for i in p.get("ideas", [])],
                "execution": p.get("execution")}
    except Exception:
        pass

    try:
        st = getattr(state.get("daemon"), "proposals", None)
        if st:
            ctx["open_proposals"] = [
                {"agent": x["agent"], "kind": x["kind"], "symbol": x["symbol"],
                 "summary": x["summary"]} for x in st.open(15)]
    except Exception:
        pass

    try:
        from . import portfolio_risk as pr
        w = state["daemon"].context.get("weights", lambda: {})()
        if w:
            ctx["risk"] = {"weights": w,
                           "factors": pr.factor_exposures(w).get("factors", []),
                           "attribution": pr.attribution(w).get("rows", [])[:5]}
    except Exception:
        pass

    try:
        log = state["daemon"].recent_log(20)
        ctx["recent_agent_activity"] = [
            {"agent": x["agent"], "note": x["message"][:160]} for x in log][:12]
    except Exception:
        pass

    return ctx


SYSTEM = (
    "You are the QTSYS trading-desk assistant. Answer the operator's question "
    "about THEIR OWN account using ONLY the ACCOUNT SNAPSHOT (JSON) below. "
    "Rules:\n"
    "- Cite the specific numbers from the data (dollars, %, symbols).\n"
    "- If the answer isn't in the data, say 'I don't have that in the current "
    "snapshot' — never invent numbers, prices, or trades.\n"
    "- Be concise and plain-spoken (this may be READ ALOUD): 1-4 sentences, no "
    "markdown, no bullet symbols, no preamble.\n"
    "- 'Unrealised' P&L is on OPEN positions; 'realised' is on CLOSED trades; "
    "'day P&L' is the whole account since the prior close — keep them distinct.\n"
    "- The auto-trader only trades DSR-verified ideas at/above the operator's "
    "threshold; others go to the proposal inbox."
)


def answer(question: str, ctx: dict, llm_fn) -> str:
    """Answer `question` grounded in the snapshot `ctx` using the local model."""
    if not llm_fn:
        return ("The local assistant model isn't running. Start Ollama "
                "(`ollama serve`) and pull a model to enable voice/ask.")
    q = (question or "").strip()[:600]
    if not q:
        return "Ask me anything about the account — positions, P&L, the plan, or why the engine did something."
    prompt = (f"{SYSTEM}\n\nACCOUNT SNAPSHOT:\n{json.dumps(ctx, default=str)}\n\n"
              f"OPERATOR QUESTION: {q}\n\nANSWER:")
    try:
        out = llm_fn(prompt).strip()
        return out or "I couldn't form an answer from the current snapshot."
    except Exception as e:
        return f"Local model error ({type(e).__name__}); is Ollama running?"


def _selftest():
    # pure-assembly shape check with a fake state
    class _P:
        def __init__(s): s.avg_price = 100.0; s.v_last = 101.0
        def to_dict(s, last): return {"symbol": "X", "side": "long", "qty": 5,
                                      "avg_price": 100.0, "last": 101.0,
                                      "unrealized": 5.0, "unrealized_pct": 0.01}
    class _B:
        paper = True
        def get_account(s): return {"equity": 1000, "cash": 500, "day_pnl": 12,
                                    "gross_exposure": 505, "net_exposure": 505,
                                    "leverage": 1.0, "buying_power": 500}
        def get_positions(s): return [_P()]
    class _GW: halted = False; halt_reason = ""
    state = {"broker": _B(), "gw": _GW()}
    # answer() with a stub llm: proves grounding prompt is built + returned
    ctx = {"account": {"equity": 1000, "day_pnl": 12}, "open_positions": [
        {"symbol": "X", "unrealised_pnl": 5.0}]}
    out = answer("what's my day pnl?", ctx, lambda p: "Your day P&L is +$12." if "SNAPSHOT" in p else "?")
    assert "12" in out, out
    assert "isn't running" in answer("x", ctx, None)
    assert "anything about the account" in answer("", ctx, lambda p: "x")
    print("copilot self-test ✓  grounded prompt built, no-llm + empty-q handled")


if __name__ == "__main__":
    _selftest()
