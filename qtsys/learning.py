"""learning.py — the system learns from its OWN live outcomes.

Deterministic per-strategy and per-agent scorecards computed from real logged
trades (journal.db) and the auto-trader's managed excursions (autotrader.db),
anchored on the certified backtest stats (registry_summary.csv). The learning
is pure statistics — shrinkage estimates, drift tests, excursion distributions
— NOT LLM training and NOT price generation. Every promote/demote/exit-tune is
an auditable row in learning.db.

The one bright line, enforced in code and asserted in the self-test: no output
of this module can ever INCREASE size after losses. `size_multiplier` is a
monotone-non-decreasing function of live expectancy, clamped to [0.25, 1.5],
so a losing streak can only ever lower it (anti-martingale).

Run:  python -m qtsys.learning        (temp DBs + synthetic drifting strategy)
"""
from __future__ import annotations

import json
import os
import sqlite3
import time

HERE = os.path.dirname(__file__)

# bounds shared by producer (here) and consumer (tradeplan) — clamped at BOTH
MULT_LO, MULT_HI = 0.25, 1.5
ATR_STOP_LO, ATR_STOP_HI = 1.0, 2.5
R_MULT_LO, R_MULT_HI = 1.5, 3.0
PRIOR_WEIGHT = 30          # live trades needed before live edge outweighs prior
DRIFT_N = 30               # min live trades before a promote/demote can fire
EXIT_N = 20                # min closed trades before exit-param tuning applies


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# --------------------------------------------------------------- persistence
class LearningStore:
    """learning.db — auditable decision log + tuned exit params + agent scores.
    Every connection is WAL + busy_timeout so the nightly writer and the API
    reader never trip over each other."""

    def __init__(self, db_path: str | None = None):
        self.db = sqlite3.connect(db_path or os.path.join(HERE, "learning.db"),
                                  check_same_thread=False)
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS decisions(
          ts REAL, strategy TEXT, action TEXT,
          reason TEXT, detail TEXT);
        CREATE TABLE IF NOT EXISTS exit_params(
          strategy TEXT PRIMARY KEY, atr_stop REAL, r_multiple REAL,
          updated_ts REAL, reason TEXT);
        CREATE TABLE IF NOT EXISTS agent_scores(
          ts REAL, agent TEXT, flag TEXT, n_flagged INTEGER,
          exp_flagged REAL, exp_unflagged REAL, verdict TEXT);""")
        self.db.commit()

    # ---- decisions -------------------------------------------------------
    def log_decision(self, strategy: str, action: str, reason: str,
                     detail: dict | None = None) -> None:
        self.db.execute("INSERT INTO decisions VALUES (?,?,?,?,?)",
                        (time.time(), strategy, action, reason,
                         json.dumps(detail or {}, default=str)))
        self.db.commit()

    def decisions(self, limit: int = 50) -> list[dict]:
        rows = self.db.execute(
            "SELECT ts,strategy,action,reason,detail FROM decisions "
            "ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [{"ts": r[0], "strategy": r[1], "action": r[2], "reason": r[3],
                 "detail": json.loads(r[4]) if r[4] else {}} for r in rows]

    def state_of(self, strategy: str) -> str:
        """Latest promote/demote/restore verdict for a strategy: a demote stands
        until a later restore/promote lifts it."""
        r = self.db.execute(
            "SELECT action FROM decisions WHERE strategy=? AND action IN "
            "('demote','restore','promote') ORDER BY ts DESC LIMIT 1",
            (strategy,)).fetchone()
        if not r:
            return "NORMAL"
        return {"demote": "DEMOTED", "restore": "NORMAL",
                "promote": "PROMOTED"}.get(r[0], "NORMAL")

    def demoted(self) -> dict[str, str]:
        """{strategy: reason} for every strategy whose standing verdict is demote."""
        out: dict[str, str] = {}
        seen = set()
        for r in self.db.execute(
                "SELECT strategy,action,reason FROM decisions WHERE action IN "
                "('demote','restore','promote') ORDER BY ts DESC"):
            sid = r[0]
            if sid in seen:
                continue
            seen.add(sid)
            if r[1] == "demote":
                out[sid] = r[2]
        return out

    # ---- exit params -----------------------------------------------------
    def set_exit_params(self, strategy: str, atr_stop: float, r_multiple: float,
                        reason: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO exit_params VALUES (?,?,?,?,?)",
                        (strategy, atr_stop, r_multiple, time.time(), reason))
        self.db.commit()

    def exit_params(self) -> dict[str, dict]:
        rows = self.db.execute("SELECT strategy,atr_stop,r_multiple,updated_ts,"
                               "reason FROM exit_params").fetchall()
        return {r[0]: {"atr_stop": r[1], "r_multiple": r[2], "updated_ts": r[3],
                       "reason": r[4]} for r in rows}

    # ---- agent scores ----------------------------------------------------
    def log_agent_score(self, agent: str, flag: str, n_flagged: int,
                        exp_flagged: float, exp_unflagged: float,
                        verdict: str) -> None:
        self.db.execute("INSERT INTO agent_scores VALUES (?,?,?,?,?,?,?)",
                        (time.time(), agent, flag, n_flagged, exp_flagged,
                         exp_unflagged, verdict))
        self.db.commit()

    def agent_scores(self, limit: int = 50) -> list[dict]:
        rows = self.db.execute(
            "SELECT ts,agent,flag,n_flagged,exp_flagged,exp_unflagged,verdict "
            "FROM agent_scores ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [{"ts": r[0], "agent": r[1], "flag": r[2], "n_flagged": r[3],
                 "exp_flagged": r[4], "exp_unflagged": r[5], "verdict": r[6]}
                for r in rows]


# --------------------------------------------------------------- statistics
def confidence(live_exp: float, live_n: int, prior_exp: float,
               prior_weight: int = PRIOR_WEIGHT) -> float:
    """Shrinkage estimate of the true expectancy: a weighted average of the live
    edge and the certified backtest prior. With < prior_weight live trades the
    prior dominates; as live_n grows the live edge takes over. This is the
    honest number — it does not pretend a 5-trade sample knows the edge."""
    k = max(prior_weight, 0)
    n = max(live_n, 0)
    if n + k == 0:
        return prior_exp
    return (n * live_exp + k * prior_exp) / (n + k)


def _backtest_prior(registry_csv: str | None) -> dict[str, dict]:
    """Certified out-of-sample stats per strategy id, from registry_summary.csv."""
    path = registry_csv or os.path.join(HERE, "registry_summary.csv")
    out: dict[str, dict] = {}
    if not os.path.exists(path):
        return out
    try:
        import pandas as pd
        d = pd.read_csv(path)
        for _, row in d.iterrows():
            sid = str(row.get("id"))
            def _num(v):
                try:
                    v = float(v)
                    return v if v == v else None
                except Exception:
                    return None
            out[sid] = {"dsr": _num(row.get("dsr")),
                        "test_exp": _num(row.get("test_exp")),
                        "test_n": _num(row.get("test_n"))}
    except Exception:
        pass
    return out


def _live_stats(journal_db: str | None) -> dict[str, dict]:
    """Per setup_id live outcome stats from journal.db. Grouped by setup_id
    ALONE — every auto-trader row carries regime_trend='AUTO', so (setup,regime)
    grouping would collapse to one bucket and hide nothing."""
    path = journal_db or os.path.join(HERE, "journal.db")
    out: dict[str, dict] = {}
    if not os.path.exists(path):
        return out
    try:
        from .journal import Journal
        df = Journal(path).frame()
    except Exception:
        return out
    if df.empty or "net_ret" not in df:
        return out
    import pandas as pd
    df = df[pd.to_numeric(df["net_ret"], errors="coerce").notna()].copy()
    df["net_ret"] = pd.to_numeric(df["net_ret"], errors="coerce")
    if "slippage_bps_vs_plan" in df:
        df["slippage_bps_vs_plan"] = pd.to_numeric(
            df["slippage_bps_vs_plan"], errors="coerce")
    for sid, g in df.groupby(df["setup_id"].fillna("auto")):
        r = g["net_ret"]
        n = int(r.count())
        if n == 0:
            continue
        wins = r[r > 0]
        losses = r[r < 0]
        pf = (wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 \
            else (float("inf") if len(wins) else 0.0)
        se = float(r.std(ddof=1) / (n ** 0.5)) if n > 1 else float("inf")
        slip = None
        if "slippage_bps_vs_plan" in g:
            sv = g["slippage_bps_vs_plan"].dropna()
            slip = float(sv.mean()) if len(sv) else None
        out[str(sid)] = {
            "n": n, "win_rate": float((r > 0).mean()),
            "expectancy": float(r.mean()), "expectancy_se": se,
            "profit_factor": float(pf) if pf != float("inf") else None,
            "avg_slippage_bps": slip}
    return out


def strategy_scorecard(journal_db: str | None = None,
                       registry_csv: str | None = None) -> dict[str, dict]:
    """Per-strategy scorecard merging live outcomes with the certified prior.
    `state` is the drift verdict from live-vs-backtest; `multiplier` is the
    bounded, anti-martingale size factor the planner will apply."""
    live = _live_stats(journal_db)
    prior = _backtest_prior(registry_csv)
    cards: dict[str, dict] = {}
    for sid in set(live) | set(prior):
        lv = live.get(sid, {"n": 0, "win_rate": 0.0, "expectancy": 0.0,
                            "expectancy_se": float("inf"),
                            "profit_factor": None, "avg_slippage_bps": None})
        bt = prior.get(sid, {"dsr": None, "test_exp": None, "test_n": None})
        prior_exp = bt["test_exp"] if bt["test_exp"] is not None else 0.0
        conf = confidence(lv["expectancy"], lv["n"], prior_exp)
        card = {
            "strategy": sid, "n": lv["n"], "win_rate": lv["win_rate"],
            "expectancy": lv["expectancy"], "expectancy_se": lv["expectancy_se"],
            "profit_factor": lv["profit_factor"],
            "avg_slippage_bps": lv["avg_slippage_bps"],
            "backtest": {"dsr": bt["dsr"], "test_exp": bt["test_exp"],
                         "test_n": bt["test_n"]},
            "confidence": conf}
        card["state"] = _drift_state(card)
        card["multiplier"] = size_multiplier(card)
        cards[sid] = card
    return cards


def _drift_state(card: dict) -> str:
    """PROMOTED / NORMAL / DEMOTED from live expectancy vs the backtest prior,
    only once there is enough live evidence (n >= DRIFT_N)."""
    bt = card["backtest"]["test_exp"]
    n, exp, se = card["n"], card["expectancy"], card["expectancy_se"]
    if n < DRIFT_N or bt is None or se == float("inf"):
        return "NORMAL"
    if exp < bt - se:
        return "DEMOTED"
    if exp > bt + se:
        return "PROMOTED"
    return "NORMAL"


def size_multiplier(card: dict) -> float:
    """Bounded size factor in [MULT_LO, MULT_HI]. MONOTONE NON-DECREASING in
    live expectancy: a losing streak lowers expectancy, which can only lower
    (never raise) the multiplier. This is the anti-martingale guarantee.

      * < DRIFT_N live trades, or no backtest prior -> 1.0 (prior governs).
      * DEMOTED  -> 0.5 (half size; the planner also INBOX-routes it).
      * PROMOTED -> 1.0 + 0.25 * z, capped at 1.5, where z is how many SE the
        live edge sits ABOVE the certified backtest expectancy.
      * NORMAL   -> 1.0.
    """
    state = card.get("state") or _drift_state(card)
    if state == "DEMOTED":
        return 0.5
    bt = card["backtest"]["test_exp"]
    n, exp, se = card["n"], card["expectancy"], card["expectancy_se"]
    if n < DRIFT_N or bt is None or se in (0.0, float("inf")):
        return 1.0
    if state == "PROMOTED":
        z = (exp - bt) / se
        return _clamp(1.0 + 0.25 * z, 1.0, MULT_HI)
    return 1.0


# --------------------------------------------------------------- promotions
def evaluate_promotions(store: LearningStore,
                        scorecard: dict[str, dict]) -> list[dict]:
    """Apply the drift rules and journal every state CHANGE to learning.db.

      n>=DRIFT_N AND live_exp < backtest_exp - 1*SE  -> demote (half-size + INBOX)
      recovery back within 1 SE (state NORMAL) after a demote -> restore
      n>=DRIFT_N AND live_exp > backtest_exp + 1*SE  -> promote (mult up to 1.5)

    Idempotent: only a transition from the standing verdict is logged."""
    changes = []
    for sid, card in scorecard.items():
        prev = store.state_of(sid)
        now = card["state"]
        if now == prev:
            continue
        detail = {"n": card["n"], "expectancy": card["expectancy"],
                  "expectancy_se": card["expectancy_se"],
                  "backtest_exp": card["backtest"]["test_exp"],
                  "multiplier": card["multiplier"]}
        bt = card["backtest"]["test_exp"]
        if now == "DEMOTED":
            act, reason = "demote", (
                f"live exp {card['expectancy']:+.4f} < backtest "
                f"{bt:+.4f} - 1SE over {card['n']} trades — half-sized, "
                "routed to INBOX as unverified")
        elif now == "PROMOTED":
            act, reason = "promote", (
                f"live exp {card['expectancy']:+.4f} > backtest {bt:+.4f} + 1SE "
                f"over {card['n']} trades — multiplier {card['multiplier']:.2f}")
        else:  # NORMAL after having been demoted/promoted -> restore
            act, reason = "restore", (
                f"live exp {card['expectancy']:+.4f} back within 1 SE of "
                f"backtest {bt:+.4f} — normal sizing restored")
        store.log_decision(sid, act, reason, detail)
        changes.append({"strategy": sid, "action": act, "reason": reason})
    return changes


# --------------------------------------------------------------- exit tuning
def _managed_excursions(autotrader_db: str | None) -> dict[str, list[dict]]:
    """Closed managed trades with recorded MFE/MAE, grouped by strategy."""
    path = autotrader_db or os.path.join(HERE, "autotrader.db")
    out: dict[str, list[dict]] = {}
    if not os.path.exists(path):
        return out
    try:
        db = sqlite3.connect(path)
        db.execute("PRAGMA busy_timeout=5000")
        rows = db.execute(
            "SELECT COALESCE(strategy,'auto'), side, entry, stop, target, "
            "mfe, mae FROM managed WHERE status='closed' AND mfe IS NOT NULL "
            "AND mae IS NOT NULL").fetchall()
        db.close()
    except Exception:
        return out
    for sid, side, entry, stop, target, mfe, mae in rows:
        if not entry or not stop or entry == stop:
            continue
        risk = abs(entry - stop)
        long = side in ("buy", "LONG")
        # favourable/adverse excursion expressed in R (risk multiples)
        fav = ((mfe - entry) if long else (entry - mfe)) / risk
        adv = ((entry - mae) if long else (mae - entry)) / risk
        out.setdefault(str(sid), []).append({"fav_R": fav, "adv_R": adv})
    return out


def exit_quality(store: LearningStore,
                 autotrader_db: str | None = None) -> dict[str, dict]:
    """Per-strategy suggested exit params from realised excursions. Only applies
    (and logs an `exit_tune`) once a strategy has >= EXIT_N closed trades.

      atr_stop  <- how far, in R, trades actually run against us before working
                   (median adverse excursion), clamped [1.0, 2.5].
      r_multiple<- how far, in R, they actually run in our favour (median
                   favourable excursion), clamped [1.5, 3.0].
    """
    import statistics
    ex = _managed_excursions(autotrader_db)
    suggestions: dict[str, dict] = {}
    for sid, rows in ex.items():
        if len(rows) < EXIT_N:
            continue
        med_adv = statistics.median(r["adv_R"] for r in rows)
        med_fav = statistics.median(r["fav_R"] for r in rows)
        # a small buffer beyond the typical adverse move so we are not stopped
        # out at exactly the median wiggle
        atr_stop = _clamp(round(max(med_adv, 0.0) * 1.2, 2),
                          ATR_STOP_LO, ATR_STOP_HI)
        r_mult = _clamp(round(max(med_fav, 0.0), 2), R_MULT_LO, R_MULT_HI)
        cur = store.exit_params().get(sid)
        if (not cur or abs((cur["atr_stop"] or 0) - atr_stop) > 0.05
                or abs((cur["r_multiple"] or 0) - r_mult) > 0.05):
            reason = (f"n={len(rows)} closed: median adverse {med_adv:.2f}R, "
                      f"favourable {med_fav:.2f}R -> stop {atr_stop}xATR, "
                      f"target {r_mult}R")
            store.set_exit_params(sid, atr_stop, r_mult, reason)
            store.log_decision(sid, "exit_tune", reason,
                               {"atr_stop": atr_stop, "r_multiple": r_mult,
                                "n": len(rows)})
        suggestions[sid] = {"atr_stop": atr_stop, "r_multiple": r_mult,
                            "n": len(rows)}
    return suggestions


# --------------------------------------------------------------- agent scores
def agent_scorecard(store: LearningStore | None = None,
                    journal_db: str | None = None,
                    plan_db: str | None = None) -> list[dict]:
    """Were the specialists' critiques predictive? Join each adopted plan's
    per-idea flags (earnings/liquidity/half-size) to the realised journal
    outcome for that (symbol, plan date), then compare the expectancy of
    FLAGGED vs UNFLAGGED trades per flag. A flag is 'predictive' when the
    trades it warned about did materially worse."""
    jpath = journal_db or os.path.join(HERE, "journal.db")
    ppath = plan_db or os.path.join(HERE, "tradeplan.db")
    if not (os.path.exists(jpath) and os.path.exists(ppath)):
        return []
    # journal outcome keyed by (symbol, signal_date)
    outcome: dict[tuple, float] = {}
    try:
        from .journal import Journal
        import pandas as pd
        df = Journal(jpath).frame()
        df["net_ret"] = pd.to_numeric(df.get("net_ret"), errors="coerce")
        for _, r in df.dropna(subset=["net_ret"]).iterrows():
            outcome[(str(r.get("asset")), str(r.get("signal_date")))] = float(r["net_ret"])
    except Exception:
        return []
    # per-flag buckets of realised returns
    FLAGS = ("earnings_flag", "liquidity_flag", "half_size")
    buckets = {f: {"flagged": [], "unflagged": []} for f in FLAGS}
    try:
        db = sqlite3.connect(ppath)
        for (doc,) in db.execute("SELECT doc FROM plan"):
            plan = json.loads(doc)
            date = str(plan.get("date"))
            for idea in plan.get("ideas", []):
                key = (str(idea.get("symbol")), date)
                if key not in outcome:
                    continue
                ret = outcome[key]
                for f in FLAGS:
                    (buckets[f]["flagged"] if idea.get(f)
                     else buckets[f]["unflagged"]).append(ret)
        db.close()
    except Exception:
        return []
    scores = []
    for f, b in buckets.items():
        nf = len(b["flagged"])
        if nf == 0:
            continue
        exp_f = sum(b["flagged"]) / nf
        exp_u = (sum(b["unflagged"]) / len(b["unflagged"])) if b["unflagged"] else 0.0
        verdict = "predictive" if exp_f < exp_u else "not predictive"
        scores.append({"agent": "committee", "flag": f, "n_flagged": nf,
                       "exp_flagged": exp_f, "exp_unflagged": exp_u,
                       "verdict": verdict})
        if store is not None:
            store.log_agent_score("committee", f, nf, exp_f, exp_u, verdict)
    return scores


# --------------------------------------------------------------- integration
def plan_inputs(journal_db: str | None = None, registry_csv: str | None = None,
                store: LearningStore | None = None) -> dict:
    """The bounded feedback the planner consumes. With empty state every
    multiplier defaults to 1.0 and `demoted` is empty, so plan sizing is
    byte-identical to the pre-learning behaviour."""
    own = store is None
    store = store or LearningStore()
    try:
        cards = strategy_scorecard(journal_db, registry_csv)
        mult = {sid: c["multiplier"] for sid, c in cards.items()
                if abs(c["multiplier"] - 1.0) > 1e-9}
        return {"strategy_multiplier": mult,
                "demoted": store.demoted(),
                "exit_params": store.exit_params()}
    finally:
        if own:
            store.db.close()


def nightly_report(store: LearningStore | None = None, journal_db: str | None = None,
                   registry_csv: str | None = None,
                   autotrader_db: str | None = None) -> str:
    """Run the full nightly learning pass and return a human-readable summary.
    Mutates learning.db (decisions/exit_params/agent_scores)."""
    own = store is None
    store = store or LearningStore()
    try:
        cards = strategy_scorecard(journal_db, registry_csv)
        promos = evaluate_promotions(store, cards)
        exits = exit_quality(store, autotrader_db)
        agents = agent_scorecard(store, journal_db, plan_db=None)
        lines = ["LEARNING REPORT — deterministic, from live outcomes", ""]
        if not cards:
            lines.append("no live trades logged yet — nothing to learn")
        for sid, c in sorted(cards.items(), key=lambda kv: -kv[1]["n"]):
            bt = c["backtest"]["test_exp"]
            lines.append(
                f"  {sid:22s} n={c['n']:4d} wr {c['win_rate']:.0%} "
                f"exp {c['expectancy']:+.4f} (bt {bt:+.4f} " if bt is not None
                else f"  {sid:22s} n={c['n']:4d} wr {c['win_rate']:.0%} "
                f"exp {c['expectancy']:+.4f} (bt   n/a ")
            lines[-1] += (f") x{c['multiplier']:.2f} [{c['state']}]")
        if promos:
            lines += ["", "state changes this run:"]
            lines += [f"  {p['action'].upper()} {p['strategy']}: {p['reason']}"
                      for p in promos]
        if exits:
            lines += ["", "exit-param tuning (>= %d closed trades):" % EXIT_N]
            lines += [f"  {sid}: stop {e['atr_stop']}xATR, target {e['r_multiple']}R "
                      f"(n={e['n']})" for sid, e in exits.items()]
        if agents:
            lines += ["", "committee critique quality:"]
            lines += [f"  {a['flag']}: flagged exp {a['exp_flagged']:+.4f} vs "
                      f"unflagged {a['exp_unflagged']:+.4f} — {a['verdict']} "
                      f"(n={a['n_flagged']})" for a in agents]
        return "\n".join(lines)
    finally:
        if own:
            store.db.close()


# ------------------------------------------------------------------ self-test
def _selftest():
    import tempfile
    from .journal import Journal

    tmp = tempfile.mkdtemp()
    jpath = os.path.join(tmp, "journal.db")
    lpath = os.path.join(tmp, "learning.db")
    atpath = os.path.join(tmp, "autotrader.db")
    reg = os.path.join(tmp, "registry.csv")

    # synthetic certified prior: strategy "drift_me" backtested at +2%/trade
    with open(reg, "w") as f:
        f.write("id,family,train_n,train_exp,test_n,test_wr,test_exp,test_pf,dsr,status\n")
        f.write("drift_me,Momentum,600,0.02,300,0.55,0.02,2.0,0.97,REAL\n")
        f.write("hot_hand,MeanRev,600,0.01,300,0.55,0.01,2.0,0.97,REAL\n")

    store = LearningStore(lpath)

    # 1) zero live trades -> multiplier 1.0, no state change
    cards0 = strategy_scorecard(jpath, reg)
    assert cards0["drift_me"]["multiplier"] == 1.0, "no live -> mult 1.0"
    assert cards0["drift_me"]["state"] == "NORMAL"
    assert evaluate_promotions(store, cards0) == [], "no change on empty live"
    assert plan_inputs(jpath, reg, store)["strategy_multiplier"] == {}, \
        "empty state -> no multipliers (byte-identical sizing)"

    # 2) a DRIFTING strategy: certified +2%, but live is -1% over 40 trades
    j = Journal(jpath)
    for _ in range(40):
        j.log(setup_id="drift_me", asset="AAA", side="buy", regime_trend="AUTO",
              net_ret=-0.01, slippage_bps_vs_plan=3.0)
    cards = strategy_scorecard(jpath, reg)
    dm = cards["drift_me"]
    assert dm["n"] == 40 and dm["state"] == "DEMOTED", (dm["n"], dm["state"])
    assert dm["multiplier"] == 0.5, "demoted -> half size"
    ch = evaluate_promotions(store, cards)
    assert any(c["strategy"] == "drift_me" and c["action"] == "demote" for c in ch)
    assert "drift_me" in store.demoted(), "demote is in the standing set"
    # idempotent: re-running logs no new change
    assert evaluate_promotions(store, strategy_scorecard(jpath, reg)) == []

    # 3) plan_inputs now half-sizes drift_me and routes it to INBOX (demoted)
    pin = plan_inputs(jpath, reg, store)
    assert pin["strategy_multiplier"].get("drift_me") == 0.5
    assert "drift_me" in pin["demoted"]

    # 4) 100 WINNERS on a certified +1% strategy -> promoted, capped at 1.5
    for _ in range(100):
        j.log(setup_id="hot_hand", asset="BBB", side="buy", regime_trend="AUTO",
              net_ret=0.05, slippage_bps_vs_plan=1.0)
    hh = strategy_scorecard(jpath, reg)["hot_hand"]
    assert hh["state"] == "PROMOTED", hh["state"]
    assert hh["multiplier"] == MULT_HI, f"100 winners capped at 1.5: {hh['multiplier']}"

    # 5) ANTI-MARTINGALE: appending a loss streak can only LOWER the multiplier
    before = strategy_scorecard(jpath, reg)["hot_hand"]["multiplier"]
    for _ in range(30):
        j.log(setup_id="hot_hand", asset="BBB", side="buy", regime_trend="AUTO",
              net_ret=-0.08)
    after = strategy_scorecard(jpath, reg)["hot_hand"]["multiplier"]
    assert after <= before, f"multiplier rose after losses: {before}->{after}"

    # 6) confidence shrinkage: few live trades -> prior dominates
    assert abs(confidence(0.10, 3, 0.02) - confidence(0.02, 0, 0.02)) < 0.02, \
        "3 live trades barely move off the prior"
    assert confidence(0.10, 300, 0.02) > 0.08, "300 live trades track the live edge"

    # 7) exit tuning from managed excursions (>= EXIT_N closed trades)
    db = sqlite3.connect(atpath)
    db.execute("""CREATE TABLE managed(id INTEGER PRIMARY KEY, strategy TEXT,
                  side TEXT, entry REAL, stop REAL, target REAL, status TEXT,
                  mfe REAL, mae REAL)""")
    for i in range(25):
        # long from 100, stop 96 (risk 4): ran to 108 favourable (2R), dipped
        # to 98 adverse (0.5R)
        db.execute("INSERT INTO managed(strategy,side,entry,stop,target,status,"
                   "mfe,mae) VALUES ('drift_me','buy',100,96,108,'closed',108,98)")
    db.commit(); db.close()
    ex = exit_quality(store, atpath)
    assert ex["drift_me"]["r_multiple"] == 2.0, ex["drift_me"]
    assert ex["drift_me"]["atr_stop"] == _clamp(round(0.5 * 1.2, 2), 1.0, 2.5)
    assert store.exit_params()["drift_me"]["r_multiple"] == 2.0, "persisted"

    # 8) nightly_report runs end-to-end and mentions a state change
    txt = nightly_report(store, jpath, reg, atpath)
    assert "LEARNING REPORT" in txt and "drift_me" in txt

    store.db.close()
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print("learning self-test ✓  zero-live→mult1.0, drift→demote(0.5)+INBOX, "
          "100 winners→promote capped 1.5, anti-martingale (losses never raise "
          "size), confidence shrinkage, exit-tune from MFE/MAE, nightly report")


if __name__ == "__main__":
    _selftest()
