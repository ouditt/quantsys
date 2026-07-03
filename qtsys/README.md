# QTSYS — a trading engine you can test to destruction before spending a cent

A compact, working implementation of the blueprint's core loop: **data → signals →
candidate trades → agent trade-selection → fee-exact backtest → reliability
verdict** — built so every claim it makes is verified at **$0 cost** before any
data subscription, server, or live dollar is committed.

## The zero-outlay test ladder (all free)

| Step | Costs | What it proves |
|---|---|---|
| 1. `python -m qtsys.validate` | $0 | The machinery is honest (see tests below) |
| 2. `python -m qtsys.demo` | $0 | Trade selection lifts win rate *and* expectancy out-of-sample, net of fees — on synthetic data with a known edge and on 150 years of real S&P 500 data (free Shiller dataset, bundled) |
| 3. Locally: `pip install yfinance ccxt` | $0 | Same pipeline on real daily equities/ETF/crypto data |
| 4. Alpaca **paper** account / exchange testnets | $0 | Live-data execution, fills, tracking error vs backtest |
| 5. Live, staged (10% → 25% → 50% → 100% of target capital) | capital only | The only step that ever costs money — and it is gated on steps 1–4 |

The ~$500/month data/infra budget in the blueprint is an *optional later upgrade*
(point-in-time fundamentals, options chains, better intraday feeds). Nothing in
this repo requires it.

## The objective metric (per the mandate)

The system maximises **net expectancy per trade**:

```
E = win_rate × avg_win_net − (1 − win_rate) × avg_loss_net
```

This is literally "win rate × profit in the win", minus the loss side, with
**every trade's P&L net of all fees** (commission + half-spread + slippage,
charged per side; a trade that gains less than its round-trip cost counts as a
loss). Win rate and average win are first-class reported numbers.

Two guardrails keep that goal honest, because a naive "maximise win rate"
objective selects blow-up profiles (many tiny wins, rare catastrophic losses):

1. **Profit factor** must exceed 1 and is always reported next to win rate.
2. **Deflated Sharpe Ratio (DSR)** — the probability the edge is real after
   correcting for how many variants were tried — must be ≥ 0.95 before anything
   advances. On pure noise, the suite *proves* this gate says no (test T6).

## How the agent "selects the right trades"

`select.py` implements **meta-labeling**: the primary strategy picks the side of
each candidate trade; a gradient-boosted model is trained on whether such trades
were historically profitable **net of fees**, using **purged, embargoed K-fold
cross-validation** (plain k-fold on overlapping trade labels leaks the answer
and manufactures fake skill). The P(win) threshold is chosen only on
training-period out-of-fold predictions — maximising net expectancy with a
minimum-coverage constraint — then frozen before it touches the held-out window.
Skipping likely losers is the *honest* mechanism by which win rate rises.

## Verified run (this exact code, this environment)

**Validation suite — all passing:**

```
T1 PASS  fee accounting exact: 11 trades on a flat series lose exactly 0.2000% each (the round-trip fee)
T2 PASS  no look-ahead: removing the future leaves all earlier signals and every feature value unchanged
T3 PASS  QC gates refuse all 3 injected defects (negative price, duplicate timestamp, high<low)
T4 PASS  purged CV: 54 train/test pairs checked, zero lifespan overlaps (embargo respected)
T5 PASS  known edge recovered: expectancy +0.644%/trade net; with 10x fees it drops to -0.496%
T6       pure-noise sweep: best of 16 configs shows in-sample Sharpe 1.65 — tempting…
T6 PASS  …but Deflated Sharpe = 0.671 → LIKELY NOISE/OVERFIT — do not deploy
```

**Demo, Part A — held-out 30% window, all numbers NET of fees (6 bps round trip):**

| Momentum strategy | trades | win rate | avg win | expectancy/trade | profit factor | Sharpe | DSR |
|---|---|---|---|---|---|---|---|
| Base (all signals) | 46 | 58.7% | +4.57% | +0.518% | 1.24 | 1.22 | 0.979 |
| **Agent-selected** | 37 | **62.2%** | +4.52% | **+1.005%** | **1.56** | **2.33** | 1.000 |

| Mean-reversion strategy | trades | win rate | avg win | expectancy/trade | profit factor | Sharpe | DSR |
|---|---|---|---|---|---|---|---|
| Base (all signals) | 127 | 62.2% | +2.61% | +0.337% | 1.26 | 1.02 | 0.956 |
| **Agent-selected** | 86 | **70.9%** | +2.34% | **+0.738%** | **1.80** | **2.18** | 0.999 |

**Demo, Part B — real S&P 500 data, 1871→present, fees on every switch:**

```
buy & hold             CAGR 9.36%   Sharpe 0.71   maxDD -81.8%
SMA-10 timing (net)    CAGR 9.34%   Sharpe 0.99   maxDD -46.3%
round trips: 105   win rate (net) 61.0%   avg win +29.57%   avg loss -2.90%
expectancy +16.89%/trade   profit factor 15.94   DSR 1.000
```

(The timing rule's real value is cutting the worst drawdown from −82% to −46% at
the same CAGR — a reminder that no single metric, including win rate, is the
whole story. Trades here are multi-month/multi-year holds.)

## Why the target is expectancy at a 60–70% win rate, not "96%"

Nobody honest trades at 96%. The best track record ever recorded (Renaissance
Medallion) was right on barely more than half its trades; the great trend funds
win 35–45%. If you *force* 96%, the optimizer obliges with strategies that win
tiny amounts 96 times and lose everything on the 97th (far-OTM option selling,
martingale sizing, no stops). The selector above raises win rate the only
durable way — by declining bad trades — and every improvement is measured on
data the model never saw, after fees, behind a luck-correction gate.

## Files

```
qtsys/
├── data.py        free data: synthetic (known-edge) generator, real Shiller CSV,
│                  yfinance/ccxt loaders for local use; QC gates that REFUSE bad data
├── signals.py     look-ahead-safe indicators, event features, 2 primary strategies
├── backtest.py    next-bar execution, triple-barrier exits, per-side FeeModel,
│                  net-of-fee trade ledger, equity curve
├── select.py      the trade-selection agent: meta-labeling + purged/embargoed CV
├── metrics.py     win rate, avg win/loss, expectancy, profit factor, Sharpe,
│                  Sortino, maxDD, CAGR, PSR, DSR, min track record, verdicts
├── risk.py        vol targeting, fixed-fractional sizing, drawdown throttle,
│                  kill-switch thresholds
├── validate.py    the honesty suite (T1–T7)
├── demo.py        end-to-end demo (synthetic + real data)
├── sp500_monthly.csv   real S&P 500 monthly data since 1871 (free)
│
│   — live stack —
├── brokers.py     Broker interface + adapters: PaperBroker (built-in venue with
│                  tick-driven limit fills), Alpaca, IBKR (ib_async), ccxt,
│                  Oanda, Tradier; RiskLimits + ExecutionGateway (pre-trade
│                  checks, halt/kill, resume) in front of EVERY venue
├── adapters.py    market-data fetchers (alpaca-py, ib_async, yfinance, ccxt)
├── feeds.py       streaming quote plumbing for the server
├── agents.py      24/7 agent daemon: 6 agents on asyncio loops, master +
│                  per-agent toggles persisted in SQLite, append-only log,
│                  propose-only (no order path)
├── server.py      FastAPI: quotes, history, account, positions, orders
│                  (place/cancel), kill/resume, agents status/toggle/log —
│                  and it serves the terminal at /
├── terminal.html  the Bloomberg-style terminal (single file, zero build step)
│
│   — the daily operating system (v1.3) —
├── strategies.py  17-spec / 14-family library (breakouts, fades, TS & x-sec
│                  momentum, real-pairs stat-arb, VIX regime/sentiment, calendar)
├── sweep.py       tests EVERYTHING, charges every trial to the DSR gate;
│                  writes registry_summary.csv + per-(strategy,asset) attribution
├── routine.py     morning scan + professional chart read; setups ranked by
│                  their OWN real out-of-sample track records
├── sizing.py      posture math (Kelly bounds, throttle, streak rules, venue
│                  floors) + milestone scaling roadmap, from REAL trade returns
├── portfolio_risk.py  hist VaR/CVaR 99%, REAL crash-window stress replays,
│                  correlation clusters, hard limit protocol
├── journal.py     before/during/after trade log; breach-rate analytics
├── options.py     BS + Greeks (10k contracts < 5 ms), IV, American binomial
├── optimizer.py   constrained max-Sharpe / risk parity (100+ assets capable)
├── factors.py     cross-asset factor model + alpha/beta decomposition
├── hft.py         REAL-tick recorder + conservative book-replay backtester,
│                  MM & imbalance strategies, cross-venue arb scan (local)
└── skills/        11 step-by-step agent playbooks (small-LLM executable):
                   operating schedule, scan, chart read, sizing, risk, plan,
                   journal, scaling, research pipeline, HFT local, go-live
```

## The live stack

**Run it:** `pip install fastapi uvicorn` then `python -m uvicorn qtsys.server:app`
and open `http://127.0.0.1:8000`. With no broker keys it runs the built-in
paper venue (simulated ticks, real gateway, real fills-at-limits logic) — still
$0. Set `QTSYS_BROKER=alpaca` (+ keys) etc. to point the same terminal at a
real paper account. `terminal.html` also works opened directly as a file — it
detects the API and falls back to a self-contained demo mode.

**Terminal (UI/UX):** near-black + amber, all-mono, command line always ready
(`NVDA` opens the security page, `NVDA GP` jumps to the chart, `POS`, `RISK`,
`AGT`, `BUY SPY 10 LMT 480`, `KILL`). Every ticker everywhere — watchlists,
positions, orders, news tags, the tape — is a link to that instrument's
**security detail page** (deep-linkable `#/sec/SYM`, opens in a new tab too):
candlestick chart with SMA20/100, volume and crosshair; TF switching; stats
grid; the **agent read** (regime, trend, P(win), TAKE / STAND ASIDE); your
position; your working orders with inline cancel; an order ticket whose
pre-trade check preview mirrors the real gateway limits; per-symbol news.
PORTFOLIO shows equity/cash/buying power/day P&L/open P&L/leverage/drawdown,
every position in detail, every **pending order** with status and cancel, and
the fill history. RISK shows the throttle ladder with a "you are here" marker
and the kill switch (typed confirm). AGENTS shows all six with heartbeats.

**Agents, 24/7:** the daemon's loops run around the clock (crypto and FX don't
sleep). A master switch and per-agent switches gate all work — `on standby =
master AND agent enabled` — state persists across restarts, every toggle is
logged. Toggling agents off never disables the deterministic safety plane
(gateway checks, throttle, kill switch): those are not agents and have no off
switch in the panel. Agents propose; they have no code path to an order.

## Running on real data locally ($0)

```bash
pip install numpy pandas scikit-learn yfinance ccxt      # all free
python - <<'PY'
from qtsys.data import fetch_yfinance, fetch_ccxt
from qtsys.signals import momentum_events, feature_frame
from qtsys.backtest import simulate_trades, BarrierSpec, US_EQUITY_FEES, CRYPTO_FEES
from qtsys.select import TradeSelector
from qtsys.metrics import trade_stats
import numpy as np

df = fetch_yfinance("SPY", start="2010-01-01")           # or fetch_ccxt("BTC/USDT")
ev = momentum_events(df, "SPY")
trades = simulate_trades(df, ev, BarrierSpec(), US_EQUITY_FEES, feature_frame(df))
split = int(len(df) * 0.7)
train = [t for t in trades if t.i_exit < split]
test  = [t for t in trades if t.i_entry >= split]
sel = TradeSelector().fit(train)
picked = sel.filter(test)
print("base    :", trade_stats(np.array([t.net_ret for t in test])).row())
print("selected:", trade_stats(np.array([t.net_ret for t in picked])).row())
PY
```

Then wire the same trade decisions to an **Alpaca paper account** (free) via the
blueprint's execution-gateway pattern, run 60–90 days, compare tracking error to
the backtest, and only then stage real capital.

## Honest limitations

Synthetic results prove the *machinery* (fee exactness, leak-freedom, luck
rejection) — not that any particular edge exists in live markets; that is what
steps 3–4 of the ladder are for, still at $0. Barrier exits check closes (no
intrabar fills — conservative). yfinance is research-grade, not an execution
feed. Single-position-per-asset ledger keeps win-rate accounting unambiguous.
None of this is investment advice; trading risks loss of capital.
