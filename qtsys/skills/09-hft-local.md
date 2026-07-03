# SKILL: HFT Stack, Local Activation (cards 13+15) — real ticks only
HONESTY FIRST: you are not co-located; your latency is ~100-1000ms. The only
retail-plausible HF edges are crypto maker spread capture on quiet pairs and
short-horizon book-imbalance — and most retail MM loses to adverse selection.
This stack exists to MEASURE that on your own recorded data before a penny moves.

SETUP (Claude Code / local):
1. `pip install ccxt`  (plus `ccxt.pro` for websockets if licensed)
2. RECORD real books: `python -m qtsys.hft record binance BTC/USDT 3600`
   -> recordings/binance_BTCUSDT.csv (1 snapshot/sec, top-5 depth + last trade)
3. BACKTEST on the recording: `python -m qtsys.hft backtest recordings/... mm`
   and `... imbalance`. The fill model is deliberately conservative: makers
   fill only on strict trade-through (whole visible queue assumed ahead of
   you); takers pay latency + touch. If it isn't profitable HERE, it will not
   be profitable live.
4. GATE: >= 20 recorded hours across >= 5 sessions, net pnl_bps > 0 after fees
   in EACH session majority, kill switch never hit -> paper via the gateway
   2 weeks -> smallest live size.
5. ARB (card 15): `python -m qtsys.hft arb` prints NET cross-venue spreads
   after both taker fees. actionable=True is rare and decays in seconds; treat
   as a monitor, not an income plan. Withdrawal fees/transfer time kill most
   of what remains — check both before believing a print.
TUNE: MMSpread.min_edge_bps >= 2x your actual maker fee + 2; inventory band
small; ImbalanceTaker theta 0.7-0.8. Never quote through news; the recorder
captures those sessions so the backtest shows you why.
