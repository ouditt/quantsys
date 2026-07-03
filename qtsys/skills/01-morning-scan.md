# SKILL: Morning Scan (cards 1+4) — Research Analyst
GOAL: rank every fresh setup on every asset by ITS OWN real out-of-sample
track record, in under 60 seconds, so the best evidence is acted on first.

TRIGGER: session start, or `python -m qtsys.routine`, or GET /api/scan.

PROCEDURE (follow exactly):
1. Run `python -m qtsys.routine`. If it errors, report the error and STOP.
2. Read the REGIMES block. For each asset note trend (UP/DOWN), vol
   (CALM/NORMAL/HIGH). HIGH vol => all sizes later get the throttle multiplier.
3. Read RANKED SETUPS. The ranking key is historical net expectancy per trade
   for that exact (strategy, asset) pair, NEVER win rate. Ignore any instinct
   to reorder by win rate: a 96% win rate with negative expectancy is the
   blow-up profile.
4. Apply the tier rule with zero exceptions:
   - SURVIVOR  (spec DSR >= 0.95): tradable at full posture risk
   - CANDIDATE (0.80-0.95): tradable at HALF posture risk, paper preferred
   - WATCH-ONLY (< 0.80): journal it, never trade it
5. Drop any setup whose (strategy, asset) hist_exp is NEGATIVE even if the
   spec is a SURVIVOR (e.g. donchian_20 on WTI): the family works, that pair
   does not. Evidence is pair-level.
6. Cluster check: consult portfolio_risk.clusters(). Max 2 concurrent
   positions per cluster (FX majors are ONE cluster; BRENT+WTI one; BTC+ETH one).
7. For each surviving setup produce the ORDER TICKET LINE:
   `SIDE ASSET | strategy | hist_exp | n | stop = 1.5 x 20-bar vol | size from skill 03`
8. If zero setups survive: output exactly "No qualified setups — standing
   aside IS a position." Do not manufacture a trade.

OUTPUT TEMPLATE:
  DATE / regimes summary (1 line) / numbered ticket lines / cluster notes.

FORBIDDEN: trading WATCH-ONLY tiers; reordering by win rate; adding
instruments not in the scan; running before data freshness is checked (Ops).
