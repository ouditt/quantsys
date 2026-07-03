# SKILL: New Strategy Pipeline (cards 8, 11-img, 16) — from idea to gate
Any idea (yours, a video's, an LLM's) goes through this pipeline. No step may
be skipped; most ideas die — that is the system working.

1. SPEC: write entry/exit as a pure function emitting Events; every window in
   BARS; stops/targets as vol multiples (BarrierSpec). Add to REGISTRY with a
   one-line economic WHY (who is on the other side and why do they pay you?).
2. GRID (the 1,000+ tester): expand honestly —
   `for p in itertools.product(fasts, slows, ...): REGISTRY.append(Spec(...))`
   n_trials in sweep.py charges AUTOMATICALLY. 1,000 trials makes the DSR gate
   harder; that is correct, never cap the count to flatter a result.
3. RUN: `python -m qtsys.sweep`. Read YOUR spec's row: train_exp sign first,
   then test, then DSR with all trials charged.
4. ML LAYER (card 16): if base test_exp > 0 but DSR < 0.95, fit the selector:
   `TradeSelector().fit(train_trades)` then filter test. Purged CV + frozen
   threshold are inside; do not touch them. Report selected stats + DSR(n=1
   for the layer) alongside base. The honest claim is per-trade expectancy,
   never a promised Sharpe band.
5. VERDICT: DSR>=0.95 -> paper at half risk 2 weeks -> full. 0.80-0.95 ->
   challenger queue. <0.80 -> archive with its stats; deleting losers hides
   the trial count and corrupts every future DSR.
FORBIDDEN: testing on the test set twice; per-asset cherry-picking without
charging trials; any parameter chosen after seeing test results.
