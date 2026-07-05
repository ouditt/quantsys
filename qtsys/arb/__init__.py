"""qtsys.arb — arbitrage & stat-arb skills (Fable core).

Three composable, self-contained skills the Arb Strategist agent (and any
backtest) can call:

  pairs       cointegration stat-arb: Engle-Granger test (own ADF, no
              statsmodels), OU half-life, z-score signals, honest
              train/test-split backtest net of fees.   -> EXECUTABLE
  triangular  crypto triangular-loop monitor: walks REAL L2 depth on every
              leg, nets per-leg taker fees, both loop directions.
              Signals only fire when edge > cost.      -> EXECUTABLE (crypto)
  cip         covered-interest-parity monitor from FRED policy/bill rates:
              rate differentials + theoretical forward points. Without live
              FX forward quotes the *deviation* is unobservable, and Alpaca
              has no forward market to hedge on — ANALYSIS/ALERTS ONLY.

Feasibility honesty (why only these three): spatial arb needs 2+ venues
(we have one broker) and latency arb needs co-location — neither is built.

Run `python -m qtsys.arb` for all self-tests.
"""
from . import cip, pairs, triangular  # noqa: F401

__all__ = ["pairs", "triangular", "cip"]
