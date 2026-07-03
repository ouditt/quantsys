# SKILL: Journal (card 9) — Report Writer
Log EVERY trade with the before/during/after fields in journal.FIELDS_*.
The one non-negotiable field on losers: loss_within_plan (True/False) +
breach_code from {NONE,SIZE,ENTRY,STOP,LIMITS,DATA}.

WEEKLY REVIEW PROCEDURE:
1. `python -m qtsys.journal` -> read the (setup, regime) expectancy table.
2. HIDDEN-PATTERN CHECKS, in order: (a) any setup whose live expectancy is
   below its backtest value by >1 SE for 2 straight weeks -> flag DECAY to
   Validation; (b) wins concentrated in one regime -> propose a regime gate as
   a challenger, don't hand-tune live; (c) slippage_bps_vs_plan trending up ->
   size or venue problem, tell Risk.
3. THE ONE QUESTION, per losing trade: "was this loss inside the plan?"
   Inside = tuition, no action. Outside = a leak: name the breach_code, name
   the fix, and the fix is always a RULE/CODE change, never "try harder".
4. Headline metric: breach rate. <5% acceptable; above -> freeze new setups,
   fix process first. An agentic book's psychology IS its breach rate.
OUTPUT: the module's report + max 3 action items, each mapped to a rule change.
