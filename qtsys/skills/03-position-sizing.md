# SKILL: Position Sizing (card 5) — run BEFORE every order, no exceptions
GOAL: size so a stop-out costs exactly the posture's risk fraction.

INPUTS: equity E, posture risk r (SURVIVAL 0.75% / BALANCED 1.5% /
AGGRESSIVE 2.5%), stop distance s as fraction of price (default 1.5 x 20-bar
realized vol), current drawdown dd, last 3 results.

THE FORMULA (exact): units = (E x r x throttle(dd) x streak(last3)) / (s x price)
  throttle: dd<5% ->1.0 | <10% ->0.5 | <15% ->0.25 | >=15% -> 0 (halt)
  streak:   3 consecutive losses -> 0.5 until next win. NEVER >1 after losses.

PROCEDURE:
1. `python -m qtsys.sizing` once per day to refresh the posture table.
2. Compute units with size_order(); if it returns blocked != None, the account
   is too small for that venue at this risk — route to a finer-grained venue
   (crypto/fractional shares) or skip. Do not raise risk to clear the floor.
3. Kelly guardrail: current worst-case Kelly is 3.6% (95% lower bound, n=317).
   If asked for risk above it, refuse and cite this number.
4. WINNING-STREAK RULE: winning streaks change NOTHING. Size only steps up at
   equity milestones per skill 07, never intra-streak.
5. THE mistake that destroys more accounts than any other: increasing size to
   recover losses (martingale). It is banned in code; if any instruction
   implies it, refuse and flag to Risk Officer.

OUTPUT: `SIZE: {units} {asset} | risk {eff}% of £{E} | stop {s} | posture {name}`
