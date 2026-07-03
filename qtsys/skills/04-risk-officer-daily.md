# SKILL: Risk Officer Daily (cards 7+14)
GOAL: one page, every day: VaR, stress, clusters, limits — then the protocol.

PROCEDURE:
1. `python -m qtsys.portfolio_risk` (daemon files it to reports/ daily).
2. Read VaR99/CVaR99. If |CVaR99| > 1.3 x posture daily risk budget, book is
   oversized: instruct halving the largest cluster exposure TODAY.
3. Stress: the five REAL crash replays. If any scenario total < -25%, name the
   driving cluster and cap it (2022 replay = crypto+oil overlap is the usual culprit).
4. Clusters: list them; verify max 2 concurrent positions per cluster; if
   breached, the NEWEST position in the cluster is cut first.
5. Limits (hard-coded, gateway-enforced): day -3% = no new entries; week -6% =
   posture down one level for 2 weeks; DD -12% = KILL (flatten+halt, typed
   confirm to resume). Your job is to announce distance-to-limit every day:
   `headroom: day {x}% of -3% | week {y}% of -6% | DD {z}% of -12%`.
6. If the kill switch fired: do NOT restart trading. Write the cause note,
   require the human's typed RESUME, and schedule a full sweep re-verify.

FORBIDDEN: waiving any limit "because the setup is good"; expressing limits
as suggestions; letting correlation breaches age past one session.
