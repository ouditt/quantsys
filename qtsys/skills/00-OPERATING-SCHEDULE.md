# OPERATING SCHEDULE — every card, every day it's relevant
The 24/7 daemon (`qtsys/agents.py`) fires these automatically at the cadence
below (SQLite-gated, restart-safe). Run any of them manually with the command.

| # | Card | Module | Agent | Cadence | Command |
|---|------|--------|-------|---------|---------|
| 1 | High-probability setups every morning | routine.scan | Research Analyst | daily | `python -m qtsys.routine` |
| 4 | (same card, ranked format) | routine.morning_briefing | Research Analyst | daily | same |
| 6 | Read any chart professionally | routine.chart_read | Research Analyst | on demand | `python -c "from qtsys.routine import chart_read; print(chart_read('WTI'))"` |
| 5 | Position sizing that protects capital | sizing | before EVERY order | always | `python -m qtsys.sizing` |
| 7 | Risk management framework | portfolio_risk | Risk Officer | daily | `python -m qtsys.portfolio_risk` |
| 14| 99% VaR + crash stress | portfolio_risk | Risk Officer | daily | same |
| 9 | Journal that improves results | journal | Report Writer | daily wrap + weekly review | `python -m qtsys.journal` |
| 11| Complete trading plan | skills/05-trading-plan.md | you + agents | living doc | read it |
| 2/12| Scale a profitable strategy | sizing.scaling_roadmap | Strategy Engineer | at each equity milestone | in 07-scaling-review |
| 8 | Strategy from scratch | strategies + sweep | Strategy Engineer | weekly challengers | `python -m qtsys.sweep` |
| 11(img)| 1,000+ strategy stat-arb tester | sweep (grid-expand) | Validation Officer | weekly gate | skills/08 §grid |
| 16| ML alpha model | select.TradeSelector | Strategy Engineer | per candidate | skills/08 §ml |
| 10| Options pricing engine | options | on demand | — | `python -m qtsys.options` |
| 17| Portfolio optimizer | optimizer | monthly rebalance | monthly | `python -m qtsys.optimizer` |
| 12(img)| Factor model | factors | monthly | monthly | `python -m qtsys.factors` |
| 13| HFT | hft | local only | when recording exists | skills/09 |
| 15| Crypto arbitrage | hft.scan_arb | local only | on demand | `python -m qtsys.hft arb` |

Learning loop ("focus on the best setups over time"): journal expectancy by
(setup, regime) -> registry_results.csv attribution -> scan ranking -> capital
follows evidence. Survivors get posture risk, candidates half, watch-only none.
