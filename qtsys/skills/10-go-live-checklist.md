# SKILL: Go-Live + Claude Code Handoff
HANDOFF: unzip qtsys, `pip install -r` (pandas, numpy, scipy, scikit-learn,
fastapi, uvicorn; ccxt/alpaca-py/ib-async/oandapyV20 as needed). Point Claude
Code at this skills/ folder — each file is a self-contained agent procedure.
1. `python -m qtsys.data` refresh_real() -> all feeds current (Ops must show <2d old)
2. `python -m qtsys.validate` -> ALL T1-T7 PASS on refreshed data
3. `python -m qtsys.sweep` -> regenerate survivors on current data
4. `python -m uvicorn qtsys.server:app` -> terminal on :8000, replay mode
5. Broker keys (paper): QTSYS_BROKER=alpaca|ibkr|ccxt|oanda|tradier + creds
6. PAPER 2+ weeks: breach rate <5%, live-vs-backtest within 1 SE, limits fire
   correctly (test the kill switch ON PURPOSE once)
7. Intraday: fetch 1m/5m bars via feeds.get_history — same engine, windows are
   bars. Verify a survivor on the NEW timeframe before trading it there.
8. LIVE at SURVIVAL posture only, smallest venue size, scale per skill 07.
NEVER: skip paper; run live during data staleness; disable the safety plane.
