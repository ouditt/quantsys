# SKILL: Professional Chart Read (card 6) — any asset, any timeframe
GOAL: state what the chart is ACTUALLY saying vs what it appears to say.

PROCEDURE:
1. `python -c "from qtsys.routine import chart_read; import json; print(json.dumps(chart_read('SYM'), indent=1))"`
2. Report in this order, one line each:
   a. REGIME: trend + vol from the output. The regime is the message.
   b. STRUCTURE: higher-highs/higher-lows = uptrend intact; lower-lows =
      downtrend intact; range/transition = no directional edge, fade extremes only.
   c. LEVELS: the 3 supports below and 3 resistances above with dates. A level
      touched more recently matters more.
   d. LOCATION: % of 52-week range. <20% or >80% = extension zone, mean-rev
      setups gain edge; 40-60% = no-man's land, only trend setups qualify.
   e. THE DIVERGENCE LINE: quote surface_read vs deeper_read. If the last
      candle is AGAINST the regime, say: "retail reads the candle; the regime
      pays." That is what untrained eyes misread.
3. Map to the library: RSI<30 in UP regime => meanrev_rsi2 territory;
   close > all resistance => donchian/roll_high territory; vol CALM after
   HIGH => squeeze watch.
4. NEVER predict a price. State structure, levels, and which registered
   setups the chart currently qualifies for. If none: say none.

Timeframes: identical procedure on any bar size — pass a 1-minute frame
locally and read it the same way. Windows are in bars, not days.
