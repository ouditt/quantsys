"""sentiment.py — fast, deterministic finance-headline sentiment.

Lexicon-based scorer tuned for market news: it tags each headline pos/neg/neu
instantly and for free (no model call, no network), so 25 headlines per asset
cost nothing. A leading negator ("no", "not", "denies", "fails to") flips the
next matched term. This is intentionally simple and auditable; the daemon's
llm_fn could be swapped in later for a model-scored version.
"""
from __future__ import annotations

import re

_BULL = {
    "surge", "surges", "surged", "soar", "soars", "soared", "jump", "jumps",
    "jumped", "rally", "rallies", "rallied", "gain", "gains", "gained", "beat",
    "beats", "upgrade", "upgraded", "record", "high", "highs", "bullish",
    "outperform", "buy", "growth", "profit", "profits", "rise", "rises", "rose",
    "top", "tops", "topped", "boost", "boosts", "strong", "wins", "win",
    "breakthrough", "approval", "approved", "partnership", "expands", "expand",
    "momentum", "upside", "raises", "raised", "rebound", "rebounds", "climb",
    "climbs", "soaring", "optimism", "upbeat", "accelerate", "accelerates",
    "beats estimates", "tops estimates", "outperforms", "double", "doubles",
    "all-time", "breakout", "bull",
}
_BEAR = {
    "plunge", "plunges", "plunged", "drop", "drops", "dropped", "fall", "falls",
    "fell", "slump", "slumps", "slumped", "crash", "crashes", "crashed", "miss",
    "misses", "missed", "downgrade", "downgraded", "cut", "cuts", "low", "lows",
    "bearish", "underperform", "sell", "loss", "losses", "decline", "declines",
    "weak", "warning", "warns", "lawsuit", "probe", "investigation", "recall",
    "halt", "halts", "bankruptcy", "fraud", "slash", "slashes", "tumble",
    "tumbles", "sink", "sinks", "dip", "dips", "concern", "concerns", "fear",
    "fears", "risk", "risks", "selloff", "short", "slide", "slides", "plummet",
    "plummets", "sued", "delay", "delays", "layoffs", "downturn", "slowdown",
    "misses estimates", "bear", "collapse", "collapses", "default",
}
_NEG = {"no", "not", "never", "denies", "denied", "without", "fails", "fail",
        "avoids", "avoid", "halts", "ends"}


def score(text: str) -> tuple[str, int]:
    """Return (label, net) where label in {'pos','neg','neu'}."""
    if not text:
        return "neu", 0
    words = re.findall(r"[a-z'\-]+", text.lower())
    net, negate = 0, False
    for w in words:
        s = 1 if w in _BULL else (-1 if w in _BEAR else 0)
        if s:
            net += -s if negate else s
            negate = False
        elif w in _NEG:
            negate = True
        else:
            negate = False
    return ("pos" if net > 0 else "neg" if net < 0 else "neu"), net
