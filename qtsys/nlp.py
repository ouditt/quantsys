"""nlp.py — sentiment engine + LLM narrative for news.

Two jobs, two right tools (per the analysis):
  - `tag(texts)`  : per-headline pos/neg/neu. Uses **FinBERT** (finance-tuned
    transformer, accurate + deterministic) when available; falls back to the
    fast finance lexicon otherwise. Chosen by QTSYS_SENTIMENT (finbert|lexicon).
  - `narrative(symbol, headlines, llm_fn)` : one LLM call that synthesises the
    dominant story + key risk across an asset's headlines — the job the LLM is
    actually best at.

FinBERT loads lazily (first call downloads ~440MB, then cached in memory). Every
path is best-effort: a failure degrades to the lexicon / empty narrative, never
an exception into a request.
"""
from __future__ import annotations

import os

from .sentiment import score as _lex

_MODE = os.environ.get("QTSYS_SENTIMENT", "finbert").lower()
_finbert = None
_finbert_failed = False
_LABELMAP = {"positive": "pos", "negative": "neg", "neutral": "neu"}


def _load_finbert():
    global _finbert, _finbert_failed
    if _finbert is not None or _finbert_failed:
        return _finbert
    try:
        from transformers import pipeline
        _finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert",
                            top_k=None)
    except Exception:
        _finbert_failed = True
    return _finbert


def engine() -> str:
    if _MODE == "finbert" and _load_finbert() is not None:
        return "finbert"
    return "lexicon"


def tag(texts: list[str]) -> list[dict]:
    """Return [{sentiment, sent_score, engine}] aligned to `texts`."""
    if _MODE == "finbert":
        clf = _load_finbert()
        if clf is not None:
            try:
                out = []
                for res in clf([(t or "")[:512] for t in texts]):
                    d = {r["label"].lower(): r["score"] for r in res}
                    best = max(d, key=d.get)
                    out.append({"sentiment": _LABELMAP.get(best, "neu"),
                                "sent_score": round(d.get("positive", 0)
                                                    - d.get("negative", 0), 3),
                                "engine": "finbert"})
                return out
            except Exception:
                pass
    return [dict(zip(("sentiment", "sent_score"), _lex(t)), engine="lexicon")
            for t in texts]


def narrative(symbol: str, headlines: list[str], llm_fn) -> str:
    """One LLM call: the dominant narrative + key risk across the headlines."""
    if not llm_fn or not headlines:
        return ""
    joined = "\n".join(f"- {h}" for h in headlines[:15] if h)
    prompt = (f"You are a markets analyst. Based ONLY on these recent {symbol} "
              "headlines, write exactly two sentences: (1) the dominant "
              "sentiment/narrative, (2) the key risk or counterpoint. Be concise "
              f"and specific; do not add disclaimers.\n\n{joined}")
    try:
        return " ".join(llm_fn(prompt).split()).strip()[:400]
    except Exception:
        return ""
