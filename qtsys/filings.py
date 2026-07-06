"""filings.py — SEC EDGAR filings + LLM summaries (free, official source).

Public company disclosure straight from the SEC:
  - `filings(sym, forms, n)` -> recent filings (form, date, accession, doc URL,
    description) for an equity, via the EDGAR submissions API.
  - `filing_text(url, max_chars)` -> plain text of a filing document (HTML
    stripped), for feeding to an LLM.
  - `summary(sym, llm_fn, ...)` -> a short LLM brief of the latest material
    filing (10-K / 10-Q / 8-K), cached.

EDGAR requires a descriptive User-Agent with a contact address (set
SEC_USER_AGENT, else a sane default). Everything is best-effort and cached;
a failure returns the last good value or an empty record, never an exception
into the agent loop. Equities only — crypto/FX/commodities have no CIK.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request

_CACHE: dict = {}
_CACHE_MAX = 512
_CIK_TTL = 24 * 3600          # ticker->CIK map is essentially static
_SUB_TTL = 3600               # submissions refresh hourly
_TEXT_TTL = 12 * 3600
_SUMMARY_TTL = 12 * 3600

# forms that actually move a thesis, in rough order of materiality
MATERIAL_FORMS = ("10-K", "10-Q", "8-K", "20-F", "6-K", "S-1", "424B4",
                  "13D", "13G", "SC 13D", "SC 13G", "4", "DEF 14A")


def _get(url: str, timeout: int = 20):
    hdr = {"User-Agent": os.environ.get("SEC_USER_AGENT")
           or "QTSYS research aj684817@gmail.com",
           "Accept-Encoding": "gzip, deflate"}
    req = urllib.request.Request(url, headers=hdr)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            import gzip
            raw = gzip.decompress(raw)
    return raw


def _cached(key: str, ttl: int, fn):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    try:
        val = fn()
    except Exception:
        val = hit[1] if hit else None
    _CACHE[key] = (now, val)
    if len(_CACHE) > _CACHE_MAX:              # size-bounded: evict oldest
        for k in sorted(_CACHE, key=lambda k: _CACHE[k][0])[:len(_CACHE) // 4]:
            _CACHE.pop(k, None)
    return val


# ---------------------------------------------------------------- ticker->CIK
def _cik_map() -> dict:
    return _cached("cikmap", _CIK_TTL, _load_cik_map) or {}


def _load_cik_map() -> dict:
    raw = _get("https://www.sec.gov/files/company_tickers.json")
    data = json.loads(raw)
    out = {}
    for row in data.values():
        t = str(row.get("ticker", "")).upper()
        if t:
            out[t] = {"cik": int(row["cik_str"]), "name": row.get("title", "")}
    return out


def cik_for(sym: str):
    sym = (sym or "").upper().split("/")[0].strip()
    return _cik_map().get(sym)


# ------------------------------------------------------------------- filings
def filings(sym: str, forms=None, n: int = 15) -> list[dict]:
    """Recent EDGAR filings for an equity ticker. `forms` optionally restricts
    to specific form types (e.g. ("10-K","10-Q","8-K")). Newest first."""
    ent = cik_for(sym)
    if not ent:
        return []
    key = f"sub:{ent['cik']}"
    rows = _cached(key, _SUB_TTL, lambda: _submissions(ent["cik"])) or []
    if forms:
        fset = {f.upper() for f in forms}
        rows = [r for r in rows if r["form"].upper() in fset]
    return rows[:n]


def _submissions(cik: int) -> list[dict]:
    raw = _get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    data = json.loads(raw)
    rec = data.get("filings", {}).get("recent", {})
    forms = rec.get("form", [])
    dates = rec.get("filingDate", [])
    accs = rec.get("accessionNumber", [])
    docs = rec.get("primaryDocument", [])
    descs = rec.get("primaryDocDescription", [])
    reports = rec.get("reportDate", [])
    out = []
    for i in range(len(forms)):
        acc = accs[i] if i < len(accs) else ""
        doc = docs[i] if i < len(docs) else ""
        acc_nodash = acc.replace("-", "")
        url = (f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"
               if doc else
               f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}")
        out.append({
            "form": forms[i],
            "date": dates[i] if i < len(dates) else "",
            "report_date": reports[i] if i < len(reports) else "",
            "accession": acc,
            "title": (descs[i] if i < len(descs) and descs[i] else forms[i]),
            "url": url,
        })
    # already newest-first from EDGAR, but be defensive
    out.sort(key=lambda x: x["date"], reverse=True)
    return out


# --------------------------------------------------------------- filing text
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_NL = re.compile(r"\n{3,}")


def filing_text(url: str, max_chars: int = 40000) -> str:
    """Plain text of a filing document (HTML/XBRL tags stripped)."""
    if not url or "/Archives/" not in url:
        return ""
    return _cached(f"txt:{url}:{max_chars}", _TEXT_TTL,
                   lambda: _filing_text(url, max_chars)) or ""


def _filing_text(url: str, max_chars: int) -> str:
    raw = _get(url, timeout=30)
    html = raw.decode("utf-8", "ignore")
    # drop scripts/styles/head and the inline-XBRL hidden header (context/fact
    # definitions that otherwise flood the top of modern 10-Q/10-K docs)
    html = re.sub(r"(?is)<(script|style|head)\b.*?</\1>", " ", html)
    html = re.sub(r"(?is)<ix:(hidden|header|references|resources)\b.*?</ix:\1>",
                  " ", html)
    html = re.sub(r'(?is)<[^>]*style="[^"]*display:\s*none[^"]*".*?>', " ", html)
    txt = _TAG.sub(" ", html)
    for a, b in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&#160;", " "), ("&#39;", "'"), ("&quot;", '"'), ("&#8217;", "'")):
        txt = txt.replace(a, b)
    txt = _WS.sub(" ", txt)
    txt = _NL.sub("\n\n", txt)
    txt = "\n".join(ln.strip() for ln in txt.splitlines())
    return txt.strip()[:max_chars]


# ------------------------------------------------------------------- summary
def summary(sym: str, llm_fn=None, forms=("10-K", "10-Q", "8-K")) -> dict:
    """LLM brief of the latest material filing for a ticker. Returns
    {form, date, url, summary}. Without an llm_fn, returns metadata only."""
    fs = filings(sym, forms=forms, n=3)
    if not fs:
        return {}
    latest = fs[0]
    base = {"form": latest["form"], "date": latest["date"], "url": latest["url"]}
    if not llm_fn:
        return {**base, "summary": ""}
    key = f"sum:{latest['url']}"
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _SUMMARY_TTL:
        return {**base, "summary": hit[1]}
    txt = filing_text(latest["url"], max_chars=24000)
    if not txt:
        return {**base, "summary": ""}
    from .llm import guard
    prompt = guard(
        f"You are an equity analyst. Summarise the SEC {latest['form']} filing "
        f"for {sym} (filed {latest['date']}) in the data block, in 4-6 tight "
        "bullet points a portfolio manager can act on: material changes, "
        "guidance, risks, anything that moves the thesis. Be specific with "
        "numbers. No preamble.", txt)
    try:
        out = llm_fn(prompt).strip()
    except Exception:
        out = ""
    _CACHE[key] = (time.time(), out)
    return {**base, "summary": out}


if __name__ == "__main__":
    import sys
    s = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print("CIK:", cik_for(s))
    for f in filings(s, n=8):
        print(f"{f['date']}  {f['form']:8}  {f['title'][:50]}")
        print("   ", f["url"])
