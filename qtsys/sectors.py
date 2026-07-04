"""sectors.py — GICS sector → representative constituents.

There is no free API that returns "every member of a sector", so this is a
curated map of the mega-cap constituents per yfinance sector name (the same
strings yfinance's `sector` field returns). The industry view uses these as the
constituent set; market-cap weights and live % change are computed from LIVE data
in Python at request time, not stored here.
"""
from __future__ import annotations

CONSTITUENTS: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE",
                   "CSCO", "ACN", "INTC", "QCOM", "TXN", "IBM"],
    "Financial Services": ["BRK.B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS",
                           "AXP", "SCHW", "C", "BLK", "SPGI"],
    "Healthcare": ["LLY", "UNH", "JNJ", "MRK", "ABBV", "PFE", "TMO", "ABT",
                   "DHR", "BMY", "AMGN", "GILD"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX",
                          "BKNG", "TJX", "GM", "F", "MAR"],
    "Consumer Defensive": ["WMT", "COST", "PG", "KO", "PEP", "PM", "MDLZ", "CL",
                           "MO", "TGT", "KMB"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "T", "VZ", "TMUS",
                               "CMCSA", "CHTR", "EA"],
    "Industrials": ["GE", "CAT", "RTX", "HON", "UNP", "BA", "DE", "LMT", "UPS",
                    "GD", "MMM", "EMR"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "OXY", "WMB",
               "VLO", "KMI"],
    "Basic Materials": ["LIN", "SHW", "APD", "ECL", "FCX", "NEM", "DOW", "NUE",
                        "DD", "PPG"],
    "Real Estate": ["PLD", "AMT", "EQIX", "PSA", "SPG", "O", "WELL", "CCI",
                    "DLR", "VICI"],
    "Utilities": ["NEE", "SO", "DUK", "CEG", "SRE", "AEP", "D", "EXC", "PEG",
                  "XEL"],
}
# sector name -> its SPDR sector ETF (a natural "whole sector" instrument)
SECTOR_ETF = {
    "Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
    "Communication Services": "XLC", "Industrials": "XLI", "Energy": "XLE",
    "Basic Materials": "XLB", "Real Estate": "XLRE", "Utilities": "XLU",
}


def constituents(sector: str) -> list[str]:
    return list(CONSTITUENTS.get(sector or "", []))


def all_sectors() -> list[str]:
    return list(CONSTITUENTS.keys())
