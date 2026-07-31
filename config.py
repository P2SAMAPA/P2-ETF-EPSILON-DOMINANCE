"""
config.py  —  Configuration for Epsilon-Dominance Engine
=========================================================

Defines:
  - UNIVERSES: ETF ticker sets
  - BENCHMARKS: Benchmark assets for comparison
  - WINDOWS: Rolling windows for testing
  - E-PROCESS: Thresholds and parameters
"""

# ── HuggingFace ──────────────────────────────────────────────────────────────

HF_TOKEN = ""
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-epsilon-dominance-results"


# ── ETF Universes ────────────────────────────────────────────────────────────

UNIVERSES = {
    "FI_COMMODITIES": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
    ],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}


# ── Benchmarks ──────────────────────────────────────────────────────────────

BENCHMARKS = {
    "SPY": "S&P 500",
    "TLT": "Long-term Treasuries",
    "GLD": "Gold",
}


# ── Windows ──────────────────────────────────────────────────────────────────

WINDOWS = [63, 252, 504, 1008, 2016, 4032, 4536]
WINDOW_LABELS = {
    63: "63d  (~3 months) — Short-term",
    252: "252d (~1 year) — Core Signal",
    504: "504d (~2 years) — Medium-term",
    1008: "1008d (~4 years) — Structural",
    2016: "2016d (~8 years) — Secular",
    4032: "4032d (~16 years) — Long-term",
    4536: "4536d (~18 years) — Full History",
}
PRIMARY_WINDOW = 252


# ── E-Process Parameters ────────────────────────────────────────────────────

E_PROCESS = {
    "threshold": 20.0,           # E-process threshold to reject non-dominance
    "epsilon": 0.01,             # Epsilon for stochastic dominance
    "test_type": "first_order",  # first_order or second_order
    "n_bootstrap": 1000,         # Bootstrap samples for null distribution
}


# ── Macro Signals ────────────────────────────────────────────────────────────

MACRO_SIGNALS = [
    ("VIX",       "VIX",           0.30, -1.0),
    ("T10Y2Y",    "10Y–2Y Spread", 0.25, +1.0),
    ("DXY",       "DXY",           0.20, -1.0),
    ("IG_SPREAD", "IG Spread",     0.15, -1.0),
    ("HY_SPREAD", "HY Spread",     0.10, -1.0),
]

MACRO_COLS_CORE = ["VIX", "T10Y2Y", "DXY"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]
