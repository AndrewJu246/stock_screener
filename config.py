"""
Configuration for the stock screener.
All thresholds, weights, and parameters in one place.
"""

# ─── Universe ────────────────────────────────────────────────────────
# Start with S&P 500 + MidCap 400 + SmallCap 600 ≈ 1500 stocks.
# We fetch the actual tickers at runtime from Wikipedia/yfinance.
# Set to "full" to scan all ~8000 US-listed stocks.
UNIVERSE_MODE = "sp1500"  # "sp1500" or "full"

# ─── Quality gate thresholds ─────────────────────────────────────────
QUALITY_GATE = {
    "min_market_cap": 300_000_000,       # $300M minimum
    "min_avg_volume": 100_000,           # 100K shares/day average
    "min_revenue": 1_000_000,            # Must have >$1M revenue (not pre-revenue)
    "min_fundamental_score": 25,         # Fundamental signal must be ≥25/100
    "min_signal_agreement": 2,           # At least 2 signal families must score ≥50
    "max_volatility_vs_peers": 2.5,      # Daily range can't be >2.5x sector median
    "min_cash_flow_trend": -0.10,        # Cash flow can decline at most 10% YoY
}

# ─── Regime definitions ──────────────────────────────────────────────
# The model auto-detects which regime we're in based on market indicators.
# Each regime defines signal weights (must sum to 100).
REGIMES = {
    "bull": {
        "description": "Strong uptrend — momentum and sector flows dominate",
        "weights": {
            "volume_anomaly":    10,
            "momentum":          20,
            "fundamentals":      10,
            "smart_money":       10,
            "sector_rotation":   18,
            "earnings_quality":   5,
            "analyst_revisions":  8,
            "tech_megatrend":    12,
            "relative_strength":  7,
        },
        # Market conditions that trigger this regime:
        "conditions": {
            "sp500_above_200dma": True,
            "sp500_above_50dma": True,
            "vix_below": 20,
        },
    },
    "bear": {
        "description": "Downturn / uncertainty — fundamentals and smart money dominate",
        "weights": {
            "volume_anomaly":     8,
            "momentum":           5,
            "fundamentals":      25,
            "smart_money":       20,
            "sector_rotation":    5,
            "earnings_quality":  20,
            "analyst_revisions":  5,
            "tech_megatrend":     5,
            "relative_strength":  7,
        },
        "conditions": {
            "sp500_above_200dma": False,
            "sp500_above_50dma": False,
            "vix_above": 25,
        },
    },
    "early_recovery": {
        "description": "Balanced — solid fundamentals with emerging momentum",
        "weights": {
            "volume_anomaly":    15,
            "momentum":          12,
            "fundamentals":      18,
            "smart_money":       15,
            "sector_rotation":   10,
            "earnings_quality":  10,
            "analyst_revisions":  8,
            "tech_megatrend":     8,
            "relative_strength":  4,
        },
        "conditions": {
            "sp500_above_200dma": False,
            "sp500_above_50dma": True,
            "vix_below": 25,
        },
    },
    "balanced": {
        "description": "Default / benchmark — even weighting across signal families",
        "weights": {
            "volume_anomaly":    12,
            "momentum":          14,
            "fundamentals":      16,
            "smart_money":       14,
            "sector_rotation":   10,
            "earnings_quality":  12,
            "analyst_revisions":  7,
            "tech_megatrend":     9,
            "relative_strength":  6,
        },
        "conditions": {},  # Fallback — always available
    },
}

# ─── Megatrend definitions ───────────────────────────────────────────
# Map sectors/industries to megatrends for the tech_megatrend signal.
MEGATRENDS = {
    "ai_ml": {
        "name": "Artificial intelligence / Machine learning",
        "keywords": ["artificial intelligence", "machine learning", "neural network",
                     "deep learning", "GPU", "data center", "cloud computing"],
        "etf_proxies": ["BOTZ", "AIQ", "ROBT"],
        "sectors": ["Technology", "Communication Services"],
    },
    "clean_energy": {
        "name": "Clean energy transition",
        "keywords": ["solar", "wind", "battery", "EV", "electric vehicle",
                     "renewable", "hydrogen", "grid"],
        "etf_proxies": ["ICLN", "TAN", "QCLN"],
        "sectors": ["Utilities", "Industrials", "Energy"],
    },
    "biotech": {
        "name": "Biotech / Genomics",
        "keywords": ["biotech", "genomics", "CRISPR", "gene therapy",
                     "pharmaceutical", "drug discovery", "mRNA"],
        "etf_proxies": ["XBI", "IBB", "ARKG"],
        "sectors": ["Healthcare"],
    },
    "cybersecurity": {
        "name": "Cybersecurity",
        "keywords": ["cybersecurity", "zero trust", "cloud security",
                     "identity", "threat detection"],
        "etf_proxies": ["CIBR", "HACK", "BUG"],
        "sectors": ["Technology"],
    },
    "space_defense": {
        "name": "Space / Defense tech",
        "keywords": ["satellite", "space", "defense", "aerospace",
                     "hypersonic", "drone", "autonomous"],
        "etf_proxies": ["ITA", "UFO", "ARKX"],
        "sectors": ["Industrials", "Technology"],
    },
    "semiconductors": {
        "name": "Semiconductor supply chain",
        "keywords": ["semiconductor", "chip", "fab", "lithography",
                     "wafer", "foundry", "ASIC"],
        "etf_proxies": ["SMH", "SOXX", "PSI"],
        "sectors": ["Technology"],
    },
}

# ─── Signal parameters ───────────────────────────────────────────────
SIGNAL_PARAMS = {
    "volume_anomaly": {
        "lookback_days": 50,
        "spike_threshold": 1.5,      # Volume must be 1.5x average to register
        "strong_spike": 2.5,         # 2.5x average = strong signal
    },
    "momentum": {
        "windows": [5, 21, 63, 126], # 1 week, 1 month, 3 months, 6 months
        "acceleration_bonus": 1.3,    # Bonus if shorter windows > longer windows
    },
    "fundamentals": {
        "revenue_growth_weight": 0.4,
        "margin_expansion_weight": 0.3,
        "earnings_growth_weight": 0.3,
    },
    "smart_money": {
        "insider_buy_lookback_days": 90,
        "insider_cluster_min": 2,     # Min 2 insiders buying = cluster
        "institutional_13f_weight": 0.4,
        "insider_weight": 0.6,
    },
    "earnings_quality": {
        "cffo_to_net_income_min": 0.7,  # Cash flow ≥ 70% of net income
        "accruals_penalty_threshold": 0.10,
    },
    "volatility_filter": {
        "lookback_days": 30,
        "peer_comparison_window": 90,
    },
}

# ─── Position sizing ─────────────────────────────────────────────────
POSITION_SIZING = {
    "max_single_position_pct": 0.20,   # Never >20% in one stock
    "max_risk_per_trade_pct": 0.02,    # Max 2% portfolio risk per trade
    "tiers": {
        # capital_range: (min_positions, max_positions)
        (0, 5_000):         (3, 5),
        (5_000, 20_000):    (5, 10),
        (20_000, 100_000):  (8, 15),
        (100_000, float("inf")): (12, 20),
    },
}

# ─── Soft landing exit rules ─────────────────────────────────────────
EXIT_RULES = {
    "stop_loss_tiers": [
        {"pct_loss": -0.07, "sell_fraction": 0.33},   # Sell 1/3 at -7%
        {"pct_loss": -0.12, "sell_fraction": 0.33},   # Sell 1/3 at -12%
        {"pct_loss": -0.18, "sell_fraction": 0.34},   # Sell remaining at -18%
    ],
    "profit_taking": [
        {"pct_gain": 0.20, "sell_fraction": 0.33},    # Take 1/3 at +20%
        # Rest rides with trailing stop
    ],
    "trailing_stop_pct": 0.12,  # 12% trailing stop on remaining position
    "thesis_break_exit": True,   # Exit if the reason for buying breaks
}

# ─── Confidence ladder ───────────────────────────────────────────────
CONFIDENCE_LADDER = {
    "min_days_signal_persists": 3,     # Signal must hold for 3+ days
    "min_signal_families_agree": 2,    # At least 2 signal families score ≥50
    "min_backtest_winrate": 0.55,      # Historical win rate must be ≥55%
    "sector_aware_backtest": True,     # Separate win rates by sector
}

# ─── Output ──────────────────────────────────────────────────────────
OUTPUT = {
    "top_candidates": 50,        # Show top 50 ranked candidates
    "output_dir": "output",
    "results_file": "screener_results.json",
    "history_file": "screener_history.json",
}
