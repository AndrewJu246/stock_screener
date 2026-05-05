"""
Generate realistic sample data for dashboard testing.
This simulates a completed scan with real-looking results.
"""

import json
import random
import os

random.seed(42)

SAMPLE_STOCKS = [
    {"ticker": "NVDA", "sector": "Technology", "industry": "Semiconductors", "market_cap": 2800000000000, "price": 875.50},
    {"ticker": "AVGO", "sector": "Technology", "industry": "Semiconductors", "market_cap": 780000000000, "price": 1680.20},
    {"ticker": "CRWD", "sector": "Technology", "industry": "Cybersecurity", "market_cap": 85000000000, "price": 365.40},
    {"ticker": "PLTR", "sector": "Technology", "industry": "Software - Infrastructure", "market_cap": 145000000000, "price": 62.30},
    {"ticker": "ANET", "sector": "Technology", "industry": "Computer Networking", "market_cap": 120000000000, "price": 390.15},
    {"ticker": "VST", "sector": "Utilities", "industry": "Independent Power Producers", "market_cap": 48000000000, "price": 132.80},
    {"ticker": "CEG", "sector": "Utilities", "industry": "Independent Power Producers", "market_cap": 78000000000, "price": 248.50},
    {"ticker": "LLY", "sector": "Healthcare", "industry": "Drug Manufacturers", "market_cap": 720000000000, "price": 780.20},
    {"ticker": "AXON", "sector": "Industrials", "industry": "Aerospace & Defense", "market_cap": 42000000000, "price": 560.30},
    {"ticker": "FTNT", "sector": "Technology", "industry": "Cybersecurity", "market_cap": 75000000000, "price": 98.40},
    {"ticker": "APP", "sector": "Technology", "industry": "Software - Application", "market_cap": 110000000000, "price": 320.10},
    {"ticker": "TOST", "sector": "Technology", "industry": "Software - Application", "market_cap": 19000000000, "price": 34.20},
    {"ticker": "DUOL", "sector": "Technology", "industry": "Software - Application", "market_cap": 14000000000, "price": 330.50},
    {"ticker": "ELF", "sector": "Consumer Defensive", "industry": "Household Products", "market_cap": 7200000000, "price": 128.40},
    {"ticker": "UBER", "sector": "Technology", "industry": "Software - Application", "market_cap": 165000000000, "price": 78.90},
    {"ticker": "GEV", "sector": "Industrials", "industry": "Specialty Industrial Machinery", "market_cap": 85000000000, "price": 310.40},
    {"ticker": "SMCI", "sector": "Technology", "industry": "Computer Hardware", "market_cap": 20000000000, "price": 35.60},
    {"ticker": "ARM", "sector": "Technology", "industry": "Semiconductors", "market_cap": 155000000000, "price": 148.70},
    {"ticker": "MELI", "sector": "Consumer Cyclical", "industry": "Internet Retail", "market_cap": 95000000000, "price": 1880.30},
    {"ticker": "NOW", "sector": "Technology", "industry": "Software - Infrastructure", "market_cap": 195000000000, "price": 940.60},
    {"ticker": "PANW", "sector": "Technology", "industry": "Cybersecurity", "market_cap": 125000000000, "price": 380.20},
    {"ticker": "TTD", "sector": "Technology", "industry": "Software - Application", "market_cap": 55000000000, "price": 112.80},
    {"ticker": "DDOG", "sector": "Technology", "industry": "Software - Application", "market_cap": 42000000000, "price": 128.90},
    {"ticker": "HUBS", "sector": "Technology", "industry": "Software - Application", "market_cap": 32000000000, "price": 640.50},
    {"ticker": "NFLX", "sector": "Communication Services", "industry": "Entertainment", "market_cap": 380000000000, "price": 880.30},
    {"ticker": "SPOT", "sector": "Communication Services", "industry": "Internet Content", "market_cap": 80000000000, "price": 395.20},
    {"ticker": "COIN", "sector": "Financial Services", "industry": "Financial Data & Exchanges", "market_cap": 52000000000, "price": 215.40},
    {"ticker": "MSTR", "sector": "Technology", "industry": "Software - Application", "market_cap": 78000000000, "price": 380.10},
    {"ticker": "RKLB", "sector": "Industrials", "industry": "Aerospace & Defense", "market_cap": 12000000000, "price": 24.80},
    {"ticker": "IONQ", "sector": "Technology", "industry": "Computer Hardware", "market_cap": 8500000000, "price": 38.20},
]

SIGNAL_NAMES = [
    "volume_anomaly", "momentum", "fundamentals", "smart_money",
    "sector_rotation", "earnings_quality", "analyst_revisions",
    "tech_megatrend", "relative_strength"
]

def gen_signal(name, bias=0):
    """Generate a realistic signal score with optional bias."""
    base = random.gauss(50 + bias, 18)
    score = max(0, min(100, round(base, 1)))
    details = {
        "volume_anomaly": f"{'Accumulation' if score > 60 else 'Normal volume'} ({score/30:.1f}x avg vol)",
        "momentum": f"{'+' if score > 50 else ''}{(score-50)*0.4:.1f}% 1w, {'+' if score > 50 else ''}{(score-50)*0.6:.1f}% 1m, {'+' if score > 50 else ''}{(score-50)*0.8:.1f}% 3m",
        "fundamentals": f"Revenue={'accelerating' if score > 65 else 'steady' if score > 40 else 'slowing'}, margin={'expanding' if score > 55 else 'flat'}",
        "smart_money": f"{'Insider buying cluster' if score > 70 else 'Moderate insider activity' if score > 45 else 'No recent insider buys'}",
        "sector_rotation": f"Sector momentum: {'+' if score > 50 else ''}{(score-50)*0.3:.1f}%",
        "earnings_quality": f"CF/Earnings ratio: {score/100*1.2:.2f}",
        "analyst_revisions": f"Target ${random.randint(80,500)} ({'+' if score > 50 else ''}{(score-50)*0.4:.0f}% upside)",
        "tech_megatrend": f"{'AI/ML' if score > 60 else 'Clean Energy' if score > 40 else 'No trend'} exposure",
        "relative_strength": f"Stock {'+' if score > 50 else ''}{(score-50)*0.3:.1f}% vs sector median",
    }
    return {"score": score, "detail": details.get(name, "")}


def generate_sample_results():
    candidates = []

    for i, stock in enumerate(SAMPLE_STOCKS):
        # Top stocks get higher bias
        bias = max(0, 20 - i * 1.2) + random.gauss(0, 5)

        signals = {name: gen_signal(name, bias) for name in SIGNAL_NAMES}

        # Compute composite (simplified)
        weights = {"volume_anomaly": 12, "momentum": 14, "fundamentals": 16,
                    "smart_money": 14, "sector_rotation": 10, "earnings_quality": 12,
                    "analyst_revisions": 7, "tech_megatrend": 9, "relative_strength": 6}
        composite = sum(signals[s]["score"] * weights[s] / 100 for s in SIGNAL_NAMES)

        fund = signals["fundamentals"]["score"]
        mom = signals["momentum"]["score"]

        if fund > 55 and fund > mom:
            strategy = "long_term_hold"
            hold = "3-12 months"
        else:
            strategy = "short_term_momentum"
            hold = "2-8 weeks"

        high_sigs = sum(1 for s in signals.values() if s["score"] >= 60)
        confidence = "high" if high_sigs >= 4 else "medium" if high_sigs >= 2 else "low"

        candidates.append({
            "rank": i + 1,
            "ticker": stock["ticker"],
            "sector": stock["sector"],
            "industry": stock["industry"],
            "market_cap": stock["market_cap"],
            "current_price": stock["price"],
            "composite_score": round(composite, 2),
            "balanced_score": round(composite * random.uniform(0.9, 1.1), 2),
            "regime": "bull",
            "signals": signals,
            "strategy": {
                "strategy": strategy,
                "long_term_score": round(fund * 0.6 + signals["earnings_quality"]["score"] * 0.4, 1),
                "short_term_score": round(mom * 0.5 + signals["volume_anomaly"]["score"] * 0.3 + signals["relative_strength"]["score"] * 0.2, 1),
                "reasoning": f"{'Strong fundamentals with growth trajectory' if strategy == 'long_term_hold' else 'Technical breakout with volume confirmation'}",
                "expected_hold": hold,
            },
            "confidence": {
                "confidence": confidence,
                "checks": {
                    "time_confirmation": {"passed": random.random() > 0.3, "detail": f"Signal active for {random.randint(1,8)} days"},
                    "strong_agreement": {"passed": high_sigs >= 3, "detail": f"{high_sigs} signals ≥60"},
                    "backtest": {"passed": True, "detail": "Historical win rate: 62%"},
                },
            },
            "quality_gate": {
                "passed": True,
                "checks_passed": 7,
                "checks_total": 7,
            },
            "position_sizing": {
                "recommended_allocation": round(10000 / 8 * (1.0 if confidence == "high" else 0.7), 2),
                "recommended_shares": max(1, int(10000 / 8 / stock["price"])),
                "position_pct_of_portfolio": round(100 / 8, 1),
            },
            "volatility_pct": round(random.uniform(0.015, 0.055), 4),
        })

    # Sort by composite score
    candidates.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, c in enumerate(candidates):
        c["rank"] = i + 1

    output = {
        "summary": {
            "scan_date": "2026-04-28T09:30:00",
            "regime": "bull",
            "regime_description": "Strong uptrend — momentum and sector flows dominate",
            "market_data": {
                "sp500_price": 5842.5,
                "sp500_ma50": 5680.2,
                "sp500_ma200": 5320.8,
                "sp500_above_50dma": True,
                "sp500_above_200dma": True,
                "vix": 14.8,
            },
            "sector_etf_momentum": {
                "ai_ml": 18.5,
                "semiconductors": 14.2,
                "cybersecurity": 11.8,
                "clean_energy": 5.3,
                "biotech": -2.1,
                "space_defense": 8.9,
            },
            "universe_size": 1500,
            "stocks_scored": 1187,
            "gate_rejected": 412,
            "skipped": 313,
            "candidates_returned": 30,
            "elapsed_seconds": 185.3,
            "capital": 10000,
            "strategy_breakdown": {
                "long_term_hold": sum(1 for c in candidates if c["strategy"]["strategy"] == "long_term_hold"),
                "short_term_momentum": sum(1 for c in candidates if c["strategy"]["strategy"] == "short_term_momentum"),
            },
        },
        "candidates": candidates,
    }

    os.makedirs("output", exist_ok=True)
    with open("output/screener_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Generated {len(candidates)} sample candidates")
    print(f"Saved to output/screener_results.json")
    return output


if __name__ == "__main__":
    generate_sample_results()
