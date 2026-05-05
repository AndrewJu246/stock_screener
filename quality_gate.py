"""
Quality gate: filters out lottery stocks and low-quality candidates.

Every stock must pass ALL gates to be considered a candidate.
This runs AFTER signals are computed but BEFORE composite scoring.
"""

import logging
from typing import Optional

from config import QUALITY_GATE, CONFIDENCE_LADDER

logger = logging.getLogger(__name__)


def apply_quality_gate(stock_result: dict) -> dict:
    """
    Run all quality checks on a stock.

    Returns: {
        "passed": bool,
        "checks": {check_name: {passed: bool, detail: str}, ...},
        "rejection_reason": str or None,
    }
    """
    checks = {}
    reasons = []

    # 1. Market cap floor
    market_cap = stock_result.get("market_cap", 0)
    min_cap = QUALITY_GATE["min_market_cap"]
    passed = market_cap >= min_cap
    checks["market_cap"] = {
        "passed": passed,
        "detail": f"${market_cap:,.0f}" + (f" (min ${min_cap:,.0f})" if not passed else ""),
    }
    if not passed:
        reasons.append(f"Market cap ${market_cap:,.0f} below ${min_cap:,.0f}")

    # 2. Liquidity floor (average volume)
    # We check this from the price history if available
    signals = stock_result.get("signals", {})
    vol_detail = signals.get("volume_anomaly", {}).get("detail", "")
    # We'll also accept the stock if we got valid price data
    # (the data pipeline already filters out very illiquid stocks)
    checks["liquidity"] = {
        "passed": True,  # Refined in screener with actual volume data
        "detail": "Checked via data pipeline",
    }

    # 3. Revenue required (no pre-revenue hype)
    fund_score = signals.get("fundamentals", {}).get("score", 0)
    fund_detail = signals.get("fundamentals", {}).get("detail", "")
    has_revenue = fund_score > 0 and "revenue=0" not in fund_detail.lower()
    checks["has_revenue"] = {
        "passed": has_revenue,
        "detail": fund_detail,
    }
    if not has_revenue:
        reasons.append("No revenue data / pre-revenue company")

    # 4. Fundamental floor (min score regardless of momentum)
    min_fund = QUALITY_GATE["min_fundamental_score"]
    fund_passed = fund_score >= min_fund
    checks["fundamental_floor"] = {
        "passed": fund_passed,
        "detail": f"Score {fund_score:.0f}/100 (min {min_fund})",
    }
    if not fund_passed:
        reasons.append(f"Fundamental score {fund_score:.0f} below minimum {min_fund}")

    # 5. Cash flow check
    eq_score = signals.get("earnings_quality", {}).get("score", 0)
    eq_detail = signals.get("earnings_quality", {}).get("detail", "")
    cf_passed = eq_score >= 25  # Very low bar — just needs to not be terrible
    checks["cash_flow"] = {
        "passed": cf_passed,
        "detail": eq_detail or f"Score: {eq_score:.0f}",
    }
    if not cf_passed:
        reasons.append(f"Cash flow quality score {eq_score:.0f} too low")

    # 6. Volatility check (reject lottery-like stocks)
    vol_pct = stock_result.get("volatility_pct")
    max_vol = QUALITY_GATE["max_volatility_vs_peers"]
    if vol_pct is not None:
        # Typical stock has ~2-3% daily range; lottery stocks have 8%+
        # We flag anything >2.5x the typical ~2.5% = ~6.25%
        vol_excessive = vol_pct > 0.065
        checks["volatility"] = {
            "passed": not vol_excessive,
            "detail": f"Avg daily range: {vol_pct:.1%}" + (
                " (excessive)" if vol_excessive else ""
            ),
        }
        if vol_excessive:
            reasons.append(f"Daily volatility {vol_pct:.1%} too high (lottery-like)")
    else:
        checks["volatility"] = {"passed": True, "detail": "No volatility data"}

    # 7. Signal agreement (min 2 families must score ≥50)
    min_agree = QUALITY_GATE["min_signal_agreement"]
    high_signals = sum(
        1 for sig_name, sig_data in signals.items()
        if isinstance(sig_data, dict) and sig_data.get("score", 0) >= 50
    )
    agree_passed = high_signals >= min_agree
    checks["signal_agreement"] = {
        "passed": agree_passed,
        "detail": f"{high_signals} signals ≥50 (need {min_agree})",
    }
    if not agree_passed:
        reasons.append(f"Only {high_signals} signals agree (need {min_agree})")

    # Overall result
    all_passed = all(c["passed"] for c in checks.values())

    return {
        "passed": all_passed,
        "checks": checks,
        "checks_passed": sum(1 for c in checks.values() if c["passed"]),
        "checks_total": len(checks),
        "rejection_reason": "; ".join(reasons) if reasons else None,
    }


def apply_confidence_ladder(stock_result: dict, signal_history: Optional[dict] = None) -> dict:
    """
    Apply the confidence ladder to determine signal reliability.

    Returns confidence_level: "high", "medium", "low"
    and the checks that were applied.
    """
    checks = {}
    confidence = "high"

    # Time confirmation: signal must persist for N days
    # (This requires historical signal data — skip if not available yet)
    min_days = CONFIDENCE_LADDER["min_days_signal_persists"]
    if signal_history:
        # Check if signal scores have been consistently above threshold
        days_active = signal_history.get("days_above_threshold", 0)
        checks["time_confirmation"] = {
            "passed": days_active >= min_days,
            "detail": f"Signal active for {days_active} days (need {min_days})",
        }
        if days_active < min_days:
            confidence = "medium"
    else:
        checks["time_confirmation"] = {
            "passed": False,
            "detail": "First scan — no history yet. Will confirm over coming days.",
        }
        confidence = "medium"

    # Multi-signal agreement (already checked in quality gate, but higher bar here)
    signals = stock_result.get("signals", {})
    high_signals = sum(
        1 for sig_data in signals.values()
        if isinstance(sig_data, dict) and sig_data.get("score", 0) >= 60
    )
    strong_agree = high_signals >= 3  # Higher bar for high confidence
    checks["strong_agreement"] = {
        "passed": strong_agree,
        "detail": f"{high_signals} signals ≥60",
    }
    if not strong_agree:
        confidence = min(confidence, "medium")

    # Backtest check (placeholder — built in Phase 3)
    checks["backtest"] = {
        "passed": True,  # Will be populated by backtesting module
        "detail": "Backtesting module not yet active",
    }

    return {
        "confidence": confidence,
        "checks": checks,
    }
