"""
Composite scorer: combines signal scores with regime-aware weights,
detects market regime, and classifies candidates into strategy buckets.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from config import REGIMES, POSITION_SIZING

logger = logging.getLogger(__name__)

WEIGHT_OVERRIDES_FILE = Path("weight_overrides.json")


def _get_weights(regime: str) -> dict:
    """
    Get signal weights for a regime, applying optimizer overrides if available.
    """
    base_weights = REGIMES[regime]["weights"]

    if WEIGHT_OVERRIDES_FILE.exists():
        try:
            with open(WEIGHT_OVERRIDES_FILE) as f:
                overrides = json.load(f)
            if regime in overrides and isinstance(overrides[regime], dict):
                logger.info(f"Using optimized weights for '{regime}' regime")
                return overrides[regime]
        except Exception as e:
            logger.debug(f"Could not load weight overrides: {e}")

    return base_weights


def detect_regime(market_data: dict) -> str:
    """
    Detect the current market regime based on S&P 500 trend and VIX.

    Returns: "bull", "bear", "early_recovery", or "balanced"
    """
    if not market_data or "regime" in market_data:
        return market_data.get("regime", "balanced")

    above_200 = market_data.get("sp500_above_200dma", True)
    above_50 = market_data.get("sp500_above_50dma", True)
    vix = market_data.get("vix", 20)

    # Bull: above both MAs, low VIX
    if above_200 and above_50 and vix < 20:
        return "bull"

    # Bear: below both MAs or high VIX
    if not above_200 and not above_50:
        return "bear"
    if vix > 28:
        return "bear"

    # Early recovery: crossed above 50 DMA but still below 200
    if not above_200 and above_50:
        return "early_recovery"

    # Default
    return "balanced"


def compute_composite_score(
    stock_signals: dict,
    regime: str,
    also_compute_balanced: bool = True,
) -> dict:
    """
    Compute the weighted composite score using regime-specific weights,
    adjusted by market cap tier.
    """
    signals = stock_signals.get("signals", {})
    market_cap = stock_signals.get("market_cap", 0)
    weights = _get_weights(regime)

    # Apply market cap tier adjustments
    weights = _adjust_weights_for_market_cap(weights, market_cap)

    # Compute weighted score
    composite = 0
    contributions = {}
    for signal_name, weight in weights.items():
        sig = signals.get(signal_name, {})
        score = sig.get("score", 0) if isinstance(sig, dict) else 0
        contribution = score * (weight / 100)
        composite += contribution
        contributions[signal_name] = {
            "raw_score": score,
            "weight": weight,
            "contribution": round(contribution, 2),
        }

    tier = _get_market_cap_tier(market_cap)

    result = {
        "composite_score": round(composite, 2),
        "regime": regime,
        "regime_description": REGIMES[regime]["description"],
        "regime_weights": weights,
        "signal_contributions": contributions,
        "market_cap_tier": tier,
    }

    # Also compute balanced score for comparison
    if also_compute_balanced and regime != "balanced":
        balanced_weights = REGIMES["balanced"]["weights"]
        balanced_score = 0
        for signal_name, weight in balanced_weights.items():
            sig = signals.get(signal_name, {})
            score = sig.get("score", 0) if isinstance(sig, dict) else 0
            balanced_score += score * (weight / 100)
        result["balanced_score"] = round(balanced_score, 2)
    else:
        result["balanced_score"] = result["composite_score"]

    return result


def _get_market_cap_tier(market_cap: float) -> str:
    if market_cap >= 50_000_000_000:
        return "large_cap"
    elif market_cap >= 2_000_000_000:
        return "mid_cap"
    elif market_cap > 0:
        return "small_cap"
    return "unknown"


def _adjust_weights_for_market_cap(base_weights: dict, market_cap: float) -> dict:
    """
    Adjust signal weights based on market cap tier.

    Rationale from backtest results:
    - Large caps ($50B+): Momentum is anti-predictive. Lean on fundamentals
      and earnings quality. These stocks are efficiently priced — only
      genuine fundamental shifts move them.
    - Mid caps ($2B-$50B): Sweet spot. Volume + momentum work well.
      Balanced approach with slight volume boost.
    - Small caps ($300M-$2B): Volume anomaly is strongest here because
      institutional accumulation is hardest to hide in thinly-traded stocks.
    """
    tier = _get_market_cap_tier(market_cap)
    weights = dict(base_weights)  # Copy

    if tier == "large_cap":
        # Shift from momentum → fundamentals + earnings quality
        weights["momentum"] = weights.get("momentum", 13) * 0.6
        weights["fundamentals"] = weights.get("fundamentals", 15) * 1.3
        weights["earnings_quality"] = weights.get("earnings_quality", 11) * 1.3
        weights["analyst_revisions"] = weights.get("analyst_revisions", 5) * 1.2

    elif tier == "small_cap":
        # Maximize volume anomaly — institutional accumulation is most
        # visible in small caps with lower daily volume
        weights["volume_anomaly"] = weights.get("volume_anomaly", 20) * 1.3
        weights["momentum"] = weights.get("momentum", 13) * 1.1
        weights["fundamentals"] = weights.get("fundamentals", 15) * 0.85

    # mid_cap and unknown: use base weights as-is

    # Re-normalize to sum to 100
    total = sum(weights.values())
    if total > 0:
        weights = {k: round(v / total * 100, 1) for k, v in weights.items()}

    return weights


def classify_strategy(stock_result: dict, scored_result: dict) -> dict:
    """
    Classify a candidate into a strategy bucket:
      - "long_term_hold": Strong fundamentals, early growth stage
      - "short_term_momentum": Technical breakout, ride the wave

    Also determines entry timing and position parameters.
    """
    signals = stock_result.get("signals", {})

    # Extract key scores
    fund_score = signals.get("fundamentals", {}).get("score", 0)
    mom_score = signals.get("momentum", {}).get("score", 0)
    vol_score = signals.get("volume_anomaly", {}).get("score", 0)
    eq_score = signals.get("earnings_quality", {}).get("score", 0)
    smart_score = signals.get("smart_money", {}).get("score", 0)
    tech_score = signals.get("tech_megatrend", {}).get("score", 0)
    analyst_score = signals.get("analyst_revisions", {}).get("score", 0)

    # Long-term hold criteria:
    # Strong fundamentals + earnings quality + megatrend exposure
    long_term_score = (
        fund_score * 0.30
        + eq_score * 0.25
        + tech_score * 0.20
        + smart_score * 0.15
        + analyst_score * 0.10
    )

    # Short-term momentum criteria:
    # Strong momentum + volume + relative strength
    rs_score = signals.get("relative_strength", {}).get("score", 0)
    short_term_score = (
        mom_score * 0.35
        + vol_score * 0.25
        + rs_score * 0.20
        + signals.get("sector_rotation", {}).get("score", 0) * 0.20
    )

    # Classification
    if long_term_score > short_term_score and fund_score >= 55:
        strategy = "long_term_hold"
        reasoning = (
            f"Strong fundamentals ({fund_score:.0f}) and earnings quality "
            f"({eq_score:.0f}) suggest sustained growth potential."
        )
        expected_hold = "3-12 months"
    elif short_term_score > 60 and mom_score >= 60:
        strategy = "short_term_momentum"
        reasoning = (
            f"Strong momentum ({mom_score:.0f}) with volume confirmation "
            f"({vol_score:.0f}) suggests a technical breakout."
        )
        expected_hold = "2-8 weeks"
    elif long_term_score > 50:
        strategy = "long_term_hold"
        reasoning = f"Moderate fundamental case ({fund_score:.0f}) with growth indicators."
        expected_hold = "3-12 months"
    else:
        strategy = "short_term_momentum"
        reasoning = f"Momentum-driven opportunity ({mom_score:.0f})."
        expected_hold = "2-8 weeks"

    return {
        "strategy": strategy,
        "long_term_score": round(long_term_score, 1),
        "short_term_score": round(short_term_score, 1),
        "reasoning": reasoning,
        "expected_hold": expected_hold,
    }


def compute_position_size(
    composite_score: float,
    confidence: str,
    total_capital: float,
    current_positions: int,
    stock_price: float,
) -> dict:
    """
    Calculate the recommended position size based on capital,
    confidence level, and the ranked position in the candidate list.
    """
    params = POSITION_SIZING

    # Determine position count range from capital tier
    max_pos = 10
    min_pos = 5
    for (lo, hi), (mn, mx) in params["tiers"].items():
        if lo <= total_capital < hi:
            min_pos, max_pos = mn, mx
            break

    # Base allocation = capital / max positions
    base_allocation = total_capital / max_pos

    # Adjust by confidence
    confidence_mult = {"high": 1.0, "medium": 0.7, "low": 0.4}
    allocation = base_allocation * confidence_mult.get(confidence, 0.7)

    # Cap at max single position
    max_single = total_capital * params["max_single_position_pct"]
    allocation = min(allocation, max_single)

    # Compute shares
    if stock_price > 0:
        shares = int(allocation / stock_price)
    else:
        shares = 0

    return {
        "recommended_allocation": round(allocation, 2),
        "recommended_shares": shares,
        "position_pct_of_portfolio": round(allocation / total_capital * 100, 1) if total_capital > 0 else 0,
        "max_positions_for_capital": max_pos,
        "confidence_multiplier": confidence_mult.get(confidence, 0.7),
    }
