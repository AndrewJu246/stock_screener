"""
Short interest tracking.

Adds short interest as a signal modifier:
  - High short interest + strong fundamentals + insider buying = squeeze potential
  - High short interest + weak fundamentals = smart money is bearish, avoid
  - Very high short interest (>30%) = extreme risk either way

Data source: yfinance stock.info (shortRatio, shortPercentOfFloat).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache")

# Thresholds
SHORT_INTEREST_HIGH = 0.15     # 15% of float shorted = elevated
SHORT_INTEREST_VERY_HIGH = 0.30  # 30% = extreme
SHORT_RATIO_HIGH = 5.0          # 5+ days to cover = elevated


def get_short_interest(ticker: str) -> Optional[dict]:
    """
    Get short interest data for a ticker.

    Returns: {
        "short_pct_float": float (0-1),
        "short_ratio": float (days to cover),
        "shares_short": int,
    }
    """
    cache_file = CACHE_DIR / f"short_{ticker}.json"

    if cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age = (datetime.now() - mtime).total_seconds() / 3600
        if age < 48:  # Cache for 2 days (short interest updates bi-monthly)
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception:
                pass

    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        result = {
            "short_pct_float": info.get("shortPercentOfFloat", 0) or 0,
            "short_ratio": info.get("shortRatio", 0) or 0,
            "shares_short": info.get("sharesShort", 0) or 0,
            "shares_short_prior": info.get("sharesShortPriorMonth", 0) or 0,
            "float_shares": info.get("floatShares", 0) or 0,
        }

        # Calculate short interest change
        if result["shares_short_prior"] and result["shares_short_prior"] > 0:
            result["short_change_pct"] = (
                (result["shares_short"] - result["shares_short_prior"])
                / result["shares_short_prior"]
            )
        else:
            result["short_change_pct"] = 0

        with open(cache_file, "w") as f:
            json.dump(result, f)

        return result

    except Exception as e:
        logger.debug(f"Failed to get short interest for {ticker}: {e}")
        return None


def assess_short_interest(
    ticker: str,
    fundamental_score: float = 50,
    insider_score: float = 50,
    short_data: Optional[dict] = None,
) -> dict:
    """
    Assess short interest and its implications for the stock.

    Cross-references with fundamentals and insider signals to determine
    whether high short interest is a squeeze opportunity or a warning.

    Returns: {
        "signal": "squeeze_potential" | "warning" | "extreme_risk" | "normal",
        "score_modifier": float (-15 to +15),
        "short_pct": float,
        "detail": str,
    }
    """
    if short_data is None:
        short_data = get_short_interest(ticker)

    if not short_data:
        return {
            "signal": "unknown",
            "score_modifier": 0,
            "short_pct": 0,
            "detail": "Short interest data unavailable",
        }

    short_pct = short_data.get("short_pct_float", 0)
    short_ratio = short_data.get("short_ratio", 0)
    short_change = short_data.get("short_change_pct", 0)

    # Normal short interest
    if short_pct < SHORT_INTEREST_HIGH and short_ratio < SHORT_RATIO_HIGH:
        return {
            "signal": "normal",
            "score_modifier": 0,
            "short_pct": round(short_pct * 100, 1),
            "short_ratio": round(short_ratio, 1),
            "detail": f"Short interest: {short_pct:.1%} of float, {short_ratio:.1f} days to cover",
        }

    # Very high short interest — extreme risk
    if short_pct >= SHORT_INTEREST_VERY_HIGH:
        return {
            "signal": "extreme_risk",
            "score_modifier": -10,
            "short_pct": round(short_pct * 100, 1),
            "short_ratio": round(short_ratio, 1),
            "detail": (
                f"⚠ Extreme short interest: {short_pct:.1%} of float — "
                f"high risk regardless of direction"
            ),
        }

    # High short interest — context matters
    if short_pct >= SHORT_INTEREST_HIGH:
        # Cross-reference with other signals
        if fundamental_score >= 60 and insider_score >= 50:
            # Strong fundamentals + insider confidence + high short = squeeze setup
            modifier = min(15, short_pct * 50)  # Up to +15 boost

            # Extra boost if shorts are increasing (more fuel for squeeze)
            if short_change > 0.05:
                modifier += 5
                change_note = f", shorts increasing ({short_change:+.1%})"
            else:
                change_note = ""

            return {
                "signal": "squeeze_potential",
                "score_modifier": round(min(modifier, 15), 1),
                "short_pct": round(short_pct * 100, 1),
                "short_ratio": round(short_ratio, 1),
                "detail": (
                    f"Squeeze potential: {short_pct:.1%} shorted + strong fundamentals "
                    f"+ insider confidence{change_note}"
                ),
            }
        else:
            # Weak fundamentals + high short = smart money is bearish
            return {
                "signal": "warning",
                "score_modifier": -10,
                "short_pct": round(short_pct * 100, 1),
                "short_ratio": round(short_ratio, 1),
                "detail": (
                    f"⚠ High short interest ({short_pct:.1%}) with weak fundamentals — "
                    f"smart money may be bearish"
                ),
            }

    # Elevated but not high
    if short_ratio >= SHORT_RATIO_HIGH:
        return {
            "signal": "elevated",
            "score_modifier": 0,
            "short_pct": round(short_pct * 100, 1),
            "short_ratio": round(short_ratio, 1),
            "detail": f"Elevated days-to-cover: {short_ratio:.1f} days ({short_pct:.1%} of float)",
        }

    return {
        "signal": "normal",
        "score_modifier": 0,
        "short_pct": round(short_pct * 100, 1),
        "detail": f"Short interest: {short_pct:.1%}",
    }
