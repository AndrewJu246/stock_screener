"""
Earnings calendar awareness.

Checks when each candidate stock has upcoming earnings and flags
entries that would be gambling on an earnings report.

Three modes:
  - "avoid": Flag stocks reporting within N days. Don't buy pre-earnings.
  - "post_earnings_momentum": Boost stocks that just reported strong
    earnings and are breaking out (the report confirmed the thesis).
  - "both": Apply both rules (default).

Data source: yfinance (free, from Yahoo Finance earnings calendar).
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import json

import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache")
EARNINGS_AVOID_DAYS = 7      # Don't enter within 7 days of earnings
POST_EARNINGS_WINDOW = 14    # Look for momentum within 14 days after earnings


def get_earnings_date(ticker: str) -> Optional[dict]:
    """
    Get the next earnings date for a ticker.

    Returns: {
        "next_earnings": "2026-05-15" or None,
        "days_until_earnings": int or None,
        "last_earnings": "2026-02-10" or None,
        "days_since_last_earnings": int or None,
    }
    """
    cache_file = CACHE_DIR / f"earnings_{ticker}.json"

    # Cache for 24 hours
    if cache_file.exists():
        age = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime))
        if age.total_seconds() < 86400:
            try:
                with open(cache_file) as f:
                    return json.load(f)
            except Exception:
                pass

    try:
        stock = yf.Ticker(ticker)
        cal = stock.calendar

        today = datetime.now().date()
        result = {
            "next_earnings": None,
            "days_until_earnings": None,
            "last_earnings": None,
            "days_since_last_earnings": None,
        }

        # yfinance calendar can return different formats
        if cal is not None:
            # Try to extract earnings date
            earnings_date = None

            if isinstance(cal, dict):
                # Sometimes returns {'Earnings Date': [date1, date2], ...}
                ed = cal.get("Earnings Date")
                if ed:
                    if isinstance(ed, list) and len(ed) > 0:
                        earnings_date = ed[0]
                    else:
                        earnings_date = ed
            elif hasattr(cal, 'iloc'):
                # DataFrame format
                try:
                    if "Earnings Date" in cal.columns:
                        earnings_date = cal["Earnings Date"].iloc[0]
                    elif "Earnings Date" in cal.index:
                        earnings_date = cal.loc["Earnings Date"].iloc[0]
                except Exception:
                    pass

            if earnings_date is not None:
                try:
                    if hasattr(earnings_date, 'date'):
                        ed = earnings_date.date()
                    else:
                        ed = datetime.strptime(str(earnings_date)[:10], "%Y-%m-%d").date()

                    days_diff = (ed - today).days

                    if days_diff >= 0:
                        result["next_earnings"] = str(ed)
                        result["days_until_earnings"] = days_diff
                    else:
                        result["last_earnings"] = str(ed)
                        result["days_since_last_earnings"] = abs(days_diff)
                except Exception:
                    pass

        # Also try earnings_dates for historical dates
        try:
            edates = stock.earnings_dates
            if edates is not None and not edates.empty:
                dates = edates.index.tolist()
                past = [d for d in dates if d.date() <= today]
                future = [d for d in dates if d.date() > today]

                if future and result["next_earnings"] is None:
                    next_ed = min(future).date()
                    result["next_earnings"] = str(next_ed)
                    result["days_until_earnings"] = (next_ed - today).days

                if past and result["last_earnings"] is None:
                    last_ed = max(past).date()
                    result["last_earnings"] = str(last_ed)
                    result["days_since_last_earnings"] = (today - last_ed).days
        except Exception:
            pass

        # Cache result
        with open(cache_file, "w") as f:
            json.dump(result, f)

        return result

    except Exception as e:
        logger.debug(f"Failed to get earnings date for {ticker}: {e}")
        return None


def assess_earnings_risk(ticker: str, earnings_data: Optional[dict] = None) -> dict:
    """
    Assess earnings-related risk for a candidate stock.

    Returns: {
        "risk_level": "high" | "medium" | "low" | "unknown",
        "action": "avoid" | "caution" | "clear" | "post_earnings_momentum",
        "detail": str,
        "next_earnings": str or None,
        "days_until_earnings": int or None,
    }
    """
    if earnings_data is None:
        earnings_data = get_earnings_date(ticker)

    if not earnings_data:
        return {
            "risk_level": "unknown",
            "action": "caution",
            "detail": "Could not determine earnings date",
            "next_earnings": None,
            "days_until_earnings": None,
        }

    days_until = earnings_data.get("days_until_earnings")
    days_since = earnings_data.get("days_since_last_earnings")

    # Check if earnings are imminent
    if days_until is not None:
        if days_until <= 3:
            return {
                "risk_level": "high",
                "action": "avoid",
                "detail": f"⚠ Earnings in {days_until} days — entry is a gamble",
                "next_earnings": earnings_data.get("next_earnings"),
                "days_until_earnings": days_until,
            }
        elif days_until <= EARNINGS_AVOID_DAYS:
            return {
                "risk_level": "medium",
                "action": "caution",
                "detail": f"⚠ Earnings in {days_until} days — consider waiting",
                "next_earnings": earnings_data.get("next_earnings"),
                "days_until_earnings": days_until,
            }
        else:
            return {
                "risk_level": "low",
                "action": "clear",
                "detail": f"Earnings in {days_until} days — safe entry window",
                "next_earnings": earnings_data.get("next_earnings"),
                "days_until_earnings": days_until,
            }

    # Check for post-earnings momentum opportunity
    if days_since is not None and days_since <= POST_EARNINGS_WINDOW:
        return {
            "risk_level": "low",
            "action": "post_earnings_momentum",
            "detail": f"Reported {days_since} days ago — post-earnings momentum window",
            "next_earnings": None,
            "days_until_earnings": None,
            "days_since_last_earnings": days_since,
        }

    # No earnings date found
    return {
        "risk_level": "unknown",
        "action": "caution",
        "detail": "Earnings date not available — check manually",
        "next_earnings": None,
        "days_until_earnings": None,
    }


def apply_earnings_filter(candidates: list[dict]) -> list[dict]:
    """
    Add earnings risk assessment to each candidate.
    Stocks with imminent earnings get flagged but NOT removed —
    the user decides whether to act on the warning.

    Also reorders: post-earnings momentum plays get a small boost,
    pre-earnings gambles get pushed down.
    """
    for candidate in candidates:
        ticker = candidate["ticker"]
        earnings = assess_earnings_risk(ticker)
        candidate["earnings"] = earnings

        # Adjust effective rank based on earnings risk
        if earnings["action"] == "avoid":
            # Don't remove — but add a score penalty so it drops in ranking
            candidate["earnings_adjusted_score"] = candidate.get("composite_score", 0) * 0.85
            logger.info(f"  {ticker}: ⚠ earnings in {earnings.get('days_until_earnings', '?')} days — score penalized")
        elif earnings["action"] == "post_earnings_momentum":
            # Boost stocks that just reported and are breaking out
            candidate["earnings_adjusted_score"] = candidate.get("composite_score", 0) * 1.05
        else:
            candidate["earnings_adjusted_score"] = candidate.get("composite_score", 0)

    # Re-sort by earnings-adjusted score
    candidates.sort(key=lambda x: x.get("earnings_adjusted_score", 0), reverse=True)

    # Re-rank
    for i, c in enumerate(candidates):
        c["rank"] = i + 1

    # Summary
    avoid_count = sum(1 for c in candidates if c.get("earnings", {}).get("action") == "avoid")
    caution_count = sum(1 for c in candidates if c.get("earnings", {}).get("action") == "caution")
    momentum_count = sum(1 for c in candidates if c.get("earnings", {}).get("action") == "post_earnings_momentum")

    if avoid_count > 0:
        logger.info(f"Earnings: {avoid_count} stocks penalized (imminent earnings)")
    if momentum_count > 0:
        logger.info(f"Earnings: {momentum_count} stocks boosted (post-earnings momentum)")

    return candidates
