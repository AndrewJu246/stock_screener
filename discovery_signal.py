"""
Discovery potential signal.

This is the core differentiator of the screener. It answers:
"Is this stock still in the early innings, or has everyone
already found it?"

A stock like NVIDIA in 2020 (small-ish, few analysts, low institutional
ownership, massive addressable market) scores high.
NVIDIA in 2026 ($2.8T, 50 analysts, 80% institutional) scores low.

Components:
  1. Analyst coverage: fewer analysts = more information asymmetry
  2. Institutional ownership: 30-60% = sweet spot (being discovered)
  3. Market cap headroom: smaller = more room to multiply
  4. Price vs analyst target: big upside gap = market hasn't caught up
  5. Revenue growth vs market cap growth: fundamentals outpacing valuation

This signal gets HIGH weight because it's the whole point — we're not
looking for stocks that are already strong, we're looking for stocks
that are about to become strong.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def discovery_potential_score(
    financials: dict,
    institutional: list[dict],
    hist=None,
) -> dict:
    """
    Compute discovery potential score (0-100).
    Higher = more undiscovered, more room to run.
    Lower = already crowded, limited upside.
    """
    if not financials:
        return {"score": 50, "detail": "No data — neutral discovery score"}

    scores = {}
    details = []

    market_cap = financials.get("market_cap", 0) or 0
    current_price = financials.get("current_price", 0) or 0
    analyst_target = financials.get("analyst_target_mean")
    recommendation = financials.get("analyst_recommendation", "")

    # ── 1. Analyst coverage (fewer = better) ──────────────────────────
    # yfinance doesn't directly give analyst count, but we can infer from
    # whether target/recommendation exist and the recommendation spread.
    # Stocks with "strong_buy" from 50 analysts are crowded.
    # Stocks with "buy" from 5 analysts are still being discovered.
    if analyst_target and current_price and current_price > 0:
        # If target exists, analysts are covering it
        upside = (analyst_target - current_price) / current_price

        if upside > 0.30:
            # Big upside gap — market hasn't caught up to analyst view
            scores["analyst_gap"] = min(80 + upside * 50, 95)
            details.append(f"Analyst upside {upside:+.0%} — market hasn't caught up")
        elif upside > 0.10:
            scores["analyst_gap"] = 60 + upside * 100
            details.append(f"Moderate analyst upside {upside:+.0%}")
        elif upside > 0:
            scores["analyst_gap"] = 40 + upside * 100
            details.append(f"Slim analyst upside {upside:+.0%}")
        else:
            # Price above target — already overshot expectations
            scores["analyst_gap"] = max(10, 30 + upside * 100)
            details.append(f"Price above analyst target ({upside:+.0%})")
    else:
        # No analyst coverage at all — could be very underfollowed
        scores["analyst_gap"] = 70
        details.append("No analyst coverage — potentially undiscovered")

    # ── 2. Institutional ownership (sweet spot: 30-60%) ───────────────
    if institutional:
        total_inst_pct = sum(h.get("pct_held", 0) for h in institutional[:20])
        # Cap at reasonable level (data can be noisy)
        total_inst_pct = min(total_inst_pct, 1.0)

        if total_inst_pct < 0.20:
            # Very low — either too small or has issues
            scores["institutional"] = 55
            details.append(f"Low institutional ownership ({total_inst_pct:.0%})")
        elif total_inst_pct < 0.45:
            # Sweet spot — being discovered by institutions
            scores["institutional"] = 85
            details.append(f"Discovery phase: {total_inst_pct:.0%} institutional")
        elif total_inst_pct < 0.65:
            # Moderate — still some room
            scores["institutional"] = 60
            details.append(f"Moderate institutional ({total_inst_pct:.0%})")
        elif total_inst_pct < 0.80:
            # Getting crowded
            scores["institutional"] = 35
            details.append(f"Crowded: {total_inst_pct:.0%} institutional")
        else:
            # Fully saturated — everyone who's going to buy already has
            scores["institutional"] = 15
            details.append(f"Saturated: {total_inst_pct:.0%} institutional — no new buyers")
    else:
        scores["institutional"] = 50  # Neutral if no data

    # ── 3. Market cap headroom ────────────────────────────────────────
    # How much can this stock realistically grow?
    if market_cap > 0:
        if market_cap < 1_000_000_000:  # Under $1B
            scores["mcap_headroom"] = 95
            details.append(f"Small cap (${market_cap/1e9:.1f}B) — massive headroom")
        elif market_cap < 5_000_000_000:  # $1-5B
            scores["mcap_headroom"] = 85
            details.append(f"${market_cap/1e9:.1f}B — strong growth headroom")
        elif market_cap < 20_000_000_000:  # $5-20B
            scores["mcap_headroom"] = 70
            details.append(f"${market_cap/1e9:.0f}B — good headroom")
        elif market_cap < 50_000_000_000:  # $20-50B
            scores["mcap_headroom"] = 50
            details.append(f"${market_cap/1e9:.0f}B — moderate headroom")
        elif market_cap < 200_000_000_000:  # $50-200B
            scores["mcap_headroom"] = 30
            details.append(f"${market_cap/1e9:.0f}B — limited headroom")
        else:  # $200B+
            scores["mcap_headroom"] = 10
            details.append(f"${market_cap/1e9:.0f}B — mega cap, minimal growth headroom")
    else:
        scores["mcap_headroom"] = 50

    # ── 4. Revenue growth vs price appreciation ───────────────────────
    # If revenue is growing faster than the stock price has appreciated,
    # the fundamentals are outpacing the market's recognition.
    rev = financials.get("revenue_quarterly")
    if rev and len(rev) >= 4 and hist is not None and len(hist) >= 252:
        rev_vals = [r["value"] for r in sorted(rev, key=lambda x: x["date"])]
        if rev_vals[-4] > 0:
            rev_growth = (rev_vals[-1] - rev_vals[-4]) / abs(rev_vals[-4])
        else:
            rev_growth = 0

        # 1-year price appreciation
        close = hist["Close"].values.astype(float)
        if close[-252] > 0:
            price_growth = (close[-1] - close[-252]) / close[-252]
        else:
            price_growth = 0

        # Fundamental outpacing = revenue growing faster than price
        if rev_growth > 0 and rev_growth > price_growth:
            gap = rev_growth - price_growth
            scores["fundamental_gap"] = min(60 + gap * 200, 95)
            details.append(
                f"Revenue outpacing price: rev {rev_growth:+.0%} vs price {price_growth:+.0%}"
            )
        elif rev_growth > 0:
            scores["fundamental_gap"] = 40
            details.append(f"Revenue growth {rev_growth:+.0%} (price already reflects it)")
        else:
            scores["fundamental_gap"] = 20
            details.append("Revenue declining")
    else:
        scores["fundamental_gap"] = 50

    # ── Composite ─────────────────────────────────────────────────────
    weights = {
        "analyst_gap": 0.20,
        "institutional": 0.25,
        "mcap_headroom": 0.30,    # Market cap is the strongest predictor of growth potential
        "fundamental_gap": 0.25,
    }

    composite = sum(scores.get(k, 50) * w for k, w in weights.items())
    top_detail = sorted(details, key=len)[:2]  # Keep it concise

    return {
        "score": round(composite, 1),
        "components": scores,
        "detail": " | ".join(top_detail) if top_detail else "No discovery data",
    }
