"""
Signal calculators for the stock screener.

Each function computes a normalized score (0–100) for one signal family.
Higher = stronger signal (more bullish).

Signal families:
  1. volume_anomaly     — unusual volume relative to history
  2. momentum           — multi-timeframe price trend strength
  3. fundamentals       — revenue/earnings growth acceleration
  4. smart_money        — insider buying + institutional accumulation
  5. sector_rotation    — is money flowing into this stock's sector?
  6. earnings_quality   — cash flow vs reported earnings
  7. analyst_revisions  — are analysts upgrading estimates?
  8. tech_megatrend     — exposure to accelerating technology trends
  9. relative_strength  — outperforming peers in the same sector
"""

import numpy as np
import pandas as pd
from typing import Optional

from config import SIGNAL_PARAMS, MEGATRENDS
from discovery_signal import discovery_potential_score


# ═══════════════════════════════════════════════════════════════════════
# 1. Volume anomaly
# ═══════════════════════════════════════════════════════════════════════

def volume_anomaly_score(hist: pd.DataFrame) -> dict:
    """
    Detect unusual volume spikes.
    High volume + price up = accumulation (bullish).
    High volume + price down = distribution (bearish).

    Returns: {score: 0-100, detail: str}
    """
    params = SIGNAL_PARAMS["volume_anomaly"]
    lookback = params["lookback_days"]

    if hist is None or len(hist) < lookback + 5:
        return {"score": 0, "detail": "Insufficient data"}

    vol = hist["Volume"].values.astype(float)
    close = hist["Close"].values.astype(float)

    # Average volume over lookback (excluding last 5 days)
    avg_vol = np.mean(vol[-lookback:-5])
    if avg_vol <= 0:
        return {"score": 0, "detail": "No volume history"}

    # Recent 5-day average volume
    recent_vol = np.mean(vol[-5:])
    vol_ratio = recent_vol / avg_vol

    # Price change during the volume period
    price_chg_5d = (close[-1] - close[-6]) / close[-6] if close[-6] != 0 else 0

    # Scoring logic
    if vol_ratio < params["spike_threshold"]:
        score = max(0, vol_ratio * 10)
        detail = f"Normal volume ({vol_ratio:.1f}x avg)"
    elif price_chg_5d < -0.03 and vol_ratio > 2.0:
        score = 0
        detail = f"Distribution signal ({vol_ratio:.1f}x vol, {price_chg_5d:+.1%} price)"
    elif price_chg_5d > 0:
        # Accumulation: volume up + price up
        raw = min(vol_ratio / params["strong_spike"] * 80, 80)
        # Boost for strong price confirmation
        price_bonus = min(price_chg_5d * 200, 20)
        score = min(raw + price_bonus, 100)
        detail = f"Accumulation ({vol_ratio:.1f}x vol, {price_chg_5d:+.1%} price)"
    else:
        score = min(vol_ratio * 12, 40)
        detail = f"Elevated volume, flat price ({vol_ratio:.1f}x)"

    return {"score": round(score, 1), "detail": detail}


# ═══════════════════════════════════════════════════════════════════════
# 2. Momentum (multi-timeframe)
# ═══════════════════════════════════════════════════════════════════════

def momentum_score(hist: pd.DataFrame) -> dict:
    """
    Multi-timeframe momentum: stocks trending up across
    1-week, 1-month, 3-month, and 6-month windows.

    Bonus if shorter windows are stronger (acceleration).
    """
    params = SIGNAL_PARAMS["momentum"]
    windows = params["windows"]

    if hist is None or len(hist) < max(windows) + 5:
        return {"score": 0, "detail": "Insufficient data"}

    close = hist["Close"].values.astype(float)
    current = close[-1]

    returns = {}
    for w in windows:
        if len(close) > w:
            past = close[-(w + 1)]
            returns[w] = (current - past) / past if past != 0 else 0
        else:
            returns[w] = 0

    # Score each window (positive return = positive contribution)
    window_scores = []
    for w in windows:
        r = returns[w]
        # Map returns to 0-100: -20% → 0, 0% → 50, +20% → 100
        s = np.clip((r + 0.20) / 0.40 * 100, 0, 100)
        window_scores.append(s)

    # Base score = weighted average (shorter windows matter more)
    weights = [0.35, 0.30, 0.20, 0.15]
    base_score = sum(s * w for s, w in zip(window_scores, weights))

    # Acceleration bonus: each shorter window > the longer one
    is_accelerating = all(
        returns[windows[i]] > returns[windows[i + 1]]
        for i in range(len(windows) - 1)
        if windows[i] in returns and windows[i + 1] in returns
    )

    # Exhaustion detection: 1-week return is much weaker than 1-month
    # This catches stocks that had a huge run but are now cooling off
    r_1w = returns.get(5, 0)
    r_1m = returns.get(21, 0)
    r_3m = returns.get(63, 0)

    # Deceleration: weekly momentum is < 25% of monthly (adjusted for time)
    # A stock doing +2% in 1w after +57% in 1m is clearly decelerating
    weekly_annualized = r_1w * 4   # rough monthly equivalent
    is_decelerating = (
        r_1m > 0.10 and                    # Had a big month (>10%)
        weekly_annualized < r_1m * 0.25     # But weekly pace is <25% of monthly pace
    )

    # Extreme run detection: +40% in a month is unusual; may be overextended
    is_extreme_run = r_1m > 0.40

    if is_accelerating and base_score > 50:
        score = min(base_score * params["acceleration_bonus"], 100)
        detail = f"Accelerating momentum ({r_1w:+.1%} 1w, {r_1m:+.1%} 1m, {r_3m:+.1%} 3m)"
    elif is_decelerating:
        score = base_score * 0.80  # 20% penalty for deceleration
        detail = f"Decelerating momentum ({r_1w:+.1%} 1w, {r_1m:+.1%} 1m, {r_3m:+.1%} 3m) ⚠ cooling off"
    else:
        score = base_score
        detail = f"Momentum: {r_1w:+.1%} 1w, {r_1m:+.1%} 1m, {r_3m:+.1%} 3m"

    # Extreme run flag (doesn't change score, but adds warning)
    if is_extreme_run:
        detail += f" ⚠ extreme run ({r_1m:+.0%} in 1mo)"

    return {"score": round(score, 1), "detail": detail}


# ═══════════════════════════════════════════════════════════════════════
# 3. Fundamentals (revenue/earnings acceleration)
# ═══════════════════════════════════════════════════════════════════════

def fundamentals_score(financials: dict) -> dict:
    """
    Score based on revenue growth acceleration, margin expansion,
    and earnings growth.
    """
    if not financials:
        return {"score": 0, "detail": "No financial data"}

    params = SIGNAL_PARAMS["fundamentals"]
    scores = {}

    # Revenue growth acceleration
    rev = financials.get("revenue_quarterly")
    if rev and len(rev) >= 4:
        values = [r["value"] for r in sorted(rev, key=lambda x: x["date"])]
        if len(values) >= 4 and values[-4] != 0 and values[-2] != 0:
            growth_recent = (values[-1] - values[-2]) / abs(values[-2])
            growth_prior = (values[-3] - values[-4]) / abs(values[-4])
            # Acceleration = growth speeding up
            if growth_recent > growth_prior and growth_recent > 0:
                accel = growth_recent - growth_prior
                scores["revenue"] = min(50 + accel * 200, 100)
            elif growth_recent > 0:
                scores["revenue"] = min(30 + growth_recent * 100, 70)
            else:
                scores["revenue"] = max(0, 30 + growth_recent * 100)
        else:
            scores["revenue"] = 30  # Neutral
    else:
        scores["revenue"] = 0

    # Margin expansion (gross profit / revenue trending up)
    gp = financials.get("gross_profit_quarterly")
    if gp and rev and len(gp) >= 2 and len(rev) >= 2:
        gp_vals = [r["value"] for r in sorted(gp, key=lambda x: x["date"])]
        rev_vals = [r["value"] for r in sorted(rev, key=lambda x: x["date"])]
        if rev_vals[-1] != 0 and rev_vals[-2] != 0:
            margin_now = gp_vals[-1] / rev_vals[-1]
            margin_prev = gp_vals[-2] / rev_vals[-2]
            margin_chg = margin_now - margin_prev
            scores["margin"] = np.clip(50 + margin_chg * 500, 0, 100)
        else:
            scores["margin"] = 30
    else:
        scores["margin"] = 0

    # Earnings growth
    ni = financials.get("net_income_quarterly")
    if ni and len(ni) >= 2:
        ni_vals = [r["value"] for r in sorted(ni, key=lambda x: x["date"])]
        if ni_vals[-2] != 0:
            earnings_growth = (ni_vals[-1] - ni_vals[-2]) / abs(ni_vals[-2])
            scores["earnings"] = np.clip(50 + earnings_growth * 100, 0, 100)
        else:
            scores["earnings"] = 30
    else:
        scores["earnings"] = 0

    # Weighted composite
    w = params
    total = (
        scores.get("revenue", 0) * w["revenue_growth_weight"]
        + scores.get("margin", 0) * w["margin_expansion_weight"]
        + scores.get("earnings", 0) * w["earnings_growth_weight"]
    )

    parts = ", ".join(f"{k}={v:.0f}" for k, v in scores.items() if v > 0)
    return {"score": round(total, 1), "detail": f"Fundamentals: {parts}"}


# ═══════════════════════════════════════════════════════════════════════
# 4. Smart money (insider + institutional)
# ═══════════════════════════════════════════════════════════════════════

def smart_money_score(
    insider_txns: list[dict],
    institutional: list[dict],
) -> dict:
    """
    Score insider buying clusters and institutional accumulation.
    """
    params = SIGNAL_PARAMS["smart_money"]
    insider_score = 0
    inst_score = 0

    # Insider buying
    if insider_txns:
        from datetime import datetime, timedelta
        lookback = datetime.now() - timedelta(days=params["insider_buy_lookback_days"])

        buys = []
        sells = []
        for txn in insider_txns:
            txn_type = str(txn.get("type", "")).lower()
            try:
                txn_date = pd.to_datetime(txn.get("date", ""))
                if pd.isna(txn_date):
                    continue
            except Exception:
                continue

            if txn_date.tz_localize(None) if txn_date.tzinfo else txn_date < lookback:
                continue

            if "purchase" in txn_type or "buy" in txn_type:
                buys.append(txn)
            elif "sale" in txn_type or "sell" in txn_type:
                sells.append(txn)

        if len(buys) >= params["insider_cluster_min"]:
            # Cluster of insider buys — very strong
            total_buy_value = sum(abs(b.get("value", 0)) for b in buys)
            insider_score = min(60 + len(buys) * 10 + total_buy_value / 500_000, 100)
        elif len(buys) > 0:
            insider_score = 30 + len(buys) * 15
        elif len(sells) > 3 and len(buys) == 0:
            insider_score = 10  # Heavy selling, no buying = bearish
        else:
            insider_score = 30  # Neutral

    # Institutional (simple: more holders + high value = positive)
    if institutional:
        num_inst = len(institutional)
        total_value = sum(h.get("value", 0) for h in institutional)
        inst_score = min(30 + num_inst * 2 + total_value / 1e9 * 10, 100)
    else:
        inst_score = 20  # Neutral — absence of data isn't bearish

    # Weighted
    w = params
    total = insider_score * w["insider_weight"] + inst_score * w["institutional_13f_weight"]

    detail = f"Insiders: {insider_score:.0f}/100, Institutional: {inst_score:.0f}/100"
    return {"score": round(total, 1), "detail": detail}


# ═══════════════════════════════════════════════════════════════════════
# 5. Sector rotation
# ═══════════════════════════════════════════════════════════════════════

def sector_rotation_score(
    stock_sector: str,
    sector_etf_momentum: dict[str, float],
) -> dict:
    """
    Is money flowing into this stock's sector/megatrend?
    Higher sector momentum = higher score.
    """
    # Find which megatrends this sector maps to
    relevant_trends = []
    for trend_key, trend_info in MEGATRENDS.items():
        if stock_sector in trend_info.get("sectors", []):
            mom = sector_etf_momentum.get(trend_key, 0)
            relevant_trends.append((trend_info["name"], mom))

    if not relevant_trends:
        return {"score": 40, "detail": f"Sector '{stock_sector}' not mapped to megatrends"}

    # Average momentum across relevant megatrends
    avg_mom = np.mean([m for _, m in relevant_trends])

    # Map momentum to score: -10% → 10, 0% → 50, +20% → 90
    score = np.clip(50 + avg_mom * 2, 0, 100)

    best = max(relevant_trends, key=lambda x: x[1])
    detail = f"Sector momentum: {best[0]} {best[1]:+.1f}%"

    return {"score": round(score, 1), "detail": detail}


# ═══════════════════════════════════════════════════════════════════════
# 6. Earnings quality
# ═══════════════════════════════════════════════════════════════════════

def earnings_quality_score(financials: dict) -> dict:
    """
    Compare operating cash flow to reported net income.
    Real companies have cash flow backing their earnings.
    Lottery stocks often have earnings without cash.
    """
    if not financials:
        return {"score": 0, "detail": "No financial data"}

    params = SIGNAL_PARAMS["earnings_quality"]

    cffo = financials.get("operating_cashflow_quarterly")
    ni = financials.get("net_income_quarterly")

    if not cffo or not ni or len(cffo) < 2 or len(ni) < 2:
        return {"score": 30, "detail": "Insufficient cash flow data"}

    cffo_vals = [r["value"] for r in sorted(cffo, key=lambda x: x["date"])]
    ni_vals = [r["value"] for r in sorted(ni, key=lambda x: x["date"])]

    # Cash flow to earnings ratio (latest quarter)
    if ni_vals[-1] != 0:
        cf_ratio = cffo_vals[-1] / abs(ni_vals[-1])
    else:
        cf_ratio = 1.0 if cffo_vals[-1] > 0 else 0.0

    # Accruals ratio: (net income - cash flow) / total assets
    total_assets = financials.get("total_assets")
    if total_assets and total_assets > 0:
        accruals = (ni_vals[-1] - cffo_vals[-1]) / total_assets
    else:
        accruals = 0

    # Scoring
    if cf_ratio >= params["cffo_to_net_income_min"]:
        # Good: cash flow supports earnings
        score = min(60 + cf_ratio * 20, 95)
    elif cf_ratio >= 0.4:
        score = 40 + cf_ratio * 30
    else:
        # Poor: earnings without cash flow = red flag
        score = max(0, cf_ratio * 40)

    # Penalize high accruals (sign of earnings manipulation)
    if abs(accruals) > params["accruals_penalty_threshold"]:
        score = max(0, score - 20)

    # Cash flow trend (is it improving?)
    if len(cffo_vals) >= 4:
        cf_trend = (cffo_vals[-1] - cffo_vals[-2]) / abs(cffo_vals[-2]) if cffo_vals[-2] != 0 else 0
        if cf_trend > 0.1:
            score = min(score + 10, 100)

    detail = f"CF/Earnings ratio: {cf_ratio:.2f}, Accruals: {accruals:.3f}"
    return {"score": round(score, 1), "detail": detail}


# ═══════════════════════════════════════════════════════════════════════
# 7. Analyst revisions
# ═══════════════════════════════════════════════════════════════════════

def analyst_revisions_score(financials: dict) -> dict:
    """
    Are analysts upgrading their estimates?
    Uses analyst target price vs current price as a proxy.
    Also detects when price has overshot analyst targets (overextension risk).
    """
    if not financials:
        return {"score": 0, "detail": "No analyst data"}

    target = financials.get("analyst_target_mean")
    current = financials.get("current_price")
    recommendation = financials.get("analyst_recommendation", "")

    if not target or not current or current <= 0:
        return {"score": 30, "detail": "No analyst target available"}

    # Upside to analyst target
    upside = (target - current) / current

    # Map to score: -10% → 20, 0% → 40, +20% → 70, +50% → 95
    score = np.clip(40 + upside * 150, 0, 100)

    # Recommendation bonus
    rec_map = {"strong_buy": 15, "buy": 10, "hold": 0, "sell": -10, "strong_sell": -20}
    rec_bonus = rec_map.get(str(recommendation).lower().replace(" ", "_"), 0)
    score = np.clip(score + rec_bonus, 0, 100)

    # Overextension detection: price is ABOVE analyst target
    overextended = upside < -0.05  # Price is >5% above target
    if overextended:
        rec_lower = str(recommendation).lower().replace(" ", "_")
        if rec_lower in ("strong_buy", "buy"):
            # Price blew past target but analysts still bullish
            # Could mean targets are lagging — moderate the penalty
            detail = (f"Target ${target:.0f} ({upside:+.1%}), Rec: {recommendation} "
                      f"⚠ price above target — analysts may be lagging")
        else:
            # Price above target AND analysts aren't bullish — red flag
            score = max(score * 0.7, 0)
            detail = (f"Target ${target:.0f} ({upside:+.1%}), Rec: {recommendation} "
                      f"⚠ overextended past target")
    else:
        detail = f"Target ${target:.0f} ({upside:+.1%} upside), Rec: {recommendation}"

    return {"score": round(score, 1), "detail": detail}


# ═══════════════════════════════════════════════════════════════════════
# 8. Tech / megatrend exposure
# ═══════════════════════════════════════════════════════════════════════

def tech_megatrend_score(
    financials: dict,
    sector_etf_momentum: dict[str, float],
) -> dict:
    """
    Score based on: (a) which megatrends the company is exposed to,
    and (b) how strongly those megatrends are accelerating,
    and (c) R&D investment level.
    """
    if not financials:
        return {"score": 0, "detail": "No data"}

    sector = financials.get("sector", "")
    industry = financials.get("industry", "")
    description = financials.get("description", "").lower()

    # Find megatrend exposure via keyword matching + sector
    exposures = []
    for trend_key, trend_info in MEGATRENDS.items():
        match_score = 0

        # Sector match
        if sector in trend_info.get("sectors", []):
            match_score += 1

        # Keyword match in company description
        keywords_found = sum(
            1 for kw in trend_info.get("keywords", [])
            if kw.lower() in description
        )
        match_score += min(keywords_found, 3)  # Cap at 3

        if match_score > 0:
            etf_mom = sector_etf_momentum.get(trend_key, 0)
            exposures.append({
                "trend": trend_info["name"],
                "match_strength": match_score,
                "etf_momentum": etf_mom,
            })

    if not exposures:
        return {"score": 25, "detail": "No megatrend exposure detected"}

    # Score = match strength × megatrend momentum
    best = max(exposures, key=lambda x: x["match_strength"] * max(x["etf_momentum"], 0))
    trend_score = min(
        30 + best["match_strength"] * 10 + max(best["etf_momentum"], 0) * 2,
        85,
    )

    # R&D bonus: companies investing in research get a bump
    rd = financials.get("rd_expense")
    rev = financials.get("revenue_quarterly")
    if rd and rev:
        rd_vals = [r["value"] for r in rd] if isinstance(rd, list) else []
        rev_vals = [r["value"] for r in rev] if isinstance(rev, list) else []
        if rd_vals and rev_vals and rev_vals[-1] > 0:
            rd_pct = abs(rd_vals[-1]) / rev_vals[-1]
            if rd_pct > 0.15:  # >15% of revenue on R&D
                trend_score = min(trend_score + 15, 100)
            elif rd_pct > 0.08:
                trend_score = min(trend_score + 8, 100)

    detail = f"Megatrend: {best['trend']} (ETF {best['etf_momentum']:+.1f}%)"
    return {"score": round(trend_score, 1), "detail": detail}


# ═══════════════════════════════════════════════════════════════════════
# 9. Relative strength vs peers
# ═══════════════════════════════════════════════════════════════════════

def relative_strength_score(
    stock_return_3m: float,
    sector_median_return_3m: float,
) -> dict:
    """
    Is this stock outperforming its sector peers?
    """
    if sector_median_return_3m is None:
        return {"score": 50, "detail": "No peer comparison data"}

    # Relative performance
    relative = stock_return_3m - sector_median_return_3m

    # Map to score: -20% underperformance → 10, 0% → 50, +20% outperformance → 90
    score = np.clip(50 + relative * 200, 0, 100)

    detail = (
        f"Stock: {stock_return_3m:+.1%} vs Sector median: "
        f"{sector_median_return_3m:+.1%} (relative: {relative:+.1%})"
    )
    return {"score": round(score, 1), "detail": detail}


# ═══════════════════════════════════════════════════════════════════════
# Volatility check (for quality gate, not a signal itself)
# ═══════════════════════════════════════════════════════════════════════

def volatility_ratio(hist: pd.DataFrame) -> Optional[float]:
    """
    Calculate the stock's average daily range as a ratio.
    Used by the quality gate to flag lottery-like volatility.
    Returns the average daily range as a percentage.
    """
    if hist is None or len(hist) < 20:
        return None

    high = hist["High"].values.astype(float)
    low = hist["Low"].values.astype(float)
    close = hist["Close"].values.astype(float)

    # Average true range (last 20 days) as % of price
    daily_ranges = (high[-20:] - low[-20:]) / close[-20:]
    avg_range_pct = float(np.mean(daily_ranges))

    return avg_range_pct


# ═══════════════════════════════════════════════════════════════════════
# Compute all signals for a stock
# ═══════════════════════════════════════════════════════════════════════

def compute_all_signals(
    ticker: str,
    hist: pd.DataFrame,
    financials: dict,
    insider_txns: list[dict],
    institutional: list[dict],
    sector_etf_momentum: dict[str, float],
    sector_median_return: float,
) -> dict:
    """
    Run all 9 signal calculators and return a unified result dict.
    """
    # 3-month return for this stock
    close = hist["Close"].values.astype(float)
    stock_return_3m = (close[-1] / close[-63] - 1) if len(close) > 63 else 0

    sector = financials.get("sector", "Unknown") if financials else "Unknown"

    signals = {
        "volume_anomaly": volume_anomaly_score(hist),
        "momentum": momentum_score(hist),
        "fundamentals": fundamentals_score(financials),
        "smart_money": smart_money_score(insider_txns, institutional),
        "sector_rotation": sector_rotation_score(sector, sector_etf_momentum),
        "earnings_quality": earnings_quality_score(financials),
        "analyst_revisions": analyst_revisions_score(financials),
        "tech_megatrend": tech_megatrend_score(financials, sector_etf_momentum),
        "relative_strength": relative_strength_score(stock_return_3m, sector_median_return),
        "discovery_potential": discovery_potential_score(financials, institutional, hist),
    }

    # Volatility (for quality gate)
    vol_ratio = volatility_ratio(hist)

    return {
        "ticker": ticker,
        "signals": signals,
        "volatility_pct": vol_ratio,
        "sector": sector,
        "industry": financials.get("industry", "Unknown") if financials else "Unknown",
        "market_cap": financials.get("market_cap", 0) if financials else 0,
        "current_price": financials.get("current_price", 0) if financials else 0,
    }
