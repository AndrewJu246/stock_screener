"""
Main screener orchestrator.

Ties together the data pipeline, signal engine, quality gate,
and scorer to produce a ranked list of stock candidates.

Usage:
    python screener.py                  # Full scan
    python screener.py --quick AAPL NVDA TSLA   # Quick scan specific tickers
    python screener.py --capital 10000  # Set portfolio capital
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from config import QUALITY_GATE, OUTPUT, CONFIDENCE_LADDER
from data_pipeline import (
    get_universe,
    get_price_history,
    get_price_history_batch,
    get_financials,
    get_insider_transactions,
    get_institutional_holders,
    get_market_regime_data,
    get_sector_etf_momentum,
)
from signals import compute_all_signals, volatility_ratio
from quality_gate import apply_quality_gate, apply_confidence_ladder
from scorer import (
    detect_regime,
    compute_composite_score,
    classify_strategy,
    compute_position_size,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(OUTPUT["output_dir"])
OUTPUT_DIR.mkdir(exist_ok=True)


def run_scan(
    tickers: list[str] = None,
    capital: float = 10_000,
    max_candidates: int = None,
) -> dict:
    """
    Run the full screening pipeline.

    Steps:
      1. Load universe (or use provided tickers)
      2. Fetch market regime data
      3. Batch-download price data
      4. For each stock: compute signals → quality gate → score → classify
      5. Rank and output top candidates

    Returns the full results dict.
    """
    start_time = time.time()
    max_candidates = max_candidates or OUTPUT["top_candidates"]

    # ── Step 1: Universe ──────────────────────────────────────────────
    if tickers:
        universe = tickers
        logger.info(f"Quick scan: {len(universe)} tickers")
    else:
        logger.info("Loading universe...")
        universe = get_universe()
        logger.info(f"Universe: {len(universe)} tickers")

    # ── Step 2: Market regime ─────────────────────────────────────────
    logger.info("Fetching market regime data...")
    market_data = get_market_regime_data()
    regime = detect_regime(market_data)
    logger.info(f"Detected regime: {regime} (VIX: {market_data.get('vix', '?')}, "
                f"S&P above 50 DMA: {market_data.get('sp500_above_50dma', '?')}, "
                f"above 200 DMA: {market_data.get('sp500_above_200dma', '?')})")

    # ── Step 3: Sector ETF momentum (for megatrend signals) ──────────
    logger.info("Fetching sector ETF momentum...")
    sector_etf_momentum = get_sector_etf_momentum()
    for trend, mom in sorted(sector_etf_momentum.items(), key=lambda x: -x[1]):
        logger.info(f"  {trend}: {mom:+.1f}% (3m)")

    # ── Step 4: Batch download price data ─────────────────────────────
    logger.info(f"Downloading price data for {len(universe)} tickers...")
    price_data = get_price_history_batch(universe, period="1y")
    logger.info(f"Price data loaded for {len(price_data)} tickers")

    # ── Step 5: Compute sector medians (for relative strength) ────────
    logger.info("Computing sector medians...")
    sector_returns = defaultdict(list)
    ticker_sectors = {}

    # Quick pass to get sectors and 3-month returns
    for ticker, hist in price_data.items():
        try:
            close = hist["Close"].values.astype(float)
            if len(close) > 63:
                ret_3m = (close[-1] / close[-63] - 1)
            else:
                ret_3m = 0

            # Try to get sector from cache (fast)
            from pathlib import Path as P
            cache_f = P("cache") / f"financials_{ticker}.json"
            sector = "Unknown"
            if cache_f.exists():
                try:
                    with open(cache_f) as f:
                        d = json.load(f)
                        sector = d.get("sector", "Unknown")
                except Exception:
                    pass

            sector_returns[sector].append(ret_3m)
            ticker_sectors[ticker] = sector
        except Exception:
            continue

    sector_medians = {
        sector: float(np.median(rets))
        for sector, rets in sector_returns.items()
        if len(rets) >= 3
    }

    # ── Step 6: Score each stock ──────────────────────────────────────
    logger.info("Scoring stocks...")
    all_results = []
    processed = 0
    skipped = 0
    gate_rejected = 0
    prefilter_skipped = 0

    for i, ticker in enumerate(universe):
        if ticker not in price_data:
            skipped += 1
            continue

        hist = price_data[ticker]

        # Progress logging
        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i + 1}/{len(universe)} "
                        f"({processed} scored, {prefilter_skipped} pre-filtered, "
                        f"{gate_rejected} gate-rejected)")

        try:
            # ── Pre-filter: cheap price-based checks first ────────────
            # Compute signals we can get from price data alone (free).
            # If none are interesting, skip the expensive API calls.
            from signals import volume_anomaly_score, momentum_score, volatility_ratio

            pre_vol = volume_anomaly_score(hist).get("score", 0)
            pre_mom = momentum_score(hist).get("score", 0)

            # Quick volume check from price data
            if len(hist) >= 20:
                avg_vol = hist["Volume"].iloc[-20:].mean()
                if avg_vol < QUALITY_GATE["min_avg_volume"]:
                    prefilter_skipped += 1
                    continue

            # If both price-based signals are weak, skip expensive calls.
            # A stock needs ≥2 signals at ≥50 to pass the quality gate.
            # If volume AND momentum are both below 35, it would need
            # ALL other signals to be high — extremely unlikely.
            best_price_signal = max(pre_vol, pre_mom)
            if best_price_signal < 30:
                prefilter_skipped += 1
                continue

            # ── Passed pre-filter: fetch expensive data ───────────────
            # Get financial data (uses cache)
            financials = get_financials(ticker)

            # Quick pre-filter: market cap
            if financials:
                mcap = financials.get("market_cap", 0)
                if mcap and mcap < QUALITY_GATE["min_market_cap"]:
                    skipped += 1
                    continue

            # Get insider and institutional data
            insider_txns = get_insider_transactions(ticker) or []
            institutional = get_institutional_holders(ticker) or []

            # Sector median for relative strength
            sector = ticker_sectors.get(ticker, "Unknown")
            sector_median = sector_medians.get(sector, 0)

            # Compute all 9 signals
            stock_result = compute_all_signals(
                ticker=ticker,
                hist=hist,
                financials=financials,
                insider_txns=insider_txns,
                institutional=institutional,
                sector_etf_momentum=sector_etf_momentum,
                sector_median_return=sector_median,
            )

            # Quality gate
            gate_result = apply_quality_gate(stock_result)
            if not gate_result["passed"]:
                gate_rejected += 1
                continue

            # Composite scoring (regime-aware + balanced benchmark)
            scored = compute_composite_score(stock_result, regime)

            # Strategy classification
            strategy = classify_strategy(stock_result, scored)

            # Confidence ladder
            confidence = apply_confidence_ladder(stock_result)

            # Position sizing
            position = compute_position_size(
                composite_score=scored["composite_score"],
                confidence=confidence["confidence"],
                total_capital=capital,
                current_positions=0,
                stock_price=stock_result.get("current_price", 0),
            )

            # Assemble full result
            result = {
                "ticker": ticker,
                "sector": stock_result.get("sector", "Unknown"),
                "industry": stock_result.get("industry", "Unknown"),
                "market_cap": stock_result.get("market_cap", 0),
                "current_price": stock_result.get("current_price", 0),
                "composite_score": scored["composite_score"],
                "balanced_score": scored.get("balanced_score", scored["composite_score"]),
                "regime": regime,
                "signals": {
                    name: {
                        "score": sig.get("score", 0),
                        "detail": sig.get("detail", ""),
                    }
                    for name, sig in stock_result.get("signals", {}).items()
                    if isinstance(sig, dict)
                },
                "strategy": strategy,
                "confidence": confidence,
                "quality_gate": gate_result,
                "position_sizing": position,
                "volatility_pct": stock_result.get("volatility_pct"),
            }

            all_results.append(result)
            processed += 1

        except Exception as e:
            logger.debug(f"Error processing {ticker}: {e}")
            skipped += 1
            continue

    # ── Step 7: Rank and trim ─────────────────────────────────────────
    all_results.sort(key=lambda x: x["composite_score"], reverse=True)
    top_candidates = all_results[:max_candidates]

    # Add rank
    for i, r in enumerate(top_candidates):
        r["rank"] = i + 1

    # ── Step 7a: Enhance top candidates with additional signals ───────
    # These run only on the top candidates (not all 2000+ stocks) for speed.
    logger.info(f"Enhancing top {len(top_candidates)} candidates...")

    for candidate in top_candidates:
        ticker = candidate["ticker"]
        signals = candidate.get("signals", {})

        # SEC EDGAR: enhance insider signal
        try:
            from sec_edgar import enhance_insider_signal
            insider_score = signals.get("smart_money", {}).get("score", 50)
            edgar = enhance_insider_signal(ticker, insider_score)
            candidate["sec_edgar"] = edgar
        except Exception:
            pass

        # Short interest: assess squeeze potential or warning
        try:
            from short_interest import assess_short_interest
            fund_score = signals.get("fundamentals", {}).get("score", 50)
            insider_score = signals.get("smart_money", {}).get("score", 50)
            short = assess_short_interest(ticker, fund_score, insider_score)
            candidate["short_interest"] = short
            # Apply score modifier
            modifier = short.get("score_modifier", 0)
            if modifier != 0:
                candidate["composite_score"] = round(
                    candidate["composite_score"] + modifier, 2
                )
        except Exception:
            pass

        # FinBERT news sentiment
        try:
            from finbert_sentiment import analyze_sentiment, is_available
            if is_available():
                sentiment = analyze_sentiment(ticker)
                candidate["news_sentiment"] = sentiment
                modifier = sentiment.get("score_modifier", 0)
                if modifier != 0:
                    candidate["composite_score"] = round(
                        candidate["composite_score"] + modifier, 2
                    )
        except Exception:
            pass

        # ML prediction (auto-trains when enough data exists)
        try:
            from ml_layer import predict_score, auto_check_and_train
            if candidate == top_candidates[0]:  # Only check once
                auto_check_and_train()
            ml = predict_score(
                {s: d.get("score", 0) for s, d in signals.items() if isinstance(d, dict)}
            )
            if ml:
                candidate["ml_prediction"] = ml
        except Exception:
            pass

    # Re-sort after score modifiers
    top_candidates.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, r in enumerate(top_candidates):
        r["rank"] = i + 1

    # ── Step 7b: Diversification guard ────────────────────────────────
    try:
        from diversifier import diversify_candidates
        top_candidates = diversify_candidates(top_candidates)
    except Exception as e:
        logger.warning(f"Diversification skipped: {e}")

    # ── Step 7c: Earnings calendar check ──────────────────────────────
    try:
        from earnings_calendar import apply_earnings_filter
        logger.info("Checking earnings calendar...")
        top_candidates = apply_earnings_filter(top_candidates)
    except Exception as e:
        logger.warning(f"Earnings calendar check skipped: {e}")

    elapsed = time.time() - start_time

    # ── Step 8: Build summary ─────────────────────────────────────────
    summary = {
        "scan_date": datetime.now().isoformat(),
        "regime": regime,
        "regime_description": scored.get("regime_description", "") if all_results else "",
        "market_data": market_data,
        "sector_etf_momentum": sector_etf_momentum,
        "universe_size": len(universe),
        "price_data_loaded": len(price_data),
        "stocks_scored": processed,
        "gate_rejected": gate_rejected,
        "prefilter_skipped": prefilter_skipped,
        "skipped": skipped,
        "candidates_returned": len(top_candidates),
        "elapsed_seconds": round(elapsed, 1),
        "capital": capital,
        "strategy_breakdown": {
            "long_term_hold": sum(
                1 for r in top_candidates
                if r["strategy"]["strategy"] == "long_term_hold"
            ),
            "short_term_momentum": sum(
                1 for r in top_candidates
                if r["strategy"]["strategy"] == "short_term_momentum"
            ),
        },
    }

    output = {
        "summary": summary,
        "candidates": top_candidates,
    }

    # ── Step 9: Save results ──────────────────────────────────────────
    results_path = OUTPUT_DIR / OUTPUT["results_file"]
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n{'='*60}")
    logger.info(f"SCAN COMPLETE in {elapsed:.1f}s")
    logger.info(f"  Universe: {len(universe)} | Scored: {processed} | "
                f"Rejected: {gate_rejected} | Pre-filtered: {prefilter_skipped} | "
                f"Skipped: {skipped}")
    logger.info(f"  Regime: {regime}")
    logger.info(f"  Top candidates: {len(top_candidates)}")
    logger.info(f"  Results saved to: {results_path}")
    logger.info(f"{'='*60}")

    # Print top 10
    if top_candidates:
        logger.info("\nTOP 10 CANDIDATES:")
        logger.info(f"{'Rank':<5} {'Ticker':<8} {'Score':<7} {'Strategy':<20} "
                     f"{'Confidence':<12} {'Sector':<20}")
        logger.info("-" * 80)
        for r in top_candidates[:10]:
            logger.info(
                f"{r['rank']:<5} {r['ticker']:<8} "
                f"{r['composite_score']:<7.1f} "
                f"{r['strategy']['strategy']:<20} "
                f"{r['confidence']['confidence']:<12} "
                f"{r['sector']:<20}"
            )

    # ── Step 10: Prediction tracker ───────────────────────────────────
    try:
        from tracker import log_predictions, update_predictions
        new_logged = log_predictions(output)
        if new_logged > 0:
            logger.info(f"Tracker: {new_logged} new predictions logged")
        # Also update existing predictions with current prices
        update_result = update_predictions()
        if update_result["updated"] > 0:
            logger.info(
                f"Tracker: updated {update_result['updated']} existing predictions "
                f"({update_result['total_active']} active, "
                f"{update_result['total_completed']} completed)"
            )
    except Exception as e:
        logger.warning(f"Tracker integration error (non-fatal): {e}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Stock Screener")
    parser.add_argument(
        "--quick",
        nargs="+",
        help="Quick scan specific tickers (e.g., --quick AAPL NVDA TSLA)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10_000,
        help="Total portfolio capital (default: $10,000)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Max candidates to return (default: 50)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging (show debug messages)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_scan(
        tickers=args.quick,
        capital=args.capital,
        max_candidates=args.max,
    )


if __name__ == "__main__":
    main()
