"""
Backtesting engine: runs the screening signals against historical data
to validate which signals and combinations actually predict returns.

This is different from the tracker (which tracks live predictions going forward).
The backtester looks backward — simulating what the screener would have
recommended in the past and checking if those picks worked.

Usage:
    python backtester.py run                    # Run backtest (last 2 years)
    python backtester.py run --years 3          # Run backtest (last 3 years)
    python backtester.py run --tickers NVDA AMD AAPL   # Backtest specific stocks
    python backtester.py report                 # Show backtest results
    python backtester.py report --sector        # Sector-level breakdown
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import REGIMES, SIGNAL_PARAMS, QUALITY_GATE
from signals import (
    volume_anomaly_score,
    momentum_score,
    earnings_quality_score,
    relative_strength_score,
    volatility_ratio,
)

logger = logging.getLogger(__name__)

BACKTEST_DIR = Path("backtest_data")
BACKTEST_DIR.mkdir(exist_ok=True)

RESULTS_FILE = BACKTEST_DIR / "backtest_results.json"

# Forward return windows to evaluate
EVAL_WINDOWS = [7, 14, 30, 60, 90]

# Minimum score threshold to consider a signal "triggered"
SIGNAL_THRESHOLD = 60


def run_backtest(
    tickers: list[str] = None,
    years: int = 2,
    sample_interval_days: int = 21,  # Check every ~month
) -> dict:
    """
    Run a historical backtest.

    For each stock, at regular intervals over the past N years:
      1. Compute signals using only data available at that date
      2. Record which signals were "high" (≥60)
      3. Check what actually happened to the stock price over the next
         7, 14, 30, 60, 90 days

    This tells us which signals actually predicted positive returns.
    """
    from data_pipeline import get_price_history, get_universe

    start_time = time.time()

    # Get tickers
    if tickers:
        universe = tickers
    else:
        universe = get_universe()
        # For full backtest, use a random sample to keep it tractable
        if len(universe) > 200:
            import random
            random.seed(42)
            universe = random.sample(universe, 200)
            logger.info(f"Sampled 200 tickers from universe for backtest")

    logger.info(f"Backtesting {len(universe)} tickers over {years} years...")

    period = f"{years + 1}y"  # Extra year for lookback data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    all_observations = []
    processed = 0

    for i, ticker in enumerate(universe):
        if (i + 1) % 25 == 0:
            logger.info(f"  Progress: {i + 1}/{len(universe)} ({processed} observations)")

        try:
            hist = get_price_history(ticker, period=period)
            if hist is None or len(hist) < 252:  # Need at least 1 year of data
                continue

            close = hist["Close"].values.astype(float)
            dates = hist.index

            # Walk through time at regular intervals
            # Start from 252 days in (need lookback) and stop 90 days from end (need forward data)
            for j in range(252, len(hist) - 90, sample_interval_days):
                # Use only data up to this point (no future leakage)
                hist_slice = hist.iloc[:j + 1]

                # Compute price-based signals
                vol_result = volume_anomaly_score(hist_slice)
                mom_result = momentum_score(hist_slice)

                # Relative strength (simplified — use stock's own 3m return vs 0)
                close_at_j = close[j]
                if j >= 63 and close[j - 63] > 0:
                    ret_3m = (close_at_j - close[j - 63]) / close[j - 63]
                else:
                    ret_3m = 0
                rs_result = relative_strength_score(ret_3m, 0)

                # Volatility
                vol_pct = volatility_ratio(hist_slice)

                # Determine which signals are "high"
                signals = {
                    "volume_anomaly": vol_result.get("score", 0),
                    "momentum": mom_result.get("score", 0),
                    "relative_strength": rs_result.get("score", 0),
                }
                high_signals = [s for s, v in signals.items() if v >= SIGNAL_THRESHOLD]

                # Compute forward returns (this is what we're predicting)
                forward_returns = {}
                for window in EVAL_WINDOWS:
                    if j + window < len(close) and close_at_j > 0:
                        fwd_price = close[j + window]
                        forward_returns[window] = round(
                            (fwd_price - close_at_j) / close_at_j, 5
                        )
                    else:
                        forward_returns[window] = None

                # Skip if we don't have at least 30-day forward data
                if forward_returns.get(30) is None:
                    continue

                observation = {
                    "ticker": ticker,
                    "date": str(dates[j].date()) if hasattr(dates[j], 'date') else str(dates[j])[:10],
                    "price": round(close_at_j, 2),
                    "signals": signals,
                    "high_signals": high_signals,
                    "volatility_pct": round(vol_pct, 4) if vol_pct else None,
                    "forward_returns": forward_returns,
                }
                all_observations.append(observation)
                processed += 1

        except Exception as e:
            logger.debug(f"Backtest error for {ticker}: {e}")
            continue

    elapsed = time.time() - start_time
    logger.info(f"Backtest complete: {processed} observations in {elapsed:.0f}s")

    # ── Analyze results ───────────────────────────────────────────────
    analysis = analyze_backtest(all_observations)

    results = {
        "metadata": {
            "run_date": datetime.now().isoformat(),
            "tickers_tested": len(universe),
            "years": years,
            "sample_interval_days": sample_interval_days,
            "total_observations": processed,
            "elapsed_seconds": round(elapsed, 1),
        },
        "analysis": analysis,
        "observations_sample": all_observations[:100],  # Save sample for inspection
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {RESULTS_FILE}")
    return results


def analyze_backtest(observations: list[dict]) -> dict:
    """
    Analyze backtest observations to determine signal effectiveness.
    """
    if not observations:
        return {"error": "No observations to analyze"}

    analysis = {
        "overall": {},
        "individual_signals": {},
        "signal_combinations": {},
        "by_volatility": {},
    }

    SIGNAL_NAMES = ["volume_anomaly", "momentum", "relative_strength"]

    # ── Overall baseline ──────────────────────────────────────────────
    all_30d = [o["forward_returns"][30] for o in observations if o["forward_returns"].get(30) is not None]
    if all_30d:
        analysis["overall"] = {
            "count": len(all_30d),
            "avg_return_30d": round(np.mean(all_30d), 5),
            "median_return_30d": round(np.median(all_30d), 5),
            "win_rate_30d": round(sum(1 for r in all_30d if r > 0) / len(all_30d), 3),
            "std_30d": round(np.std(all_30d), 5),
        }

    # ── Individual signal analysis ────────────────────────────────────
    for sig in SIGNAL_NAMES:
        high = [o for o in observations if o["signals"].get(sig, 0) >= SIGNAL_THRESHOLD]
        low = [o for o in observations if o["signals"].get(sig, 0) < SIGNAL_THRESHOLD]

        for window in [30, 60, 90]:
            w_key = str(window)
            high_ret = [o["forward_returns"][window] for o in high if o["forward_returns"].get(window) is not None]
            low_ret = [o["forward_returns"][window] for o in low if o["forward_returns"].get(window) is not None]

            if high_ret and low_ret:
                edge = np.mean(high_ret) - np.mean(low_ret)
                analysis["individual_signals"][f"{sig}_{w_key}d"] = {
                    "high_count": len(high_ret),
                    "low_count": len(low_ret),
                    "high_avg_return": round(np.mean(high_ret), 5),
                    "low_avg_return": round(np.mean(low_ret), 5),
                    "edge": round(edge, 5),
                    "high_win_rate": round(sum(1 for r in high_ret if r > 0) / len(high_ret), 3),
                    "low_win_rate": round(sum(1 for r in low_ret if r > 0) / len(low_ret), 3),
                }

    # ── Signal combination analysis ───────────────────────────────────
    from itertools import combinations

    for size in [2, 3]:
        for combo in combinations(SIGNAL_NAMES, size):
            combo_key = "+".join(combo)
            matching = [
                o for o in observations
                if all(s in o.get("high_signals", []) for s in combo)
            ]

            if len(matching) < 10:
                continue

            for window in [30, 60]:
                returns = [
                    o["forward_returns"][window]
                    for o in matching
                    if o["forward_returns"].get(window) is not None
                ]

                if returns:
                    baseline_returns = [
                        o["forward_returns"][window]
                        for o in observations
                        if o["forward_returns"].get(window) is not None
                    ]

                    analysis["signal_combinations"][f"{combo_key}_{window}d"] = {
                        "count": len(returns),
                        "avg_return": round(np.mean(returns), 5),
                        "win_rate": round(sum(1 for r in returns if r > 0) / len(returns), 3),
                        "edge_vs_baseline": round(np.mean(returns) - np.mean(baseline_returns), 5),
                        "best": round(max(returns), 4),
                        "worst": round(min(returns), 4),
                    }

    # ── Volatility buckets ────────────────────────────────────────────
    vol_obs = [o for o in observations if o.get("volatility_pct") is not None]
    if vol_obs:
        for label, lo, hi in [("low_vol", 0, 0.025), ("mid_vol", 0.025, 0.045), ("high_vol", 0.045, 1.0)]:
            bucket = [o for o in vol_obs if lo <= o["volatility_pct"] < hi]
            if bucket:
                returns = [o["forward_returns"][30] for o in bucket if o["forward_returns"].get(30) is not None]
                if returns:
                    analysis["by_volatility"][label] = {
                        "count": len(returns),
                        "avg_return_30d": round(np.mean(returns), 5),
                        "win_rate_30d": round(sum(1 for r in returns if r > 0) / len(returns), 3),
                        "volatility_range": f"{lo*100:.1f}% - {hi*100:.1f}%",
                    }

    return analysis


def print_report(sector_breakdown: bool = False):
    """Print backtest results."""
    if not RESULTS_FILE.exists():
        print("No backtest results. Run: python backtester.py run")
        return

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    meta = results.get("metadata", {})
    analysis = results.get("analysis", {})

    print(f"\n{'='*65}")
    print(f"BACKTEST REPORT")
    print(f"{'='*65}")
    print(f"Run date: {meta.get('run_date', '?')[:10]}")
    print(f"Tickers tested: {meta.get('tickers_tested', 0)}")
    print(f"Period: {meta.get('years', 0)} years")
    print(f"Total observations: {meta.get('total_observations', 0):,}")

    # Overall baseline
    overall = analysis.get("overall", {})
    if overall:
        print(f"\nBASELINE (all observations, 30-day forward):")
        print(f"  Avg return: {overall.get('avg_return_30d', 0):+.2%}")
        print(f"  Win rate: {overall.get('win_rate_30d', 0):.1%}")
        print(f"  Std dev: {overall.get('std_30d', 0):.2%}")

    # Individual signals
    indiv = analysis.get("individual_signals", {})
    if indiv:
        print(f"\nINDIVIDUAL SIGNAL EDGES:")
        print(f"{'Signal':<35} {'Edge':>8} {'High Avg':>10} {'Low Avg':>10} {'High Win%':>10} {'n':>6}")
        print(f"{'─'*80}")
        for key in sorted(indiv.keys(), key=lambda k: indiv[k].get("edge", 0), reverse=True):
            d = indiv[key]
            print(
                f"  {key:<33} {d['edge']:>+7.2%} {d['high_avg_return']:>+9.2%} "
                f"{d['low_avg_return']:>+9.2%} {d['high_win_rate']:>9.1%} {d['high_count']:>6}"
            )

    # Signal combinations
    combos = analysis.get("signal_combinations", {})
    if combos:
        print(f"\nSIGNAL COMBINATIONS (sorted by edge vs baseline):")
        print(f"{'Combo':<45} {'Edge':>8} {'Avg Ret':>10} {'Win%':>8} {'n':>6}")
        print(f"{'─'*80}")
        sorted_combos = sorted(combos.items(), key=lambda x: x[1].get("edge_vs_baseline", 0), reverse=True)
        for key, d in sorted_combos[:15]:
            print(
                f"  {key:<43} {d['edge_vs_baseline']:>+7.2%} "
                f"{d['avg_return']:>+9.2%} {d['win_rate']:>7.1%} {d['count']:>6}"
            )

    # Volatility analysis
    vol = analysis.get("by_volatility", {})
    if vol:
        print(f"\nVOLATILITY BUCKET ANALYSIS (30-day forward):")
        for bucket_name, d in vol.items():
            print(
                f"  {bucket_name:<12} ({d['volatility_range']}): "
                f"avg {d['avg_return_30d']:+.2%}, "
                f"win {d['win_rate_30d']:.1%}, "
                f"n={d['count']}"
            )

    print()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Backtesting Engine")
    parser.add_argument(
        "command", choices=["run", "report"],
        help="run: execute backtest. report: show results.",
    )
    parser.add_argument("--years", type=int, default=2, help="Years of history (default: 2)")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to backtest")
    parser.add_argument("--sector", action="store_true", help="Show sector breakdown")

    args = parser.parse_args()

    if args.command == "run":
        run_backtest(tickers=args.tickers, years=args.years)
        print_report()
    elif args.command == "report":
        print_report(sector_breakdown=args.sector)


if __name__ == "__main__":
    main()
