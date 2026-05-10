"""
Prediction tracker: logs every recommendation, tracks what actually
happened, and analyzes which signals/combinations are predictive.

This is the foundation for the model's self-improvement loop.

Usage:
    # Automatically called after each scan (integrated into screener.py)
    # Or manually:
    python tracker.py log               # Log latest scan results as predictions
    python tracker.py update            # Update all active predictions with current prices
    python tracker.py report            # Show performance report
    python tracker.py report --detail   # Detailed signal-level analysis
"""

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

TRACKER_DIR = Path("tracker_data")
TRACKER_DIR.mkdir(exist_ok=True)

PREDICTIONS_FILE = TRACKER_DIR / "predictions.json"
SIGNAL_STATS_FILE = TRACKER_DIR / "signal_performance.json"

# Checkpoints: we evaluate performance at these intervals
EVAL_DAYS = [7, 14, 30, 60, 90]


# ═══════════════════════════════════════════════════════════════════════
# Data persistence
# ═══════════════════════════════════════════════════════════════════════

def _load_predictions() -> list[dict]:
    if PREDICTIONS_FILE.exists():
        with open(PREDICTIONS_FILE) as f:
            return json.load(f)
    return []


def _save_predictions(predictions: list[dict]):
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(predictions, f, indent=2, default=str)


def _load_signal_stats() -> dict:
    if SIGNAL_STATS_FILE.exists():
        with open(SIGNAL_STATS_FILE) as f:
            return json.load(f)
    return {"signal_combos": {}, "individual_signals": {}, "last_updated": None}


def _save_signal_stats(stats: dict):
    with open(SIGNAL_STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════
# Logging predictions
# ═══════════════════════════════════════════════════════════════════════

def log_predictions(scan_results: dict) -> int:
    """
    Log the top candidates from a scan as predictions to track.

    Called automatically after each scan. Only logs NEW predictions
    (won't duplicate if you run the scan twice on the same day).

    Returns: number of new predictions logged.
    """
    existing = _load_predictions()
    existing_keys = {
        (p["ticker"], p["entry_date"]) for p in existing
    }

    candidates = scan_results.get("candidates", [])
    summary = scan_results.get("summary", {})
    scan_date = summary.get("scan_date", datetime.now().isoformat())
    date_str = scan_date[:10]  # YYYY-MM-DD

    new_count = 0
    for c in candidates:
        key = (c["ticker"], date_str)
        if key in existing_keys:
            continue

        prediction = {
            "id": f"{c['ticker']}_{date_str}",
            "ticker": c["ticker"],
            "entry_date": date_str,
            "entry_price": c.get("current_price", 0),
            "composite_score": c.get("composite_score", 0),
            "rank": c.get("rank", 0),
            "regime": c.get("regime", "unknown"),
            "sector": c.get("sector", "Unknown"),
            "industry": c.get("industry", "Unknown"),

            # Full signal snapshot
            "signals": {
                name: sig.get("score", 0)
                for name, sig in c.get("signals", {}).items()
            },
            # Which signals were "high" (≥60)
            "high_signals": [
                name for name, sig in c.get("signals", {}).items()
                if isinstance(sig, dict) and sig.get("score", 0) >= 60
            ],

            "strategy": c.get("strategy", {}).get("strategy", "unknown"),
            "confidence": c.get("confidence", {}).get("confidence", "unknown"),

            # Performance tracking (filled in by update_predictions)
            "status": "active",
            "performance": {str(d): None for d in EVAL_DAYS},
            "peak_return": 0,
            "trough_return": 0,
            "current_price": c.get("current_price", 0),
            "current_return": 0,
            "last_updated": date_str,
            "days_tracked": 0,
        }

        existing.append(prediction)
        existing_keys.add(key)
        new_count += 1

    _save_predictions(existing)
    logger.info(f"Tracker: logged {new_count} new predictions (total: {len(existing)})")
    return new_count


# ═══════════════════════════════════════════════════════════════════════
# Updating predictions with current prices
# ═══════════════════════════════════════════════════════════════════════

def update_predictions() -> dict:
    """
    Update all active predictions with current prices.
    Fills in performance checkpoints (7d, 14d, 30d, 60d, 90d).

    Call this daily (or whenever you run the screener).

    Returns: summary of updates.
    """
    predictions = _load_predictions()
    if not predictions:
        logger.info("Tracker: no predictions to update")
        return {"updated": 0, "completed": 0}

    # Import here to avoid circular dependency
    from data_pipeline import get_price_history

    active = [p for p in predictions if p["status"] == "active"]
    today = datetime.now().date()

    updated = 0
    completed = 0

    for pred in active:
        ticker = pred["ticker"]
        entry_date = datetime.strptime(pred["entry_date"], "%Y-%m-%d").date()
        entry_price = pred["entry_price"]

        if entry_price <= 0:
            continue

        days_since_entry = (today - entry_date).days
        pred["days_tracked"] = days_since_entry

        # Get current price
        try:
            hist = get_price_history(ticker, period="6mo")
            if hist is None or hist.empty:
                continue

            close = hist["Close"]
            current_price = float(close.iloc[-1])
            pred["current_price"] = current_price
            pred["current_return"] = round(
                (current_price - entry_price) / entry_price, 4
            )
            pred["last_updated"] = str(today)

            # Track peak and trough
            # Look at all prices since entry
            entry_ts = entry_date.isoformat()
            mask = hist.index >= entry_ts
            if mask.any():
                prices_since = close[mask]
                peak_price = float(prices_since.max())
                trough_price = float(prices_since.min())
                pred["peak_return"] = round(
                    (peak_price - entry_price) / entry_price, 4
                )
                pred["trough_return"] = round(
                    (trough_price - entry_price) / entry_price, 4
                )

            # Fill in checkpoint performance
            for eval_d in EVAL_DAYS:
                checkpoint_key = str(eval_d)
                if pred["performance"].get(checkpoint_key) is not None:
                    continue  # Already filled

                if days_since_entry >= eval_d:
                    # Find the price on the checkpoint date
                    target_date = entry_date + timedelta(days=eval_d)
                    # Find nearest trading day
                    nearest = hist.index.asof(str(target_date))
                    if nearest is not None and str(nearest) != "NaT":
                        checkpoint_price = float(close.loc[nearest])
                        checkpoint_return = round(
                            (checkpoint_price - entry_price) / entry_price, 4
                        )
                        pred["performance"][checkpoint_key] = {
                            "price": checkpoint_price,
                            "return": checkpoint_return,
                            "date": str(target_date),
                        }

            updated += 1

            # Mark as completed after 90 days
            if days_since_entry >= 90:
                pred["status"] = "completed"
                completed += 1

        except Exception as e:
            logger.debug(f"Tracker: failed to update {ticker}: {e}")
            continue

    _save_predictions(predictions)

    # Recalculate signal stats whenever we update
    if updated > 0:
        _update_signal_stats(predictions)

    result = {
        "updated": updated,
        "completed": completed,
        "total_active": len([p for p in predictions if p["status"] == "active"]),
        "total_completed": len([p for p in predictions if p["status"] == "completed"]),
    }
    logger.info(
        f"Tracker: updated {updated} predictions, {completed} newly completed, "
        f"{result['total_active']} still active"
    )
    return result


# ═══════════════════════════════════════════════════════════════════════
# Signal performance analysis
# ═══════════════════════════════════════════════════════════════════════

def _update_signal_stats(predictions: list[dict]):
    """
    Analyze which signals and signal combinations have been predictive.
    This is the core of the self-improvement loop.
    """
    # Only analyze predictions with at least 30-day data
    evaluable = [
        p for p in predictions
        if p["performance"].get("30") is not None
    ]

    if len(evaluable) < 5:
        logger.info("Tracker: need at least 5 predictions with 30-day data for analysis")
        return

    stats = {
        "individual_signals": {},
        "signal_combos": {},
        "sector_performance": {},
        "strategy_performance": {},
        "regime_performance": {},
        "sample_size": len(evaluable),
        "last_updated": str(datetime.now().date()),
    }

    SIGNAL_NAMES = [
        "volume_anomaly", "momentum", "fundamentals", "smart_money",
        "sector_rotation", "earnings_quality", "analyst_revisions",
        "tech_megatrend", "relative_strength",
    ]

    # ── Individual signal analysis ────────────────────────────────────
    for sig_name in SIGNAL_NAMES:
        # Split predictions into "high signal" (≥60) vs "low signal" (<60)
        high = [p for p in evaluable if p["signals"].get(sig_name, 0) >= 60]
        low = [p for p in evaluable if p["signals"].get(sig_name, 0) < 60]

        high_returns = [p["performance"]["30"]["return"] for p in high if p["performance"]["30"]]
        low_returns = [p["performance"]["30"]["return"] for p in low if p["performance"]["30"]]

        stats["individual_signals"][sig_name] = {
            "high_count": len(high),
            "low_count": len(low),
            "high_avg_return_30d": round(np.mean(high_returns), 4) if high_returns else None,
            "low_avg_return_30d": round(np.mean(low_returns), 4) if low_returns else None,
            "high_win_rate_30d": round(
                sum(1 for r in high_returns if r > 0) / len(high_returns), 3
            ) if high_returns else None,
            "edge": round(
                (np.mean(high_returns) if high_returns else 0)
                - (np.mean(low_returns) if low_returns else 0),
                4,
            ) if high_returns and low_returns else None,
        }

    # ── Signal combination analysis ───────────────────────────────────
    # Check every pair and triple of high signals
    from itertools import combinations
    for combo_size in [2, 3]:
        for combo in combinations(SIGNAL_NAMES, combo_size):
            combo_key = "+".join(sorted(combo))
            matching = [
                p for p in evaluable
                if all(s in p.get("high_signals", []) for s in combo)
            ]

            if len(matching) < 3:  # Need minimum sample
                continue

            returns = [
                p["performance"]["30"]["return"]
                for p in matching
                if p["performance"]["30"]
            ]

            if returns:
                stats["signal_combos"][combo_key] = {
                    "count": len(matching),
                    "avg_return_30d": round(np.mean(returns), 4),
                    "win_rate_30d": round(sum(1 for r in returns if r > 0) / len(returns), 3),
                    "best": round(max(returns), 4),
                    "worst": round(min(returns), 4),
                }

    # ── Sector performance ────────────────────────────────────────────
    sector_groups = defaultdict(list)
    for p in evaluable:
        r = p["performance"]["30"]["return"] if p["performance"]["30"] else None
        if r is not None:
            sector_groups[p.get("sector", "Unknown")].append(r)

    for sector, returns in sector_groups.items():
        if len(returns) >= 2:
            stats["sector_performance"][sector] = {
                "count": len(returns),
                "avg_return_30d": round(np.mean(returns), 4),
                "win_rate_30d": round(sum(1 for r in returns if r > 0) / len(returns), 3),
            }

    # ── Strategy performance ──────────────────────────────────────────
    for strategy in ["long_term_hold", "short_term_momentum"]:
        matching = [
            p for p in evaluable
            if p.get("strategy") == strategy and p["performance"]["30"]
        ]
        returns = [p["performance"]["30"]["return"] for p in matching]
        if returns:
            stats["strategy_performance"][strategy] = {
                "count": len(returns),
                "avg_return_30d": round(np.mean(returns), 4),
                "win_rate_30d": round(sum(1 for r in returns if r > 0) / len(returns), 3),
            }

    # ── Regime performance ────────────────────────────────────────────
    for regime in ["bull", "bear", "early_recovery", "balanced"]:
        matching = [
            p for p in evaluable
            if p.get("regime") == regime and p["performance"]["30"]
        ]
        returns = [p["performance"]["30"]["return"] for p in matching]
        if returns:
            stats["regime_performance"][regime] = {
                "count": len(returns),
                "avg_return_30d": round(np.mean(returns), 4),
                "win_rate_30d": round(sum(1 for r in returns if r > 0) / len(returns), 3),
            }

    _save_signal_stats(stats)
    logger.info(f"Tracker: signal stats updated ({len(evaluable)} evaluable predictions)")


# ═══════════════════════════════════════════════════════════════════════
# Weight recommendations
# ═══════════════════════════════════════════════════════════════════════

def suggest_weight_updates() -> Optional[dict]:
    """
    Based on accumulated signal performance data, suggest
    updates to the regime weights in config.py.

    Returns None if not enough data yet.
    """
    stats = _load_signal_stats()

    if stats.get("sample_size", 0) < 20:
        logger.info(
            f"Tracker: need 20+ evaluated predictions for weight suggestions "
            f"(currently {stats.get('sample_size', 0)})"
        )
        return None

    individual = stats.get("individual_signals", {})

    # Rank signals by their "edge" (high_signal avg return - low_signal avg return)
    edges = {}
    for sig_name, sig_stats in individual.items():
        edge = sig_stats.get("edge")
        if edge is not None:
            edges[sig_name] = edge

    if not edges:
        return None

    # Normalize edges to sum to 100 for weight suggestions
    min_edge = min(edges.values())
    shifted = {k: v - min_edge + 0.01 for k, v in edges.items()}  # Shift to positive
    total = sum(shifted.values())
    suggested_weights = {k: round(v / total * 100, 1) for k, v in shifted.items()}

    # Find best signal combos
    combos = stats.get("signal_combos", {})
    best_combos = sorted(
        combos.items(),
        key=lambda x: x[1].get("avg_return_30d", 0),
        reverse=True,
    )[:5]

    return {
        "suggested_weights": suggested_weights,
        "signal_edges": edges,
        "best_combos": [
            {"signals": k, **v} for k, v in best_combos
        ],
        "sample_size": stats["sample_size"],
        "note": (
            "These are data-driven suggestions based on actual prediction outcomes. "
            "Review before applying — small sample sizes can be misleading."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════

def print_report(detailed: bool = False):
    """Print a performance report to the console."""
    predictions = _load_predictions()
    stats = _load_signal_stats()

    if not predictions:
        print("No predictions tracked yet. Run a scan first.")
        return

    active = [p for p in predictions if p["status"] == "active"]
    completed = [p for p in predictions if p["status"] == "completed"]
    total = len(predictions)

    print(f"\n{'='*60}")
    print(f"PREDICTION TRACKER REPORT")
    print(f"{'='*60}")
    print(f"Total predictions: {total}")
    print(f"Active: {len(active)} | Completed (90d+): {len(completed)}")

    # Active predictions summary
    if active:
        returns = [p["current_return"] for p in active if p["current_return"] != 0]
        if returns:
            avg_ret = np.mean(returns)
            win_rate = sum(1 for r in returns if r > 0) / len(returns)
            print(f"\nActive predictions:")
            print(f"  Avg return: {avg_ret:+.2%}")
            print(f"  Win rate: {win_rate:.1%}")
            print(f"  Best: {max(returns):+.2%}")
            print(f"  Worst: {min(returns):+.2%}")

    # Show each active prediction
    print(f"\n{'─'*60}")
    print(f"{'Ticker':<8} {'Days':<6} {'Return':>8} {'Peak':>8} {'Trough':>8} {'Strategy':<18} {'Conf':<8}")
    print(f"{'─'*60}")
    for p in sorted(active, key=lambda x: x["current_return"], reverse=True):
        print(
            f"{p['ticker']:<8} {p['days_tracked']:<6} "
            f"{p['current_return']:>+7.2%} {p['peak_return']:>+7.2%} "
            f"{p['trough_return']:>+7.2%} {p['strategy']:<18} {p['confidence']:<8}"
        )

    # Checkpoint performance
    for d in EVAL_DAYS:
        has_data = [
            p for p in predictions
            if p["performance"].get(str(d)) is not None
        ]
        if has_data:
            returns = [p["performance"][str(d)]["return"] for p in has_data]
            wr = sum(1 for r in returns if r > 0) / len(returns)
            print(f"\n{d}-day checkpoint ({len(has_data)} predictions):")
            print(f"  Avg return: {np.mean(returns):+.2%} | Win rate: {wr:.1%}")

    # Signal performance (if detailed)
    if detailed and stats.get("individual_signals"):
        print(f"\n{'='*60}")
        print("SIGNAL PERFORMANCE (when signal ≥60 vs <60)")
        print(f"{'='*60}")
        print(f"{'Signal':<22} {'High Avg':>10} {'Low Avg':>10} {'Edge':>8} {'Win Rate':>10}")
        print(f"{'─'*60}")
        for sig_name, sig_data in sorted(
            stats["individual_signals"].items(),
            key=lambda x: x[1].get("edge", 0) or 0,
            reverse=True,
        ):
            h_avg = sig_data.get("high_avg_return_30d")
            l_avg = sig_data.get("low_avg_return_30d")
            edge = sig_data.get("edge")
            wr = sig_data.get("high_win_rate_30d")
            print(
                f"{sig_name:<22} "
                f"{h_avg:>+9.2%} " if h_avg is not None else f"{'N/A':>10} ",
                f"{l_avg:>+9.2%} " if l_avg is not None else f"{'N/A':>10} ",
                f"{edge:>+7.2%} " if edge is not None else f"{'N/A':>8} ",
                f"{wr:>9.1%}" if wr is not None else f"{'N/A':>10}",
            )

        # Best signal combos
        combos = stats.get("signal_combos", {})
        if combos:
            print(f"\nBEST SIGNAL COMBINATIONS (30-day):")
            best = sorted(combos.items(), key=lambda x: x[1].get("avg_return_30d", 0), reverse=True)[:10]
            for combo_name, combo_data in best:
                print(
                    f"  {combo_name}: "
                    f"avg {combo_data['avg_return_30d']:+.2%}, "
                    f"win rate {combo_data['win_rate_30d']:.1%}, "
                    f"n={combo_data['count']}"
                )

    # Weight suggestions
    suggestions = suggest_weight_updates()
    if suggestions:
        print(f"\n{'='*60}")
        print("SUGGESTED WEIGHT UPDATES (based on actual performance)")
        print(f"{'='*60}")
        for sig, weight in sorted(
            suggestions["suggested_weights"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            edge = suggestions["signal_edges"].get(sig, 0)
            print(f"  {sig:<22} → {weight:>5.1f}%  (edge: {edge:+.4f})")

    print()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Prediction Tracker")
    parser.add_argument(
        "command",
        choices=["log", "update", "report"],
        help="log: log latest scan. update: fetch current prices. report: show performance.",
    )
    parser.add_argument("--detail", action="store_true", help="Detailed signal analysis")

    args = parser.parse_args()

    if args.command == "log":
        results_file = Path("output/screener_results.json")
        if not results_file.exists():
            print("No scan results found. Run screener.py first.")
            return
        with open(results_file) as f:
            scan_results = json.load(f)
        log_predictions(scan_results)

    elif args.command == "update":
        update_predictions()

    elif args.command == "report":
        print_report(detailed=args.detail)


if __name__ == "__main__":
    main()
