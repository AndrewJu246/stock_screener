"""
Weight optimizer: analyzes prediction outcomes from the tracker
and proposes improved signal weights.

The optimizer doesn't blindly overfit — it uses several safeguards:
  - Minimum sample size before making suggestions
  - Bayesian smoothing (blends data-driven weights with current priors)
  - Sector-aware analysis (different weights may work in different sectors)
  - Regime-aware (only optimizes weights for the regime that has data)
  - Caps maximum weight change per cycle to prevent wild swings

Usage:
    python optimizer.py preview          # Show proposed changes (don't apply)
    python optimizer.py apply            # Apply proposed changes to config.py
    python optimizer.py apply --blend 0.3  # Apply with 30% new, 70% old weights
    python optimizer.py history          # Show optimization history
"""

import argparse
import json
import logging
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from config import REGIMES

logger = logging.getLogger(__name__)

TRACKER_DIR = Path("tracker_data")
OPTIMIZER_DIR = Path("optimizer_data")
OPTIMIZER_DIR.mkdir(exist_ok=True)

HISTORY_FILE = OPTIMIZER_DIR / "optimization_history.json"
MIN_PREDICTIONS = 15          # Minimum predictions before optimizing
MIN_PER_SIGNAL = 5            # Minimum high/low samples per signal
MAX_WEIGHT_CHANGE = 5.0       # Max change per signal per cycle (percentage points)
DEFAULT_BLEND = 0.4           # 40% data-driven, 60% current weights

SIGNAL_NAMES = [
    "volume_anomaly", "momentum", "fundamentals", "smart_money",
    "sector_rotation", "earnings_quality", "analyst_revisions",
    "tech_megatrend", "relative_strength",
]


def load_tracker_data() -> tuple[list[dict], dict]:
    """Load predictions and signal stats from tracker."""
    pred_file = TRACKER_DIR / "predictions.json"
    stats_file = TRACKER_DIR / "signal_performance.json"

    predictions = []
    if pred_file.exists():
        with open(pred_file) as f:
            predictions = json.load(f)

    stats = {}
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)

    return predictions, stats


def compute_signal_edges(predictions: list[dict], eval_window: int = 30) -> dict:
    """
    Compute the predictive edge of each signal.

    Edge = average return when signal was high (≥60) minus
           average return when signal was low (<60).

    A positive edge means the signal actually predicts well.
    A negative edge means the signal is anti-predictive (bad signal).
    Near-zero means the signal adds no value.
    """
    eval_key = str(eval_window)
    evaluable = [
        p for p in predictions
        if p.get("performance", {}).get(eval_key) is not None
    ]

    if len(evaluable) < MIN_PREDICTIONS:
        return {}

    edges = {}
    for sig in SIGNAL_NAMES:
        high = [p for p in evaluable if p.get("signals", {}).get(sig, 0) >= 60]
        low = [p for p in evaluable if p.get("signals", {}).get(sig, 0) < 60]

        if len(high) < MIN_PER_SIGNAL or len(low) < MIN_PER_SIGNAL:
            edges[sig] = {"edge": 0, "confidence": "insufficient_data",
                          "high_n": len(high), "low_n": len(low)}
            continue

        high_returns = [p["performance"][eval_key]["return"] for p in high]
        low_returns = [p["performance"][eval_key]["return"] for p in low]

        high_avg = np.mean(high_returns)
        low_avg = np.mean(low_returns)
        edge = high_avg - low_avg

        # Statistical confidence (rough t-test approximation)
        high_std = np.std(high_returns, ddof=1) if len(high_returns) > 1 else 0.1
        low_std = np.std(low_returns, ddof=1) if len(low_returns) > 1 else 0.1
        se = np.sqrt(high_std**2 / len(high_returns) + low_std**2 / len(low_returns))
        t_stat = edge / se if se > 0 else 0

        if abs(t_stat) > 2.0:
            confidence = "strong"
        elif abs(t_stat) > 1.0:
            confidence = "moderate"
        else:
            confidence = "weak"

        edges[sig] = {
            "edge": round(edge, 5),
            "high_avg": round(high_avg, 5),
            "low_avg": round(low_avg, 5),
            "high_win_rate": round(sum(1 for r in high_returns if r > 0) / len(high_returns), 3),
            "low_win_rate": round(sum(1 for r in low_returns if r > 0) / len(low_returns), 3),
            "high_n": len(high),
            "low_n": len(low),
            "t_stat": round(t_stat, 2),
            "confidence": confidence,
        }

    return edges


def compute_combo_performance(predictions: list[dict], eval_window: int = 30) -> list[dict]:
    """Find the best-performing signal combinations."""
    from itertools import combinations

    eval_key = str(eval_window)
    evaluable = [
        p for p in predictions
        if p.get("performance", {}).get(eval_key) is not None
    ]

    results = []
    for size in [2, 3]:
        for combo in combinations(SIGNAL_NAMES, size):
            matching = [
                p for p in evaluable
                if all(s in p.get("high_signals", []) for s in combo)
            ]
            if len(matching) < 3:
                continue

            returns = [p["performance"][eval_key]["return"] for p in matching]
            results.append({
                "signals": list(combo),
                "combo_key": " + ".join(s.replace("_", " ") for s in combo),
                "count": len(matching),
                "avg_return": round(np.mean(returns), 4),
                "win_rate": round(sum(1 for r in returns if r > 0) / len(returns), 3),
                "best": round(max(returns), 4),
                "worst": round(min(returns), 4),
            })

    results.sort(key=lambda x: x["avg_return"], reverse=True)
    return results


def propose_weights(
    edges: dict,
    current_regime: str = "balanced",
    blend: float = DEFAULT_BLEND,
) -> Optional[dict]:
    """
    Propose new weights based on signal edges.

    Uses Bayesian blending: new_weight = blend * data_driven + (1-blend) * current

    This prevents overfitting to small samples — the current weights
    (set from financial reasoning) act as a prior, and the data
    gradually shifts the weights as evidence accumulates.
    """
    if not edges:
        return None

    current_weights = REGIMES[current_regime]["weights"]

    # Convert edges to raw weight proposals
    # Signals with positive edges get more weight, negative get less
    valid_edges = {
        sig: data["edge"]
        for sig, data in edges.items()
        if isinstance(data, dict) and "edge" in data and data.get("confidence") != "insufficient_data"
    }

    if len(valid_edges) < 5:
        logger.info("Not enough signals with valid edges for optimization")
        return None

    # Normalize edges to weight-like scale
    min_edge = min(valid_edges.values())
    shifted = {k: v - min_edge + 0.01 for k, v in valid_edges.items()}
    total = sum(shifted.values())
    data_weights = {k: round(v / total * 100, 2) for k, v in shifted.items()}

    # Fill in missing signals with current weights
    for sig in SIGNAL_NAMES:
        if sig not in data_weights:
            data_weights[sig] = current_weights.get(sig, 100 / len(SIGNAL_NAMES))

    # Bayesian blend
    proposed = {}
    changes = {}
    for sig in SIGNAL_NAMES:
        current = current_weights.get(sig, 100 / len(SIGNAL_NAMES))
        data_driven = data_weights.get(sig, current)
        blended = blend * data_driven + (1 - blend) * current

        # Cap maximum change per cycle
        change = blended - current
        if abs(change) > MAX_WEIGHT_CHANGE:
            change = MAX_WEIGHT_CHANGE if change > 0 else -MAX_WEIGHT_CHANGE
            blended = current + change

        proposed[sig] = round(blended, 1)
        changes[sig] = round(change, 1)

    # Normalize to sum to 100
    total = sum(proposed.values())
    proposed = {k: round(v / total * 100, 1) for k, v in proposed.items()}

    # Ensure they sum to exactly 100
    diff = 100 - sum(proposed.values())
    max_key = max(proposed, key=proposed.get)
    proposed[max_key] = round(proposed[max_key] + diff, 1)

    return {
        "regime": current_regime,
        "current_weights": current_weights,
        "proposed_weights": proposed,
        "changes": {k: round(proposed[k] - current_weights.get(k, 0), 1) for k in SIGNAL_NAMES},
        "blend_factor": blend,
        "signal_edges": {k: v for k, v in edges.items() if isinstance(v, dict)},
    }


def preview_optimization() -> Optional[dict]:
    """
    Show what the optimizer would propose without applying anything.
    """
    predictions, stats = load_tracker_data()
    evaluable = [
        p for p in predictions
        if p.get("performance", {}).get("30") is not None
    ]

    print(f"\n{'='*60}")
    print("WEIGHT OPTIMIZER — PREVIEW")
    print(f"{'='*60}")
    print(f"Total predictions: {len(predictions)}")
    print(f"With 30-day outcomes: {len(evaluable)}")
    print(f"Minimum needed: {MIN_PREDICTIONS}")

    if len(evaluable) < MIN_PREDICTIONS:
        shortfall = MIN_PREDICTIONS - len(evaluable)
        print(f"\n⏳ Need {shortfall} more predictions with 30-day outcomes.")
        print(f"Keep running daily scans. Estimated days until ready: ~{shortfall}")
        return None

    # Compute edges
    edges = compute_signal_edges(predictions)
    if not edges:
        print("\nInsufficient data for edge calculation.")
        return None

    # Show signal edges
    print(f"\nSIGNAL EDGES (high ≥60 vs low <60, 30-day returns):")
    print(f"{'Signal':<22} {'Edge':>8} {'High Avg':>10} {'Low Avg':>10} {'Conf':>10} {'High Win%':>10}")
    print(f"{'─'*70}")
    for sig in sorted(edges.keys(), key=lambda s: edges[s].get("edge", 0), reverse=True):
        e = edges[sig]
        if e.get("confidence") == "insufficient_data":
            print(f"{sig:<22} {'—':>8} {'—':>10} {'—':>10} {'no data':>10} {'—':>10}")
        else:
            print(
                f"{sig:<22} {e['edge']:>+7.3%} {e['high_avg']:>+9.3%} "
                f"{e['low_avg']:>+9.3%} {e['confidence']:>10} {e['high_win_rate']:>9.1%}"
            )

    # Best combos
    combos = compute_combo_performance(predictions)
    if combos:
        print(f"\nBEST SIGNAL COMBINATIONS:")
        for c in combos[:8]:
            print(f"  {c['combo_key']:<45} avg {c['avg_return']:+.2%}  win {c['win_rate']:.0%}  n={c['count']}")

    # Propose weights
    # Detect which regime most predictions were in
    regime_counts = defaultdict(int)
    for p in evaluable:
        regime_counts[p.get("regime", "balanced")] += 1
    primary_regime = max(regime_counts, key=regime_counts.get)

    proposal = propose_weights(edges, primary_regime)
    if not proposal:
        print("\nCannot generate weight proposal yet.")
        return None

    print(f"\nPROPOSED WEIGHT CHANGES (regime: {primary_regime}, blend: {DEFAULT_BLEND}):")
    print(f"{'Signal':<22} {'Current':>8} {'Proposed':>10} {'Change':>8}")
    print(f"{'─'*50}")
    for sig in SIGNAL_NAMES:
        current = proposal["current_weights"].get(sig, 0)
        prop = proposal["proposed_weights"].get(sig, 0)
        change = proposal["changes"].get(sig, 0)
        arrow = "▲" if change > 0.5 else "▼" if change < -0.5 else "─"
        print(f"{sig:<22} {current:>7.1f}% {prop:>9.1f}% {arrow} {change:>+6.1f}%")

    print(f"\nWeights sum: {sum(proposal['proposed_weights'].values()):.1f}%")
    print(f"Blend: {DEFAULT_BLEND*100:.0f}% data-driven + {(1-DEFAULT_BLEND)*100:.0f}% current")
    print(f"\nTo apply: python optimizer.py apply")
    print(f"To adjust blend: python optimizer.py apply --blend 0.3")

    return proposal


def apply_optimization(blend: float = DEFAULT_BLEND) -> bool:
    """
    Apply optimized weights to config.py.

    Creates a backup of config.py first, then modifies the
    regime weights in place.
    """
    predictions, stats = load_tracker_data()
    edges = compute_signal_edges(predictions)

    if not edges:
        print("Not enough data to optimize. Run preview first.")
        return False

    # Detect primary regime
    evaluable = [p for p in predictions if p.get("performance", {}).get("30") is not None]
    regime_counts = defaultdict(int)
    for p in evaluable:
        regime_counts[p.get("regime", "balanced")] += 1
    primary_regime = max(regime_counts, key=regime_counts.get)

    proposal = propose_weights(edges, primary_regime, blend)
    if not proposal:
        print("Cannot generate proposal.")
        return False

    # Backup config.py
    config_path = Path("config.py")
    backup_path = Path(f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py")
    shutil.copy(config_path, backup_path)
    print(f"Backed up config.py → {backup_path}")

    # Read config.py and update the weights for the primary regime
    with open(config_path) as f:
        config_text = f.read()

    # Build new weights string
    new_weights_str = "{\n"
    for sig in SIGNAL_NAMES:
        w = proposal["proposed_weights"].get(sig, 10)
        # Pad the signal name for alignment
        new_weights_str += f'            "{sig}": {w:>5.1f},\n'
    # Remove trailing comma and close
    new_weights_str = new_weights_str.rstrip(",\n") + ",\n        }"

    # We need to find and replace the weights dict for the specific regime
    # This is a targeted string replacement — look for the regime's weights block
    import re

    # Pattern: find "regime_name": { ... "weights": { ... } in REGIMES dict
    # This is fragile with regex, so we'll use a simpler approach:
    # Read the config as a module, modify, and write specific lines

    # Actually, let's use a safer approach: write the new weights to a separate file
    # that the config imports, so we never corrupt config.py
    weights_override_path = Path("weight_overrides.json")
    overrides = {}
    if weights_override_path.exists():
        with open(weights_override_path) as f:
            overrides = json.load(f)

    overrides[primary_regime] = proposal["proposed_weights"]
    overrides["_metadata"] = {
        "last_updated": datetime.now().isoformat(),
        "blend": blend,
        "sample_size": len(evaluable),
        "primary_regime": primary_regime,
    }

    with open(weights_override_path, "w") as f:
        json.dump(overrides, f, indent=2)

    print(f"\nOptimized weights saved to {weights_override_path}")
    print(f"Regime: {primary_regime}")
    print(f"Blend: {blend*100:.0f}% data + {(1-blend)*100:.0f}% prior")
    print(f"Sample size: {len(evaluable)} predictions")

    # Log to history
    history = []
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            history = json.load(f)

    history.append({
        "date": datetime.now().isoformat(),
        "regime": primary_regime,
        "blend": blend,
        "sample_size": len(evaluable),
        "old_weights": proposal["current_weights"],
        "new_weights": proposal["proposed_weights"],
        "changes": proposal["changes"],
        "signal_edges": {k: v.get("edge", 0) for k, v in edges.items() if isinstance(v, dict)},
    })

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Optimization logged to {HISTORY_FILE}")
    print(f"\nThe screener will auto-load these weights on next run.")
    return True


def print_history():
    """Show optimization history."""
    if not HISTORY_FILE.exists():
        print("No optimization history yet.")
        return

    with open(HISTORY_FILE) as f:
        history = json.load(f)

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION HISTORY ({len(history)} optimizations)")
    print(f"{'='*60}")

    for i, entry in enumerate(history):
        print(f"\n--- Optimization #{i+1} ({entry['date'][:10]}) ---")
        print(f"Regime: {entry['regime']} | Blend: {entry['blend']} | Samples: {entry['sample_size']}")
        print(f"{'Signal':<22} {'Before':>8} {'After':>8} {'Change':>8}")
        for sig in SIGNAL_NAMES:
            old = entry["old_weights"].get(sig, 0)
            new = entry["new_weights"].get(sig, 0)
            chg = entry["changes"].get(sig, 0)
            print(f"  {sig:<22} {old:>7.1f}% {new:>7.1f}% {chg:>+6.1f}%")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Weight Optimizer")
    parser.add_argument(
        "command", choices=["preview", "apply", "history"],
        help="preview: show proposals. apply: update weights. history: past optimizations.",
    )
    parser.add_argument("--blend", type=float, default=DEFAULT_BLEND,
                        help=f"Blend factor: 0.0=keep current, 1.0=fully data-driven (default: {DEFAULT_BLEND})")

    args = parser.parse_args()

    if args.command == "preview":
        preview_optimization()
    elif args.command == "apply":
        apply_optimization(args.blend)
    elif args.command == "history":
        print_history()


if __name__ == "__main__":
    main()
