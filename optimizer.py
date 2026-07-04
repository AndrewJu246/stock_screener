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
    "tech_megatrend", "relative_strength", "discovery_potential",
]

# Signals whose semantics changed at signal_version 2 (Jul 3, 2026):
# sector_rotation was redefined, smart_money's insider component and
# discovery_potential's institutional component were inert before the fix.
# Their v1 scores measure a different (or broken) quantity — edges for
# these signals only use rows stamped with the current version.
VERSION_SENSITIVE = {"sector_rotation", "smart_money", "discovery_potential"}
CURRENT_SIGNAL_VERSION = 2


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


def _perf_return(p: dict, eval_key: str):
    """Checkpoint return for edge analysis: excess over SPY when available
    (isolates stock picking from the market tape), else raw return."""
    perf = p.get("performance", {}).get(eval_key)
    if not isinstance(perf, dict):
        return perf
    if perf.get("excess_return") is not None:
        return perf["excess_return"]
    return perf.get("return")


def _cluster_by_ticker(preds: list[dict], eval_key: str) -> list[float]:
    """Collapse to one observation per ticker (mean of its returns).
    Repeated flags of the same stock are near-duplicates, not independent
    samples — leaving them uncollapsed inflates n and the t-stat."""
    by_ticker = defaultdict(list)
    for p in preds:
        r = _perf_return(p, eval_key)
        if r is not None:
            by_ticker[p.get("ticker", "?")].append(r)
    return [float(np.mean(rs)) for rs in by_ticker.values()]


def compute_signal_edges(predictions: list[dict], eval_window: int = 30) -> dict:
    """
    Compute the predictive edge of each signal.

    Edge = median excess return (vs SPY) across tickers where the signal
           was high (≥60) minus the same where it was low (<60).

    Median, because a handful of +60-90% outliers dominate any mean.
    Observations are clustered per ticker before comparing, and the
    t-stat is computed on the clustered values.

    A positive edge means the signal actually predicts well.
    A negative edge means the signal is anti-predictive (bad signal).
    Near-zero means the signal adds no value.
    """
    eval_key = str(eval_window)
    evaluable = [
        p for p in predictions
        if _perf_return(p, eval_key) is not None
    ]

    if len(evaluable) < MIN_PREDICTIONS:
        return {}

    edges = {}
    for sig in SIGNAL_NAMES:
        # Version-sensitive signals: pre-v2 scores mean something else —
        # mixing them in would average a live signal with a dead one
        if sig in VERSION_SENSITIVE:
            pool = [p for p in evaluable
                    if p.get("signal_version", 1) >= CURRENT_SIGNAL_VERSION]
        else:
            pool = evaluable
        high = [p for p in pool if p.get("signals", {}).get(sig, 0) >= 60]
        low = [p for p in pool if p.get("signals", {}).get(sig, 0) < 60]

        high_returns = _cluster_by_ticker(high, eval_key)
        low_returns = _cluster_by_ticker(low, eval_key)

        # Rank IC over the full score distribution (ticker-clustered) —
        # unlike the ≥60 split it uses every observation, works when scores
        # cluster on one side of 60, and can't hinge on a handful of rows
        score_by_ticker = defaultdict(list)
        ret_by_ticker = defaultdict(list)
        for p in pool:
            r = _perf_return(p, eval_key)
            if r is None:
                continue
            t = p.get("ticker", "?")
            score_by_ticker[t].append(p.get("signals", {}).get(sig, 0))
            ret_by_ticker[t].append(r)
        tickers = list(score_by_ticker)
        ic = None
        if len(tickers) >= 10:
            ic = _spearman(
                [float(np.mean(score_by_ticker[t])) for t in tickers],
                [float(np.mean(ret_by_ticker[t])) for t in tickers],
            )

        if len(high_returns) < MIN_PER_SIGNAL or len(low_returns) < MIN_PER_SIGNAL:
            edges[sig] = {"edge": 0, "confidence": "insufficient_data",
                          "high_n": len(high_returns), "low_n": len(low_returns),
                          "ic": round(ic, 4) if ic is not None else None,
                          "ic_n": len(tickers)}
            continue

        high_med = float(np.median(high_returns))
        low_med = float(np.median(low_returns))
        edge = high_med - low_med

        high_avg = float(np.mean(high_returns))
        low_avg = float(np.mean(low_returns))
        mean_edge = high_avg - low_avg

        # Statistical confidence: t-test on the ticker-clustered means
        high_std = np.std(high_returns, ddof=1) if len(high_returns) > 1 else 0.1
        low_std = np.std(low_returns, ddof=1) if len(low_returns) > 1 else 0.1
        se = np.sqrt(high_std**2 / len(high_returns) + low_std**2 / len(low_returns))
        t_stat = mean_edge / se if se > 0 else 0

        if abs(t_stat) > 2.0:
            confidence = "strong"
        elif abs(t_stat) > 1.0:
            confidence = "moderate"
        else:
            confidence = "weak"

        edges[sig] = {
            "edge": round(edge, 5),
            "ic": round(ic, 4) if ic is not None else None,
            "ic_n": len(tickers),
            "mean_edge": round(mean_edge, 5),
            "high_median": round(high_med, 5),
            "low_median": round(low_med, 5),
            "high_avg": round(high_avg, 5),
            "low_avg": round(low_avg, 5),
            "high_win_rate": round(sum(1 for r in high_returns if r > 0) / len(high_returns), 3),
            "low_win_rate": round(sum(1 for r in low_returns if r > 0) / len(low_returns), 3),
            "high_n": len(high_returns),
            "low_n": len(low_returns),
            "t_stat": round(float(t_stat), 2),
            "confidence": confidence,
        }

    return edges


def compute_combo_performance(predictions: list[dict], eval_window: int = 30) -> list[dict]:
    """Find the best-performing signal combinations."""
    from itertools import combinations

    eval_key = str(eval_window)
    evaluable = [
        p for p in predictions
        if _perf_return(p, eval_key) is not None
    ]

    # ~175 combos get tested here — with a tiny floor the "best" one is
    # nearly guaranteed to be noise. Require 15 unique tickers (repeated
    # flags of the same stock are not independent evidence).
    MIN_COMBO_TICKERS = 15

    results = []
    for size in [2, 3]:
        for combo in combinations(SIGNAL_NAMES, size):
            # Combos touching a version-sensitive signal only count rows
            # scored under the current signal semantics
            if VERSION_SENSITIVE & set(combo):
                pool = [p for p in evaluable
                        if p.get("signal_version", 1) >= CURRENT_SIGNAL_VERSION]
            else:
                pool = evaluable
            matching = [
                p for p in pool
                if all(s in p.get("high_signals", []) for s in combo)
            ]
            unique_tickers = len({p["ticker"] for p in matching})
            if unique_tickers < MIN_COMBO_TICKERS:
                continue

            returns = [_perf_return(p, eval_key) for p in matching]
            results.append({
                "signals": list(combo),
                "combo_key": " + ".join(s.replace("_", " ") for s in combo),
                "count": len(matching),
                "unique_tickers": unique_tickers,
                "avg_return": round(np.mean(returns), 4),
                "median_return": round(float(np.median(returns)), 4),
                "win_rate": round(sum(1 for r in returns if r > 0) / len(returns), 3),
                "best": round(max(returns), 4),
                "worst": round(min(returns), 4),
            })

    # Rank by median — one ARM-sized outlier shouldn't crown a combo
    results.sort(key=lambda x: x["median_return"], reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════════════
# Counterfactual weight evaluation (replay history under any weight set)
# ═══════════════════════════════════════════════════════════════════════

def _rank_avg_ties(a) -> np.ndarray:
    """Ranks 1..n with ties assigned their average rank."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def _spearman(x, y) -> Optional[float]:
    """Spearman rank correlation, numpy-only (no scipy dependency)."""
    if len(x) < 3:
        return None
    rx, ry = _rank_avg_ties(x), _rank_avg_ties(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _replay_composite(signals: dict, weights: dict) -> Optional[float]:
    """Composite score a prediction WOULD have had under `weights`.
    Renormalizes over the signals present in the stored snapshot."""
    acc, total_w = 0.0, 0.0
    for sig, w in weights.items():
        s = signals.get(sig)
        if s is None:
            continue
        acc += s * w
        total_w += w
    return acc / total_w if total_w > 0 else None


def evaluate_weight_sets(
    predictions: list[dict],
    weight_sets: dict[str, dict],
    eval_window: int = 30,
    regime: Optional[str] = None,
    min_per_day: int = 8,
) -> Optional[dict]:
    """
    Replay tracked predictions under candidate weight vectors and measure
    ranking skill: per-scan-date Spearman IC between the counterfactual
    composite score and the realized 30d excess return.

    Per-date IC is the standard quant metric here — each scan date is one
    quasi-independent cross-section (no ticker appears twice on a date),
    so averaging daily ICs sidesteps both ticker clustering and the
    market-tape component. Controls (once they accumulate outcomes) widen
    each day's cross-section and make the IC meaningfully harder to game.

    Caveat: replays raw weighted sums of stored scores — no_data
    renormalization, short-interest/FinBERT modifiers, and cap-tier
    adjustments aren't reconstructable from the snapshot. All weight sets
    face the same handicap, so the *comparison* stays fair.

    Returns per-set summary: mean/median daily IC, % positive days, n.
    """
    eval_key = str(eval_window)
    by_date = defaultdict(list)
    for p in predictions:
        if regime and p.get("regime") != regime:
            continue
        r = _perf_return(p, eval_key)
        if r is None:
            continue
        by_date[p.get("entry_date", "?")].append((p.get("signals", {}), r))

    daily_ics = {name: [] for name in weight_sets}
    days_used = 0
    rows_used = 0
    for date, rows in sorted(by_date.items()):
        if len(rows) < min_per_day:
            continue
        returns = [r for _, r in rows]
        day_ics = {}
        for name, weights in weight_sets.items():
            composites = [_replay_composite(sigs, weights) for sigs, _ in rows]
            paired = [(c, r) for c, r in zip(composites, returns) if c is not None]
            if len(paired) < min_per_day:
                day_ics = None
                break
            ic = _spearman([c for c, _ in paired], [r for _, r in paired])
            if ic is None:
                day_ics = None
                break
            day_ics[name] = ic
        # Only keep dates where every weight set got an IC — same sample
        if day_ics is None:
            continue
        for name, ic in day_ics.items():
            daily_ics[name].append(ic)
        days_used += 1
        rows_used += len(rows)

    if days_used < 5:
        return None

    results = {}
    for name, ics in daily_ics.items():
        ics_arr = np.array(ics)
        results[name] = {
            "mean_ic": round(float(ics_arr.mean()), 4),
            "median_ic": round(float(np.median(ics_arr)), 4),
            "pct_days_positive": round(float((ics_arr > 0).mean()), 3),
            "ic_std": round(float(ics_arr.std(ddof=1)), 4) if len(ics_arr) > 1 else None,
        }
    return {
        "per_set": results,
        "n_days": days_used,
        "n_rows": rows_used,
        "eval_window": eval_window,
        "regime": regime,
    }


def _print_evaluation(evaluation: dict):
    print(f"\nWEIGHT-SET REPLAY (per-scan-date Spearman IC, composite vs "
          f"{evaluation['eval_window']}d excess return):")
    print(f"Sample: {evaluation['n_days']} scan dates, {evaluation['n_rows']} rows"
          + (f", regime={evaluation['regime']}" if evaluation.get("regime") else ""))
    print(f"{'Weight set':<14} {'Mean IC':>9} {'Median IC':>11} {'Days IC>0':>11}")
    print(f"{'─'*48}")
    ranked = sorted(evaluation["per_set"].items(),
                    key=lambda x: x[1]["mean_ic"], reverse=True)
    for name, r in ranked:
        print(f"{name:<14} {r['mean_ic']:>+9.4f} {r['median_ic']:>+11.4f} "
              f"{r['pct_days_positive']:>10.0%}")


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

    # Data-driven basis: prefer rank IC (full score distribution,
    # ticker-clustered) over the binary ≥60 edge — the split can hinge on
    # a handful of low-side tickers and goes blind when a signal's scores
    # sit mostly on one side of 60. Fall back to edges if ICs are scarce.
    valid_ics = {
        sig: data["ic"]
        for sig, data in edges.items()
        if isinstance(data, dict) and data.get("ic") is not None
    }
    if len(valid_ics) >= 5:
        valid_edges = valid_ics
        basis = "spearman_ic"
    else:
        valid_edges = {
            sig: data["edge"]
            for sig, data in edges.items()
            if isinstance(data, dict) and "edge" in data
            and data.get("confidence") != "insufficient_data"
        }
        basis = "binary_edge"

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
        "basis": basis,
        "signal_edges": {k: v for k, v in edges.items() if isinstance(v, dict)},
    }


def _build_weight_sets(predictions: list[dict], primary_regime: str,
                       blend: float = DEFAULT_BLEND) -> dict[str, dict]:
    """Weight sets to compare in a replay: what runs today (current,
    including any applied overrides), the fresh proposal, the balanced
    baseline, and naive equal weights as a sanity floor."""
    from scorer import _get_weights  # reuse the exact override-loading logic

    sets = {
        "current": _get_weights(primary_regime),
        "balanced": REGIMES["balanced"]["weights"],
        "equal": {s: 10.0 for s in SIGNAL_NAMES},
    }
    edges = compute_signal_edges(predictions)
    proposal = propose_weights(edges, primary_regime, blend) if edges else None
    if proposal:
        sets["proposed"] = proposal["proposed_weights"]
    return sets


def run_evaluation(blend: float = DEFAULT_BLEND) -> Optional[dict]:
    """Replay history under current/proposed/balanced/equal weights."""
    predictions, _ = load_tracker_data()
    evaluable = [p for p in predictions if p.get("performance", {}).get("30") is not None]
    if len(evaluable) < MIN_PREDICTIONS:
        print(f"Need {MIN_PREDICTIONS - len(evaluable)} more predictions with 30d outcomes.")
        return None

    regime_counts = defaultdict(int)
    for p in evaluable:
        regime_counts[p.get("regime", "balanced")] += 1
    primary_regime = max(regime_counts, key=regime_counts.get)

    weight_sets = _build_weight_sets(predictions, primary_regime, blend)
    evaluation = evaluate_weight_sets(predictions, weight_sets, regime=primary_regime)
    if evaluation is None:
        print("Not enough scan dates with ≥8 evaluable rows for a replay yet.")
        return None
    _print_evaluation(evaluation)
    return evaluation


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
    print(f"\nSIGNAL EDGES (median, ticker-clustered, 30-day excess returns vs SPY):")
    print(f"{'Signal':<22} {'IC':>7} {'Edge':>8} {'High Med':>10} {'Low Med':>10} {'Conf':>10} {'Win%':>7} {'n hi/lo':>10}")
    print(f"{'─'*82}")
    def _ic_str(e):
        return f"{e['ic']:>+7.3f}" if e.get("ic") is not None else f"{'—':>7}"
    for sig in sorted(edges.keys(),
                      key=lambda s: (edges[s].get("ic") if edges[s].get("ic") is not None
                                     else edges[s].get("edge", 0)),
                      reverse=True):
        e = edges[sig]
        if e.get("confidence") == "insufficient_data":
            print(f"{sig:<22} {_ic_str(e)} {'—':>8} {'—':>10} {'—':>10} {'no data':>10} {'—':>7} "
                  f"{e.get('high_n', 0):>4}/{e.get('low_n', 0):<4}")
        else:
            print(
                f"{sig:<22} {_ic_str(e)} {e['edge']:>+7.3%} {e['high_median']:>+9.3%} "
                f"{e['low_median']:>+9.3%} {e['confidence']:>10} {e['high_win_rate']:>6.1%} "
                f"{e['high_n']:>5}/{e['low_n']:<4}"
            )

    # Best combos
    combos = compute_combo_performance(predictions)
    if combos:
        print(f"\nBEST SIGNAL COMBINATIONS:")
        for c in combos[:8]:
            print(f"  {c['combo_key']:<45} med {c['median_return']:+.2%}  avg {c['avg_return']:+.2%}  "
                  f"win {c['win_rate']:.0%}  n={c['count']} ({c['unique_tickers']} tickers)")

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

    print(f"\nPROPOSED WEIGHT CHANGES (regime: {primary_regime}, blend: {DEFAULT_BLEND}, "
          f"basis: {proposal.get('basis', '?')}):")
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

    # Counterfactual replay: would the proposal have ranked better?
    weight_sets = _build_weight_sets(predictions, primary_regime)
    evaluation = evaluate_weight_sets(predictions, weight_sets, regime=primary_regime)
    if evaluation:
        _print_evaluation(evaluation)

    print(f"\nTo apply: python optimizer.py apply")
    print(f"To adjust blend: python optimizer.py apply --blend 0.3")

    return proposal


def apply_optimization(blend: float = DEFAULT_BLEND, force: bool = False) -> bool:
    """
    Apply optimized weights to weight_overrides.json.

    Evidence gate: the proposal must beat the current weights on the
    counterfactual replay (mean daily IC) before it's applied — a weight
    change with no forward evidence behind it is exactly how overfitting
    sneaks in. Use --force to override.
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

    # ── Evidence gate: replay must favor the proposal ─────────────────
    weight_sets = _build_weight_sets(predictions, primary_regime, blend)
    evaluation = evaluate_weight_sets(predictions, weight_sets, regime=primary_regime)
    if evaluation and "proposed" in evaluation["per_set"]:
        _print_evaluation(evaluation)
        prop_ic = evaluation["per_set"]["proposed"]["mean_ic"]
        curr_ic = evaluation["per_set"]["current"]["mean_ic"]
        if prop_ic < curr_ic and not force:
            print(
                f"\n✗ BLOCKED: proposed weights score LOWER on the replay "
                f"(mean IC {prop_ic:+.4f} vs current {curr_ic:+.4f}).\n"
                f"  Applying them would be fitting noise. Let more outcomes "
                f"accumulate, or use --force if you have a reason."
            )
            return False
        print(f"\n✓ Replay gate passed: proposed IC {prop_ic:+.4f} vs current {curr_ic:+.4f}")
    elif not force:
        print(
            "\n✗ BLOCKED: not enough scan dates for a counterfactual replay — "
            "no evidence the new weights rank better. Use --force to override."
        )
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
        "replay": evaluation["per_set"] if evaluation else None,
        "forced": force,
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
        "command", choices=["preview", "apply", "evaluate", "history"],
        help="preview: show proposals. apply: update weights (gated on replay). "
             "evaluate: replay history under candidate weight sets. "
             "history: past optimizations.",
    )
    parser.add_argument("--blend", type=float, default=DEFAULT_BLEND,
                        help=f"Blend factor: 0.0=keep current, 1.0=fully data-driven (default: {DEFAULT_BLEND})")
    parser.add_argument("--force", action="store_true",
                        help="Apply weights even if the replay gate fails")

    args = parser.parse_args()

    if args.command == "preview":
        preview_optimization()
    elif args.command == "apply":
        apply_optimization(args.blend, force=args.force)
    elif args.command == "evaluate":
        run_evaluation(args.blend)
    elif args.command == "history":
        print_history()


if __name__ == "__main__":
    main()
