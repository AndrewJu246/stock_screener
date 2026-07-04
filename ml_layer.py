"""
ML signal layer.

Trains a gradient-boosted model on accumulated prediction outcomes
to find nonlinear signal interactions that the weighted average misses.

Auto-trigger: checks if enough data exists (200+ predictions with
30-day outcomes). If yes, trains and provides an ML-based score
as a 10th signal. If no, skips silently.

Uses scikit-learn's GradientBoostingClassifier (lightweight, no GPU needed).
Falls back gracefully if sklearn isn't installed.

Usage:
    python ml_layer.py status     # Check if ready to train
    python ml_layer.py train      # Force training (if enough data)
    python ml_layer.py report     # Show model performance
"""

import argparse
import json
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

ML_DIR = Path("ml_data")
ML_DIR.mkdir(exist_ok=True)

MODEL_FILE = ML_DIR / "signal_model.pkl"
MODEL_META_FILE = ML_DIR / "model_metadata.json"

MIN_PREDICTIONS = 200     # Need 200+ predictions with outcomes
MIN_POSITIVE_CLASS = 30   # Need at least 30 winning trades
EVAL_WINDOW = 30          # Predict 30-day forward returns

SIGNAL_FEATURES = [
    "volume_anomaly", "momentum", "fundamentals", "smart_money",
    "sector_rotation", "earnings_quality", "analyst_revisions",
    "tech_megatrend", "relative_strength", "discovery_potential",
]

# signal_version is an extra feature: sector_rotation/smart_money/
# discovery_potential scores mean different things before v2 (Jul 3, 2026),
# and the tree can learn to condition on the version instead of averaging
# a live signal with a dead one. At predict time it's always the current
# version, so v1-conditional branches simply never fire on new picks.
CURRENT_SIGNAL_VERSION = 2
FEATURE_NAMES = SIGNAL_FEATURES + ["signal_version"]

# Check if sklearn is available
_SKLEARN_AVAILABLE = False
try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, GroupKFold, GroupShuffleSplit
    from sklearn.metrics import classification_report, accuracy_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    logger.info(
        "scikit-learn not available — install with: pip install scikit-learn\n"
        "ML layer will be skipped."
    )


def check_data_readiness() -> dict:
    """
    Check if we have enough prediction data to train.
    """
    tracker_file = Path("tracker_data/predictions.json")
    if not tracker_file.exists():
        return {
            "ready": False,
            "total_predictions": 0,
            "with_outcomes": 0,
            "needed": MIN_PREDICTIONS,
            "message": "No prediction data yet. Run daily scans to accumulate data.",
        }

    with open(tracker_file) as f:
        predictions = json.load(f)

    total = len(predictions)
    with_outcomes = sum(
        1 for p in predictions
        if p.get("performance", {}).get(str(EVAL_WINDOW)) is not None
    )

    ready = with_outcomes >= MIN_PREDICTIONS and _SKLEARN_AVAILABLE
    shortfall = max(0, MIN_PREDICTIONS - with_outcomes)

    if not _SKLEARN_AVAILABLE:
        msg = "Install scikit-learn: pip install scikit-learn"
    elif shortfall > 0:
        est_days = shortfall // 5  # ~5 new predictions per day
        msg = f"Need {shortfall} more predictions with {EVAL_WINDOW}-day outcomes (~{est_days} days)"
    else:
        msg = "Ready to train!"

    return {
        "ready": ready,
        "sklearn_available": _SKLEARN_AVAILABLE,
        "total_predictions": total,
        "with_outcomes": with_outcomes,
        "needed": MIN_PREDICTIONS,
        "shortfall": shortfall,
        "message": msg,
    }


def prepare_training_data() -> Optional[tuple]:
    """
    Prepare feature matrix (X), labels (y), groups (tickers), and entry
    dates from prediction history.

    X = signal scores
    y = 1 if the stock beat SPY over EVAL_WINDOW days, 0 otherwise
    groups/dates power leak-free validation: the same ticker flagged on
    consecutive scans produces near-identical rows, so ticker groups must
    never straddle a train/test boundary.
    """
    tracker_file = Path("tracker_data/predictions.json")
    if not tracker_file.exists():
        return None

    with open(tracker_file) as f:
        predictions = json.load(f)

    X = []
    y = []
    groups = []
    dates = []

    eval_key = str(EVAL_WINDOW)

    for p in predictions:
        perf = p.get("performance", {}).get(eval_key)
        if perf is None:
            continue

        forward_return = perf.get("return", 0) if isinstance(perf, dict) else perf

        # Artifact guard: a real 30d loss beyond -75% doesn't survive the
        # quality gate — anything there is an unadjusted split or bad price
        if forward_return is None or forward_return < -0.75:
            continue

        # Feature vector: signal scores + signal-semantics version
        signals = p.get("signals", {})
        features = [signals.get(s, 0) for s in SIGNAL_FEATURES]
        features.append(p.get("signal_version", 1))

        # Label: 1 if the stock beat SPY over the window — raw direction
        # mostly measures the market (in a bull tape everything is "up")
        if isinstance(perf, dict) and perf.get("excess_return") is not None:
            label_return = perf["excess_return"]
        else:
            label_return = forward_return  # legacy rows without benchmark data
        label = 1 if label_return > 0 else 0

        X.append(features)
        y.append(label)
        groups.append(p.get("ticker", "?"))
        dates.append(p.get("entry_date", ""))

    if len(X) < MIN_PREDICTIONS:
        return None

    return np.array(X), np.array(y), np.array(groups), np.array(dates)


def train_model(force: bool = False) -> Optional[dict]:
    """
    Train the gradient boosted model on prediction data.

    Uses cross-validation to estimate performance.
    Only saves the model if it beats 55% accuracy.
    """
    if not _SKLEARN_AVAILABLE:
        logger.warning("scikit-learn not installed")
        return None

    readiness = check_data_readiness()
    if not readiness["ready"] and not force:
        logger.info(f"ML: {readiness['message']}")
        return None

    data = prepare_training_data()
    if data is None:
        logger.info("ML: insufficient training data")
        return None

    X, y, groups, dates = data
    logger.info(f"ML: training on {len(X)} samples ({sum(y)} positive, {len(y) - sum(y)} negative)")

    # Check class balance
    if sum(y) < MIN_POSITIVE_CLASS or (len(y) - sum(y)) < MIN_POSITIVE_CLASS:
        logger.warning("ML: class imbalance too extreme, skipping training")
        return None

    # Sort by entry date so the split is temporal: train on earlier scans,
    # test on later ones (a random split would leak duplicate ticker rows
    # into both sides and inflate accuracy)
    order = np.argsort(dates, kind="stable")
    X, y, groups, dates = X[order], y[order], groups[order], dates[order]

    split_at = int(len(X) * 0.8)
    # Don't cut a scan day in half
    while split_at < len(X) and dates[split_at] == dates[split_at - 1]:
        split_at += 1
    test_idx = np.arange(split_at, len(X))

    # Embargo: a train row entered <EVAL_WINDOW days before the boundary
    # has its label measured INSIDE the test period — that overlap leaks
    # test-period market moves into training. Drop those rows entirely.
    embargo_days = 0
    train_idx = np.arange(split_at)
    if split_at < len(X):
        try:
            boundary = datetime.strptime(str(dates[split_at])[:10], "%Y-%m-%d")
            embargo_cut = (boundary - timedelta(days=EVAL_WINDOW)).strftime("%Y-%m-%d")
            train_idx = np.array(
                [i for i in range(split_at) if str(dates[i])[:10] < embargo_cut],
                dtype=int,
            )
            embargo_days = EVAL_WINDOW
        except ValueError:
            pass  # unparseable dates — keep un-embargoed temporal split

    # Purge: drop test rows whose ticker already appears in training
    train_tickers = set(groups[train_idx])
    purged_test = np.array(
        [i for i in test_idx if groups[i] not in train_tickers], dtype=int
    )

    split_method = "temporal_purged_embargo" if embargo_days else "temporal_purged"
    if (
        len(purged_test) >= 20
        and len(np.unique(y[purged_test])) == 2
        and len(train_idx) >= 100
        and len(np.unique(y[train_idx])) == 2
    ):
        test_idx = purged_test
    else:
        # Purged/embargoed sets too small — fall back to a grouped random
        # split (leak-free by ticker; time overlap returns, so treat its
        # accuracy with more suspicion than the temporal split's)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups))
        split_method = "group_shuffle"
        embargo_days = 0

    X_train, y_train, g_train = X[train_idx], y[train_idx], groups[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    logger.info(
        f"ML: split={split_method}, train={len(X_train)}, test={len(X_test)} "
        f"({len(set(groups[test_idx]))} unique test tickers)"
    )

    # Gradient boosted base learner, wrapped in sigmoid calibration so
    # ml_score reads as a real probability (raw GBM scores cluster near
    # the extremes and overstate confidence)
    base_model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=10,
        random_state=42,
    )
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)

    # Cross-validation grouped by ticker so folds never share a stock
    n_folds = min(5, len(set(g_train)))
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=GroupKFold(n_splits=n_folds),
        groups=g_train, scoring="accuracy",
    )
    logger.info(f"ML: grouped cross-val accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Train on full training set
    model.fit(X_train, y_train)

    # Test set evaluation
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"ML: test accuracy: {test_accuracy:.3f}")

    # Feature importances (averaged over the calibration ensemble)
    imp = np.mean(
        [cc.estimator.feature_importances_ for cc in model.calibrated_classifiers_],
        axis=0,
    )
    importances = dict(zip(FEATURE_NAMES, imp))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    logger.info("ML: feature importances:")
    for feat, imp in sorted_imp:
        logger.info(f"  {feat:<22} {imp:.3f}")

    # Only save if model beats baseline (55%)
    if test_accuracy < 0.55:
        logger.warning(
            f"ML: model accuracy {test_accuracy:.3f} below threshold 0.55 — not saving. "
            f"Need more/better data."
        )
        return {
            "status": "below_threshold",
            "accuracy": test_accuracy,
            "cv_accuracy": cv_scores.mean(),
        }

    # Save model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    base_rate = max(np.mean(y_test), 1 - np.mean(y_test))
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "samples": len(X),
        "positive_samples": int(sum(y)),
        "split_method": split_method,
        "embargo_days": embargo_days,
        "calibration": "sigmoid_cv3",
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_base_rate": round(float(base_rate), 4),
        "test_accuracy": round(test_accuracy, 4),
        "cv_accuracy": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
        "feature_importances": {k: round(v, 4) for k, v in sorted_imp},
        "features": FEATURE_NAMES,
    }
    with open(MODEL_META_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"ML: model saved ({test_accuracy:.1%} accuracy)")
    return metadata


def predict_score(signal_scores: dict) -> Optional[dict]:
    """
    Use the trained model to predict whether a stock will go up.

    Returns: {
        "ml_score": float (0-100, probability of positive return),
        "ml_prediction": "bullish" | "bearish",
        "model_accuracy": float,
    }
    """
    if not MODEL_FILE.exists():
        # Try to train if enough data exists
        readiness = check_data_readiness()
        if readiness["ready"]:
            train_model()

    if not MODEL_FILE.exists():
        return None

    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)

        with open(MODEL_META_FILE) as f:
            metadata = json.load(f)

        # Build feature vector — live picks are always the current version
        features = [signal_scores.get(s, 0) for s in SIGNAL_FEATURES]
        features.append(CURRENT_SIGNAL_VERSION)
        X = np.array([features])

        # Predict probability
        proba = model.predict_proba(X)[0]
        # proba[1] = probability of positive return
        ml_score = round(proba[1] * 100, 1)

        return {
            "ml_score": ml_score,
            "ml_prediction": "bullish" if ml_score > 55 else "bearish" if ml_score < 45 else "neutral",
            "model_accuracy": metadata.get("test_accuracy", 0),
            "model_date": metadata.get("trained_at", "")[:10],
        }

    except Exception as e:
        logger.debug(f"ML prediction failed: {e}")
        return None


def auto_check_and_train():
    """
    Called during each scan. Checks if:
    1. We have enough data but no model → train
    2. We have a model but 50+ new predictions since last train → retrain
    """
    readiness = check_data_readiness()

    if not readiness["ready"]:
        return

    if not MODEL_FILE.exists():
        logger.info("ML: enough data accumulated — training initial model")
        train_model()
        return

    # Check if model is stale (50+ new predictions since last train)
    if MODEL_META_FILE.exists():
        with open(MODEL_META_FILE) as f:
            meta = json.load(f)
        samples_at_train = meta.get("samples", 0)
        current_samples = readiness["with_outcomes"]
        if current_samples - samples_at_train >= 50:
            logger.info(f"ML: {current_samples - samples_at_train} new predictions — retraining")
            train_model()


def print_status():
    readiness = check_data_readiness()
    print(f"\n{'='*50}")
    print("ML LAYER STATUS")
    print(f"{'='*50}")
    print(f"scikit-learn: {'✓ installed' if _SKLEARN_AVAILABLE else '✗ not installed'}")
    print(f"Predictions total: {readiness['total_predictions']}")
    print(f"With {EVAL_WINDOW}-day outcomes: {readiness['with_outcomes']}")
    print(f"Needed for training: {readiness['needed']}")
    print(f"Status: {readiness['message']}")

    if MODEL_FILE.exists() and MODEL_META_FILE.exists():
        with open(MODEL_META_FILE) as f:
            meta = json.load(f)
        print(f"\nTrained model:")
        print(f"  Date: {meta.get('trained_at', '?')[:10]}")
        print(f"  Accuracy: {meta.get('test_accuracy', 0):.1%}")
        print(f"  CV accuracy: {meta.get('cv_accuracy', 0):.1%} (±{meta.get('cv_std', 0):.1%})")
        print(f"  Trained on: {meta.get('samples', 0)} samples")
        print(f"  Split: {meta.get('split_method', '?')} | test n={meta.get('test_size', '?')} "
              f"| base rate {meta.get('test_base_rate', 0):.1%}")
        print(f"\n  Feature importances:")
        for feat, imp in meta.get("feature_importances", {}).items():
            bar = "█" * int(imp * 50)
            print(f"    {feat:<22} {imp:.3f} {bar}")
    print()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="ML Signal Layer")
    parser.add_argument("command", choices=["status", "train", "report"])
    args = parser.parse_args()

    if args.command == "status":
        print_status()
    elif args.command == "train":
        result = train_model(force=True)
        if result:
            print_status()
        else:
            print("Training failed or insufficient data.")
    elif args.command == "report":
        print_status()


if __name__ == "__main__":
    main()
