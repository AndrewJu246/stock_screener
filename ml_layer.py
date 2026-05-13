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
from datetime import datetime
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
    "tech_megatrend", "relative_strength",
]

# Check if sklearn is available
_SKLEARN_AVAILABLE = False
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score, train_test_split
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
    Prepare feature matrix (X) and labels (y) from prediction history.

    X = signal scores (9 features)
    y = 1 if stock went up over EVAL_WINDOW days, 0 if down
    """
    tracker_file = Path("tracker_data/predictions.json")
    if not tracker_file.exists():
        return None

    with open(tracker_file) as f:
        predictions = json.load(f)

    X = []
    y = []

    eval_key = str(EVAL_WINDOW)

    for p in predictions:
        perf = p.get("performance", {}).get(eval_key)
        if perf is None:
            continue

        # Feature vector: signal scores
        signals = p.get("signals", {})
        features = [signals.get(s, 0) for s in SIGNAL_FEATURES]

        # Label: 1 if positive return, 0 if negative
        forward_return = perf.get("return", 0) if isinstance(perf, dict) else perf
        label = 1 if forward_return > 0 else 0

        X.append(features)
        y.append(label)

    if len(X) < MIN_PREDICTIONS:
        return None

    return np.array(X), np.array(y)


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

    X, y = data
    logger.info(f"ML: training on {len(X)} samples ({sum(y)} positive, {len(y) - sum(y)} negative)")

    # Check class balance
    if sum(y) < MIN_POSITIVE_CLASS or (len(y) - sum(y)) < MIN_POSITIVE_CLASS:
        logger.warning("ML: class imbalance too extreme, skipping training")
        return None

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train gradient boosted classifier
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=10,
        random_state=42,
    )

    # Cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    logger.info(f"ML: cross-val accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Train on full training set
    model.fit(X_train, y_train)

    # Test set evaluation
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"ML: test accuracy: {test_accuracy:.3f}")

    # Feature importances
    importances = dict(zip(SIGNAL_FEATURES, model.feature_importances_))
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

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "samples": len(X),
        "positive_samples": int(sum(y)),
        "test_accuracy": round(test_accuracy, 4),
        "cv_accuracy": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
        "feature_importances": {k: round(v, 4) for k, v in sorted_imp},
        "features": SIGNAL_FEATURES,
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

        # Build feature vector
        features = [signal_scores.get(s, 0) for s in SIGNAL_FEATURES]
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
