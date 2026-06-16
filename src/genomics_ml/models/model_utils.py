"""
Central model utilities — all model load/save/predict/introspection in one place.

Both train.py, predict.py, and api/main.py should import from here
instead of calling joblib or sklearn directly.
"""

import joblib
import os
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score


# ── Load / Save ───────────────────────────────────────────────────────


def load_model(model_path: str) -> Any:
    """Load a fitted sklearn pipeline from disk."""
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Unable to load model from {model_path}: {e}")
    return model


def save_model(pipeline: Any, model_path: str) -> str:
    """Save a fitted pipeline to disk, creating directories if needed."""
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(pipeline, model_path)
    return model_path


# ── Introspection ─────────────────────────────────────────────────────


def get_model_type(model: Any) -> str:
    """Extract the classifier class name from a Pipeline."""
    return type(model.named_steps["clf"]).__name__


def get_n_features(model: Any) -> int:
    """Extract the number of features the model expects."""
    # The scaler (if present) stores n_features_in_; fall back to classifier
    step = model.named_steps.get("scaler") or model.named_steps["clf"]
    return step.n_features_in_


# ── Prediction ────────────────────────────────────────────────────────


def predict(model: Any, X: np.ndarray) -> np.ndarray:
    """Return predicted class labels."""
    y_pred = model.predict(X)
    if len(X) != len(y_pred):
        raise ValueError("Prediction length mismatch with input")
    return y_pred


def predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Return class probabilities."""
    y_prob = model.predict_proba(X)
    if len(X) != len(y_prob):
        raise ValueError("Probability length mismatch with input")
    return y_prob


# ── Comparison ────────────────────────────────────────────────────────


def compare_models(
    model_paths: List[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> List[Dict[str, Any]]:
    """Load multiple models, score them, return ranking sorted by accuracy."""
    results = []
    for path in model_paths:
        model = load_model(path)
        y_pred = predict(model, X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append(
            {
                "model_path": path,
                "model_type": get_model_type(model),
                "accuracy": round(acc, 4),
            }
        )
    results.sort(key=lambda r: r["accuracy"], reverse=True)
    return results
