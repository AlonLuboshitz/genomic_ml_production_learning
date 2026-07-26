"""
Central model utilities — all model load/save/predict/introspection in one place.

Both train.py, predict.py, and api/main.py should import from here
instead of calling joblib or sklearn directly.

Cloud storage support:
    Pass a config dict to save_model/load_model to enable S3 sync.
    Without config, behavior is identical to the original local-only version.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score

from genomics_ml.utils.storage import StorageManager

# ── Load / Save ───────────────────────────────────────────────────────


def load_model(model_path: str, config: dict | None = None) -> Any:
    """
    Load a fitted sklearn pipeline from disk.

    If config is provided with storage.backend == "s3", will attempt
    to download from S3 if the local file doesn't exist.

    Args:
        model_path: Path to the saved model file.
        config: Optional pipeline config dict for cloud storage.

    Returns:
        The deserialized model object.

    Raises:
        RuntimeError: If the model cannot be loaded.
    """
    storage_manager = StorageManager(config)
    model = storage_manager.load(model_path)
    return model


def save_model(pipeline: Any, model_path: str, config: dict | None = None) -> str:
    """
    Save a fitted pipeline to disk, optionally syncing to cloud.

    If config is provided with storage.backend == "s3", will also
    upload the model to S3 after saving locally.

    Args:
        pipeline: The fitted sklearn pipeline to save.
        model_path: Where to save locally (e.g., "models/baseline.pkl").
        config: Optional pipeline config dict for cloud storage.

    Returns:
        The local path where the model was saved.
    """
    storage_manager = StorageManager(config)
    storage_manager.save(pipeline, model_path)
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
    model_paths: list[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> list[dict[str, Any]]:
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
