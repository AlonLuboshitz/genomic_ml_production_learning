"""
Tests for the training module.

Run with:
    pytest tests/test_training.py -v
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from genomics_ml.models.train import train_model, _get_classifier


# ── Fixture: tiny synthetic dataset ──────────────────────────


@pytest.fixture
def tiny_data():
    """Return (X, y) — a 20-row, 3-feature dataset with binary target."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(20, 3), columns=["g1", "g2", "g3"])
    y = pd.Series(np.random.randint(0, 2, size=20), name="target")
    return X, y


@pytest.fixture
def minimal_config():
    """Return a minimal config dict for training (no YAML file needed)."""
    return {
        "data": {"test_size": 0.3, "random_state": 42},
        "preprocessing": {"impute_strategy": "mean", "scaler": "standard"},
        "model": {
            "type": "RandomForestClassifier",
            "params": {"n_estimators": 10, "max_depth": 3, "random_state": 42},
            "save_path": "/tmp/test_baseline.pkl",
        },
    }


# ── Tests ────────────────────────────────────────────────────


def test_train_model_returns_metrics_and_pipeline(tiny_data, minimal_config):
    X, y = tiny_data
    metrics, pipeline = train_model(X, y, config=minimal_config)
    assert isinstance(metrics, dict)
    for key in ("accuracy", "classification_report", "model_path"):
        assert key in metrics
    assert hasattr(pipeline, "predict")


def test_train_model_saves_loadable_model(tiny_data, minimal_config):
    X, y = tiny_data
    metrics, _ = train_model(X, y, config=minimal_config)
    loaded = joblib.load(metrics["model_path"])
    assert hasattr(loaded, "predict")
    preds = loaded.predict(X.iloc[:3])
    assert len(preds) == 3


def test_get_classifier_returns_estimator():
    config = {
        "model": {
            "type": "LogisticRegression",
            "params": {"max_iter": 100, "random_state": 42},
        }
    }
    clf = _get_classifier(config)
    assert hasattr(clf, "fit")
