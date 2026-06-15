"""
Tests for the preprocessing module.

Run with:
    pytest tests/test_preprocessing.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from genomics_ml.features.preprocessing import (
    build_preprocessing_pipeline,
    impute_missing,
    scale_features,
)


# ── Fixture: sample data with NaN ────────────────────────────


@pytest.fixture
def sample_data_with_nan():
    """Return a DataFrame with a known missing value pattern."""
    return pd.DataFrame(
        {
            "gene_1": [1.0, 2.0, np.nan, 4.0],
            "gene_2": [np.nan, 5.0, 6.0, 7.0],
        }
    )


# ── Tests ────────────────────────────────────────────────────


def test_impute_missing_fills_nan(sample_data_with_nan):
    result = impute_missing(sample_data_with_nan, strategy="mean")
    assert result.isna().sum().sum() == 0
    assert result.shape == sample_data_with_nan.shape


def test_impute_missing_unknown_strategy(sample_data_with_nan):
    with pytest.raises(ValueError):
        impute_missing(sample_data_with_nan, strategy="invalid")


def test_scale_features_zero_mean_unit_var():
    X = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"])
    result = scale_features(X, scaler="standard")
    for col in result.columns:
        assert result[col].mean() == pytest.approx(0, abs=1e-10)
        # Use np.std (ddof=0) to match sklearn's StandardScaler
        assert np.std(result[col]) == pytest.approx(1, abs=1e-10)


def test_build_preprocessing_pipeline():
    pipeline = build_preprocessing_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == "imputer"
    assert pipeline.steps[1][0] == "scaler"
