"""
Preprocessing pipeline for the genomics ML pipeline.

Provides:
  - impute_missing: fill NaN values via sklearn SimpleImputer
  - scale_features: scale numeric features via sklearn StandardScaler / etc.
  - build_preprocessing_pipeline: compose both into a single Pipeline
"""

from typing import Any, Dict, Optional

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from genomics_ml.utils.logging import get_logger

logger = get_logger("genomics_ml.features")

IMPUTERS = {
    "mean": SimpleImputer(strategy="mean"),
    "median": SimpleImputer(strategy="median"),
    "most_frequent": SimpleImputer(strategy="most_frequent"),
}

SCALERS = {
    "standard": StandardScaler(),
    "minmax": "minmax",  # placeholder — not imported to keep deps light
    "robust": RobustScaler(),
}


def impute_missing(
    X: pd.DataFrame,
    strategy: str = "mean",
) -> pd.DataFrame:
    """Fill missing values in the feature matrix.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (may contain NaN).
    strategy : str
        One of ``"mean"``, ``"median"``, ``"most_frequent"``.

    Returns
    -------
    pd.DataFrame with same columns/index, no NaN.
    """
    logger.info("Imputing missing values (strategy=%s)", strategy)
    imputer = IMPUTERS.get(strategy)
    if imputer is None:
        raise ValueError(
            f"Unknown impute strategy: {strategy}. Choose from {list(IMPUTERS)}"
        )

    X_imp = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    n_imputed = X.isna().sum().sum()
    logger.info("Imputed %d missing value(s)", n_imputed)
    return X_imp


def scale_features(
    X: pd.DataFrame,
    scaler: str = "standard",
) -> pd.DataFrame:
    """Scale numeric features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (no NaN expected).
    scaler : str
        One of ``"standard"``, ``"robust"``.

    Returns
    -------
    pd.DataFrame with scaled values, same shape.
    """
    logger.info("Scaling features (scaler=%s)", scaler)

    if scaler == "minmax":
        logger.warning("MinMaxScaler not available — falling back to StandardScaler")
        scaler = "standard"

    scaler_obj = SCALERS.get(scaler)
    if scaler_obj is None:
        raise ValueError(f"Unknown scaler: {scaler}. Choose from {list(SCALERS)}")

    X_scaled = pd.DataFrame(
        scaler_obj.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )
    logger.info(
        "Scaling done — mean ~%.3f, std ~%.3f",
        X_scaled.values.mean(),
        X_scaled.values.std(),
    )
    return X_scaled


def build_preprocessing_pipeline(
    config: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Build a sklearn Pipeline that imputes then scales.

    Parameters
    ----------
    config : dict or None
        Should contain a ``preprocessing`` key with ``impute_strategy`` and
        ``scaler``. Falls back to defaults if missing.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    if config is None:
        config = {}

    preproc_cfg = config.get("preprocessing", {})
    impute_strategy = preproc_cfg.get("impute_strategy", "mean")
    scaler_name = preproc_cfg.get("scaler", "standard")

    logger.info(
        "Building preprocessing pipeline (impute=%s, scaler=%s)",
        impute_strategy,
        scaler_name,
    )

    imputer = IMPUTERS.get(impute_strategy)
    if imputer is None:
        raise ValueError(f"Unknown impute strategy: {impute_strategy}")

    if scaler_name == "minmax":
        logger.warning("MinMaxScaler not available — using StandardScaler")
        scaler_name = "standard"

    scaler_obj = SCALERS.get(scaler_name)
    if scaler_obj is None:
        raise ValueError(f"Unknown scaler: {scaler_name}")

    pipeline = Pipeline(
        [
            ("imputer", imputer),
            ("scaler", scaler_obj),
        ]
    )

    logger.info("Preprocessing pipeline ready: %s", pipeline)

    return pipeline
