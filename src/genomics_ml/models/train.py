"""
TODO: Training module for the genomics ML pipeline.

Your task:
  1. Implement ``train_model()`` that:
     - Splits X, y into train/test using ``train_test_split`` from sklearn
       (use test_size=0.2 and random_state=42 from config["data"])
     - Builds a Pipeline with steps: imputer, scaler, classifier
       (use ``build_preprocessing_pipeline()`` from
        ``genomics_ml.features.preprocessing`` for the first steps)
     - Instantiates the classifier from config (see ``_get_classifier`` below)
     - Fits the pipeline on X_train, y_train
     - Prints a classification_report on X_test, y_test
     - Saves the trained pipeline with ``joblib.dump`` to the path in config
     - Returns (metrics_dict, trained_pipeline)

  2. Implement ``_get_classifier()`` — returns an sklearn classifier instance
     based on config["model"]["type"] and config["model"]["params"].

Supported model types (add your own):
  - RandomForestClassifier
  - LogisticRegression
  - GradientBoostingClassifier

Run the training with:
    python scripts/train_model.py
"""

from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from genomics_ml.features.preprocessing import build_preprocessing_pipeline
from genomics_ml.models.model_utils import save_model
import mlflow
from genomics_ml.utils.config import load_config, get_config_path
from genomics_ml.utils.logging import get_logger
from genomics_ml.utils.database import (
    close_connection,
    init_database,
    insert_metrics,
    insert_params,
    insert_predictions,
    insert_run,
)

logger = get_logger("genomics_ml.models.train")


def _get_classifier(config: Dict[str, Any]):
    """TODO: Instantiate the classifier class from config['model'].

    Steps:
      1. Get model_type and params from config["model"].
      2. Map model_type string to an sklearn class (e.g.
         "RandomForestClassifier" -> RandomForestClassifier).
      3. Instantiate the class with **params.
      4. Return the instance.

    Raises ValueError if model_type is unknown.
    """
    model_type, model_params = _get_model_configs(config)
    model_types = {
        "RandomForestClassifier": RandomForestClassifier,
        "LogisticRegression": LogisticRegression,
        "GradientBoostingClassifier": GradientBoostingClassifier,
    }

    model_class = model_types.get(model_type)
    if model_class is None:
        raise ValueError(
            f"Unknown model type '{model_type}'. Choose from {list(model_types)}"
        )
    model = model_class(**model_params)
    return model


def _get_model_configs(config):
    model_configs = config.get("model")
    model_type = model_configs.get("type")
    if model_type is None:
        raise ValueError("None model type")
    model_params = model_configs.get("params")
    # if None then raise and defualts.
    return model_type, model_params


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[Dict[str, Any]] = None,
    model_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Tuple[Dict[str, Any], Any]:
    """Train a classifier, log to MLflow + SQLite, return metrics and pipeline."""
    if config is None:
        config = load_config(get_config_path())

    # ── MLflow setup ──────────────────────────────────────────
    mlflow.autolog()
    mlflow.set_experiment(experiment_name or "genomic_baseline")

    # ── Split ────────────────────────────────────────────────
    test_size = config.get("data", {}).get("test_size", 0.2)
    random_state = config.get("data", {}).get("random_state", 42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        "Split: train=%d, test=%d (test_size=%.2f)",
        len(X_train),
        len(X_test),
        test_size,
    )

    # ── Build pipeline ────────────────────────────────────────
    pipeline = build_preprocessing_pipeline(config)
    model = _get_classifier(config)
    pipeline.steps.append(("clf", model))
    logger.info("Pipeline: %s", pipeline)

    # ── Train (inside MLflow run) ─────────────────────────────
    with mlflow.start_run(run_name=run_name):
        logger.info("Training %s ...", type(model).__name__)
        pipeline.fit(X_train, y_train)
        logger.info("Training complete.")

        # ── Evaluate ──────────────────────────────────────────
        y_pred = pipeline.predict(X_test)
        report_text = classification_report(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        accuracy = report_dict.pop("accuracy")
        mlflow.log_metric("accuracy", accuracy)
        logger.info("Test accuracy: %.4f", accuracy)

        # ── Save model ────────────────────────────────────────
        if model_path is None:
            model_type, _ = _get_model_configs(config)
            from datetime import datetime

            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"models/{model_type}_{date_str}.pkl"

        save_model(pipeline, model_path)
        logger.info("Model saved to %s", model_path)

    # ── SQLite logging (after MLflow run closes) ──────────────
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    db_connection = init_database("ml_metadata.db")
    run_id = insert_run(
        db_connection,
        model_name=config["model"]["type"],
        run_name=run_name or "unnamed",
        dataset_path=config["data"]["raw_path"],
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        training_rows=len(X_train),
        testing_rows=len(X_test),
        num_features=X.shape[1],
    )

    insert_params(db_connection, run_id, config["model"]["params"])
    insert_metrics(db_connection, run_id, {"accuracy": accuracy})
    insert_predictions(db_connection, run_id, "test", y_pred, y_test, probabilities)
    close_connection(db_connection, model_path, run_id)

    # ── Build metrics dict ────────────────────────────────────
    metrics = {
        "accuracy": accuracy,
        "classification_report": report_text,
        "model_path": model_path,
    }
    return metrics, pipeline
