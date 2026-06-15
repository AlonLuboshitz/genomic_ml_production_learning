#!/usr/bin/env python
"""
Evaluate a trained model on test data and print metrics.

Usage:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --model-path models/baseline.pkl
"""

import argparse
import sys

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from genomics_ml.data.load_data import load_data
from genomics_ml.models.model_utils import load_model, predict
from genomics_ml.utils.config import get_config_path, load_config
from genomics_ml.utils.logging import get_logger

logger = get_logger("scripts.evaluate_model")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model-path", default=None, help="Path to model .pkl file")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    args = parser.parse_args()

    # Load config
    config_path = args.config or get_config_path()
    config = load_config(config_path)

    # Determine model path
    model_path = args.model_path or config["model"]["path"]

    # Load model
    logger.info("Loading model from %s", model_path)
    model = load_model(model_path)

    # Load data and split
    X, y = load_data(config_path=config_path)
    test_size = config.get("data", {}).get("test_size", 0.2)
    random_state = config.get("data", {}).get("random_state", 42)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Test set: %d samples", len(X_test))

    # Predict and evaluate
    y_pred = predict(model, X_test.values if hasattr(X_test, "values") else X_test)
    report = classification_report(y_test, y_pred)
    print("\n=== Evaluation Report ===\n")
    print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
