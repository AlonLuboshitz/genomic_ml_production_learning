#!/usr/bin/env python
"""
CLI entrypoint for training the baseline model.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --config configs/default.yaml
"""

import argparse
import json
import sys

from genomics_ml.data.load_data import load_data
from genomics_ml.models.train import train_model
from genomics_ml.utils.config import load_config, get_config_path
from genomics_ml.utils.logging import get_logger

logger = get_logger("scripts.train_model")


def main():
    parser = argparse.ArgumentParser(description="Train baseline ML model")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--model-type", default=None, help="Override model type")
    parser.add_argument(
        "--model-params", default=None, help="Override model params as JSON"
    )
    parser.add_argument(
        "--experiment-name", default=None, help="MLflow experiment name"
    )
    parser.add_argument("--run-name", default=None, help="Specific run name")
    args = parser.parse_args()

    config_path = args.config or get_config_path()
    config = load_config(config_path)

    if args.model_type:
        config["model"]["type"] = args.model_type
        logger.info("Overriding model type to %s", args.model_type)

    if args.model_params:
        config["model"]["params"] = json.loads(args.model_params)
        logger.info("Overriding model params to %s", config["model"]["params"])

    X, y = load_data(config_path=config_path)

    metrics, pipeline = train_model(
        X,
        y,
        config=config,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )

    print(metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
