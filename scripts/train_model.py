#!/usr/bin/env python
"""
CLI entrypoint for training the baseline model.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --config configs/default.yaml
"""

import sys

from genomics_ml.data.load_data import load_data
from genomics_ml.models.train import train_model
from genomics_ml.utils.cli import (
    build_common_parser,
    override_config_from_args,
    train_kwargs_from_args,
)
from genomics_ml.utils.config import load_config, get_config_path
from genomics_ml.utils.logging import get_logger

logger = get_logger("scripts.train_model")


def main():
    parser = build_common_parser("Train baseline ML model")
    args = parser.parse_args()

    config_path = args.config_path or get_config_path()
    config = load_config(config_path)
    config = override_config_from_args(config, args)

    X, y = load_data(config=config)

    kwargs = train_kwargs_from_args(args)
    metrics, pipeline = train_model(X, y, config=config, **kwargs)

    print(metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
