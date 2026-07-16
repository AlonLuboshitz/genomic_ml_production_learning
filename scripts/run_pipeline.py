#!/usr/bin/env python
"""
CLI entry point to run the Prefect training pipeline.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --config configs/default.yaml
    python scripts/run_pipeline.py --model-type GradientBoostingClassifier
"""

import sys

from genomics_ml.orchestration.prefect_flow import training_pipeline
from genomics_ml.utils.cli import build_common_parser


def main():
    parser = build_common_parser("Run the Prefect training pipeline")
    args = parser.parse_args()
    metrics = training_pipeline(**vars(args))
    print(metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
