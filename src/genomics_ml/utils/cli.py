"""Shared CLI argument parser and config override logic for pipeline scripts."""

import argparse
import json
from typing import Any, Dict


def build_common_parser(description: str) -> argparse.ArgumentParser:
    """Return an ArgumentParser with arguments shared across all pipeline scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", dest="config_path", default=None, help="Path to config YAML")
    parser.add_argument("--model-type", default=None, help="Override model type")
    parser.add_argument(
        "--model-params", default=None, help="Override model params as JSON"
    )
    parser.add_argument(
        "--experiment-name", default=None, help="MLflow experiment name"
    )
    parser.add_argument("--run-name", default=None, help="Specific run name")
    return parser


def override_config_from_args(
    config: Dict[str, Any],
    args,
) -> Dict[str, Any]:
    """Apply CLI argument overrides to a config dict (mutates and returns it).

    Accepts an argparse Namespace or any object with ``.model_type``
    and ``.model_params`` attributes (or ``None`` for both).
    """
    return override_config(
        config,
        model_type=getattr(args, "model_type", None),
        model_params=getattr(args, "model_params", None),
    )


# Default params per model type (used when switching types without providing new params)
_DEFAULT_PARAMS = {
    "RandomForestClassifier": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
    "LogisticRegression": {"C": 1.0, "max_iter": 1000, "random_state": 42},
    "GradientBoostingClassifier": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
}


def override_config(
    config: Dict[str, Any],
    model_type: str = None,
    model_params: str = None,
) -> Dict[str, Any]:
    """Apply model overrides to a config dict (mutates and returns it).

    This lower-level function works with raw values so it can be called
    from both CLI scripts and the Prefect flow.

    If ``model_type`` changes but no ``model_params`` are given, sensible
    defaults for the new model type are used instead of the old model's params.
    """
    if model_type:
        config["model"]["type"] = model_type
    if model_params:
        config["model"]["params"] = json.loads(model_params)
    elif model_type:
        # Type changed but no params given — use defaults for the new type
        config["model"]["params"] = _DEFAULT_PARAMS.get(model_type, {})
    return config


def train_kwargs_from_args(args) -> Dict[str, Any]:
    """Return extra keyword arguments for train_model() from parsed args."""
    kwargs = {}
    if getattr(args, "experiment_name", None):
        kwargs["experiment_name"] = args.experiment_name
    if getattr(args, "run_name", None):
        kwargs["run_name"] = args.run_name
    return kwargs
