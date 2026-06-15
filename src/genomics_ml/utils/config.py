"""Configuration loader — reads YAML config files and returns a dict."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return it as a dictionary.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Dictionary with configuration values.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with open(config_path) as f:
        configs = yaml.safe_load(f)

    return configs


def get_config_path(env_var: str = "GENOMICS_ML_CONFIG") -> str:
    """
    Return the config path from an environment variable, or default to
    'configs/default.yaml' relative to the project root.

    The project root is detected by walking up from this file's location
    until a 'pyproject.toml' is found.
    """
    env_path = os.environ.get(env_var)
    if env_path:
        return env_path

    # Walk up from this file -> utils -> src -> genomics_ml -> ... -> root
    here = Path(__file__).resolve().parent
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return str(parent / "configs" / "default.yaml")

    # Fallback: assume running from project root
    return "configs/default.yaml"

