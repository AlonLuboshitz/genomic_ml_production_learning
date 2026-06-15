"""
TODO: Tests for the config loader module.

Run with:
    pytest tests/test_config.py -v

Implementation steps for each test:
  1. Create test data inline (no external files needed).
  2. Call the function under test.
  3. Assert the expected behaviour.

Fixtures available:
  - ``sample_config_path`` — writes a temporary valid YAML and returns its path.
    The fixture cleans up the file after the test.
"""

import pytest
from genomics_ml.utils.config import load_config, get_config_path


# ── Fixture: temporary valid YAML file ───────────────────────


@pytest.fixture
def sample_config_path(tmp_path):
    """Create a minimal valid YAML config for testing.

    ``tmp_path`` is a pytest-provided fixture — a temporary directory
    that is automatically cleaned up after each test.
    """
    import yaml

    config = {"data": {"raw_path": "data.csv"}, "model": {"type": "RF"}}
    path = tmp_path / "test_config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return str(path)


# ── Tests ────────────────────────────────────────────────────


def test_load_config_returns_dict(sample_config_path):
    """TODO: Call load_config with the sample_config_path and assert the
    result is a dict (isinstance(result, dict))."""
    assert isinstance(load_config(sample_config_path), dict)


def test_load_config_missing_file():
    """TODO: Call load_config with a nonexistent path and use
    ``pytest.raises(FileNotFoundError)`` to verify it raises."""
    # YOUR CODE HERE
    with pytest.raises(FileNotFoundError):
        load_config("nopath")


def test_get_config_path_returns_string():
    """TODO: Call get_config_path() and assert the result is a string
    and ends with 'configs/default.yaml'."""
    assert isinstance(get_config_path(), str)
    assert get_config_path().endswith(".yaml")
