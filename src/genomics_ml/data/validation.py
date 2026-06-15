"""
Data validation checks for the genomics ML pipeline.

Runs configurable checks on a raw DataFrame before preprocessing:
  - Column existence and dtype checks (schema)
  - Missing-value fraction per column
  - Numeric range checks (e.g., no negative gene expression)
  - Target-class balance
  - Duplicate row detection

Each check returns a (passed: bool, message: str) pair.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from genomics_ml.utils.logging import get_logger

logger = get_logger("genomics_ml.data.validation")


def _check_schema(
    df: pd.DataFrame,
    schema: Optional[Dict[str, str]] = None,
) -> List[Tuple[bool, str]]:
    """Verify expected columns exist and have the correct dtype.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset.
    schema : dict or None
        Mapping of column name -> expected dtype (e.g. ``{"age": "float64"}``).
        If ``None``, the check is skipped.

    Returns
    -------
    list of (bool, str)
        One entry per column in the schema.
    """
    results: List[Tuple[bool, str]] = []
    if schema is None:
        results.append((True, "Schema check skipped (no schema defined in config)"))
        return results

    for col, expected_dtype in schema.items():
        if col not in df.columns:
            results.append((False, f"Missing column: '{col}'"))
            continue
        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            results.append(
                (
                    False,
                    f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}",
                )
            )
        else:
            results.append((True, f"Column '{col}' OK ({actual_dtype})"))
    return results


def _check_missing_values(
    df: pd.DataFrame,
    max_missing_frac: float = 0.5,
    columns: Optional[List[str]] = None,
) -> List[Tuple[bool, str]]:
    """Check that missing-value fraction is below a threshold.

    Parameters
    ----------
    df : pd.DataFrame
    max_missing_frac : float
        Maximum allowed fraction of missing values per column (default 0.5).
    columns : list of str or None
        Columns to check. If None, checks all columns.

    Returns
    -------
    list of (bool, str)
    """
    results: List[Tuple[bool, str]] = []
    if columns is None:
        columns = list(df.columns)

    for col in columns:
        missing_frac = df[col].isna().mean()
        if missing_frac > max_missing_frac:
            results.append(
                (
                    False,
                    f"Column '{col}' has {missing_frac:.1%} missing "
                    f"(threshold: {max_missing_frac:.0%})",
                )
            )
        else:
            results.append((True, f"Column '{col}' missing rate {missing_frac:.1%} OK"))
    return results


def _check_value_ranges(
    df: pd.DataFrame,
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[Tuple[bool, str]]:
    """Verify numeric columns stay within expected [low, high] bounds.

    Parameters
    ----------
    df : pd.DataFrame
    ranges : dict or None
        Mapping of column name -> (min, max). If None, the check is skipped.

    Returns
    -------
    list of (bool, str)
    """
    results: List[Tuple[bool, str]] = []
    if ranges is None:
        results.append((True, "Range check skipped (no ranges defined in config)"))
        return results

    for col, (lo, hi) in ranges.items():
        if col not in df.columns:
            results.append((False, f"Range check skipped: column '{col}' not found"))
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            results.append(
                (False, f"Column '{col}' is not numeric — cannot check range")
            )
            continue
        col_min = df[col].min()
        col_max = df[col].max()
        if col_min < lo or col_max > hi:
            results.append(
                (
                    False,
                    f"Column '{col}' out of range [{lo}, {hi}] "
                    f"(actual min={col_min}, max={col_max})",
                )
            )
        else:
            results.append((True, f"Column '{col}' in range [{lo}, {hi}] OK"))
    return results


def _check_target_balance(
    y: pd.Series,
    min_class_frac: float = 0.05,
) -> List[Tuple[bool, str]]:
    """Warn if any class has very few samples.

    Parameters
    ----------
    y : pd.Series
        Target series.
    min_class_frac : float
        Minimum fraction of total samples a class must have (default 0.05).

    Returns
    -------
    list of (bool, str)
    """
    results: List[Tuple[bool, str]] = []
    counts = y.value_counts(normalize=True)
    for cls, frac in counts.items():
        if frac < min_class_frac:
            results.append(
                (
                    False,
                    f"Class '{cls}' has only {frac:.1%} of samples "
                    f"(threshold: {min_class_frac:.0%})",
                )
            )
    if not results:
        results.append((True, "Target class balance: all classes above threshold OK"))
    return results


def _check_duplicates(df: pd.DataFrame) -> List[Tuple[bool, str]]:
    """Detect fully duplicate rows."""
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        return [(False, f"Found {n_dupes} duplicate row(s)")]
    return [(True, "No duplicate rows found")]


def validate_dataframe(
    df: pd.DataFrame,
    target_column: str = "target",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run all configured validation checks on a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset (including the target column).
    target_column : str
        Name of the target column (default ``"target"``).
    config : dict or None
        Validation configuration dictionary. Expected structure:

        .. code-block:: yaml

            validation:
              schema:
                gene_1: float64
                gene_2: float64
                target: int64
              max_missing_frac: 0.5
              value_ranges:
                gene_1: [0.0, 100.0]
              min_class_frac: 0.05
              check_duplicates: true

        If ``None`` or missing keys, sensible defaults are used.

    Returns
    -------
    dict with keys:
        - ``passed`` (bool): True only if every check passed.
        - ``n_checks`` (int): total number of individual checks run.
        - ``n_failed`` (int): number of checks that failed.
        - ``results`` (list of dict): each entry has ``check``, ``passed``,
          ``message``.
    """
    val_config = (config or {}).get("validation", {})

    # Separate checks are accumulated into *all_results*
    all_results: List[Dict[str, Any]] = []

    # 1. Schema
    schema = val_config.get("schema")
    for passed, msg in _check_schema(df, schema=schema):
        all_results.append({"check": "schema", "passed": passed, "message": msg})

    # 2. Missing values
    max_missing = val_config.get("max_missing_frac", 0.5)
    for passed, msg in _check_missing_values(df, max_missing_frac=max_missing):
        all_results.append(
            {"check": "missing_values", "passed": passed, "message": msg}
        )

    # 3. Value ranges
    ranges = val_config.get("value_ranges")
    for passed, msg in _check_value_ranges(df, ranges=ranges):
        all_results.append({"check": "value_ranges", "passed": passed, "message": msg})

    # 4. Target balance
    if target_column in df.columns:
        min_class_frac = val_config.get("min_class_frac", 0.05)
        for passed, msg in _check_target_balance(
            df[target_column], min_class_frac=min_class_frac
        ):
            all_results.append(
                {"check": "target_balance", "passed": passed, "message": msg}
            )
    else:
        all_results.append(
            {
                "check": "target_balance",
                "passed": False,
                "message": f"Target column '{target_column}' not found in DataFrame",
            }
        )

    # 5. Duplicates
    if val_config.get("check_duplicates", True):
        for passed, msg in _check_duplicates(df):
            all_results.append(
                {"check": "duplicates", "passed": passed, "message": msg}
            )

    n_failed = sum(1 for r in all_results if not r["passed"])
    n_passed = len(all_results) - n_failed

    logger.info(
        "Validation complete: %d passed, %d failed out of %d checks",
        n_passed,
        n_failed,
        len(all_results),
    )

    return {
        "passed": n_failed == 0,
        "n_checks": len(all_results),
        "n_failed": n_failed,
        "results": all_results,
    }
