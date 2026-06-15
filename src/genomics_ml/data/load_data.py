from typing import Optional, Tuple

import pandas as pd
from genomics_ml.utils.config import load_config, get_config_path
from genomics_ml.utils.logging import get_logger
from genomics_ml.data.validation import validate_dataframe


def load_data(
    config_path: Optional[str] = None,
    run_validation: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load genomic CSV, split into feature matrix X and target vector y.

    Parameters
    ----------
    config_path : str, optional
        Path to config YAML. Defaults to project config.
    run_validation : bool
        Whether to run data validation after loading (default True).

    Returns
    -------
    X : pd.DataFrame
        Feature columns (all except ``target``).
    y : pd.Series
        Target column.

    Raises
    ------
    FileNotFoundError
        If the CSV path from config does not exist.
    KeyError
        If the CSV is missing the ``target`` column.
    ValueError
        If validation is enabled and critical checks fail.
    """
    logger = get_logger("genomics_ml.data")

    if config_path is None:
        config_path = get_config_path()

    config = load_config(config_path)
    raw_path = config["data"]["raw_path"]

    logger.info("Loading data from %s", raw_path)
    df = pd.read_csv(raw_path)
    logger.info("Loaded %d rows, %d columns", *df.shape)

    # --- Validation ---
    if run_validation:
        report = validate_dataframe(df, config=config)
        for r in report["results"]:
            status = "PASS" if r["passed"] else "FAIL"
            logger.info("[%s] %s: %s", status, r["check"], r["message"])

        if not report["passed"]:
            logger.warning(
                "Validation: %d / %d checks FAILED — proceeding anyway",
                report["n_failed"],
                report["n_checks"],
            )

    # --- Extract target ---
    if "target" not in df.columns:
        raise KeyError("CSV is missing required 'target' column")

    y = df["target"]
    X = df.drop(columns=["target"])

    logger.info("Features: %s | Target: %s", X.shape, y.shape)
    logger.info("Target distribution:\n%s", y.value_counts().to_string())

    return X, y
