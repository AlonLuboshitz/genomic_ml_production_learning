"""
Prefect orchestration flow for the genomics ML pipeline.

Usage:
    python -m genomics_ml.orchestration.prefect_flow
"""

from typing import Optional

from prefect import flow, task

from genomics_ml.data.load_data import load_data
from genomics_ml.models.promote import promote_if_better
from genomics_ml.models.train import train_model
from genomics_ml.utils.cli import override_config
from genomics_ml.utils.config import load_config, get_config_path
from genomics_ml.utils.logging import get_logger

logger = get_logger("genomics_ml.orchestration")


@task(retries=2, retry_delay_seconds=5)
def load_data_task(config: dict):
    """Load and validate the dataset (retries up to 2 times on failure)."""
    X, y = load_data(config=config)
    logger.info("Loaded %d samples with %d features", X.shape[0], X.shape[1])
    return X, y


@task
def train_model_task(
    X, y, config: dict, experiment_name: str = None, run_name: str = None
):
    """Train, evaluate, save model, log to MLflow + SQLite."""
    kwargs = {}
    if experiment_name:
        kwargs["experiment_name"] = experiment_name
    if run_name:
        kwargs["run_name"] = run_name
    metrics, pipeline = train_model(X, y, config=config, **kwargs)
    logger.info("Training complete - accuracy: %.4f", metrics["accuracy"])
    return metrics


@flow(log_prints=True)
def training_pipeline(
    config_path: Optional[str] = None,
    model_type: Optional[str] = None,
    model_params: Optional[str] = None,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
):
    """Orchestrate the full ML pipeline: load data -> train -> evaluate -> log."""
    if config_path is None:
        config_path = get_config_path()

    config = load_config(config_path)
    print(f"Config loaded from {config_path}")

    # Apply CLI overrides to config (reuses shared override_config)
    config = override_config(config, model_type=model_type, model_params=model_params)

    # Run pipeline steps
    X, y = load_data_task(config)
    metrics = train_model_task(
        X,
        y,
        config=config,
        experiment_name=experiment_name,
        run_name=run_name,
    )

    # Model promotion — check if new model beats champion
    promoted = promote_if_better(
        db_path="ml_metadata.db",
        new_model_path=metrics["model_path"],
        new_accuracy=metrics["accuracy"],
        new_run_id=metrics["run_id"],
    )
    print(f"Model promoted to champion: {promoted}")

    print(f"Pipeline complete - accuracy: {metrics['accuracy']:.4f}")
    print(f"  Model saved to: {metrics['model_path']}")

    return metrics


if __name__ == "__main__":
    from genomics_ml.utils.cli import build_common_parser

    parser = build_common_parser("Run the Prefect training pipeline")
    args = parser.parse_args()
    result = training_pipeline(**vars(args))
    print(f"Result: {result}")
