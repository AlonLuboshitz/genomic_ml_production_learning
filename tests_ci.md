# CI Test Configuration

## Why some tests are skipped in CI

CI runs `pytest tests/ -v --tb=short -m "not slow"` to skip tests marked `@pytest.mark.slow`.

### Skipped tests

- `test_flow_returns_metrics` — end-to-end Prefect pipeline that trains a real model
- `test_flow_model_override` — end-to-end pipeline with a different model type

### Why they're skipped

Both tests call `training_pipeline()`, which loads real data from `data/raw/genomics_data.csv`. This file is:
- Large (not committed to the repo)
- Environment-specific (may vary per developer or deployment)

The Docker container used in CI doesn't have this dataset, so these tests fail with `FileNotFoundError`.

### What runs in CI

The other 19 tests cover:
- **API endpoints** (`test_api.py`) — health, model-info, predict
- **Config loading** (`test_config.py`)
- **Promotion logic** (`test_pipeline.py`) — uses temp SQLite DBs, no real models
- **Preprocessing** (`test_preprocessing.py`) — imputation, scaling, pipeline building
- **Training utilities** (`test_training.py`) — model training with synthetic data

### How to run slow tests locally

```bash
# All tests
pytest tests/ -v

# Slow tests only
pytest tests/ -v -m "slow"

# Fast tests only (same as CI)
pytest tests/ -v -m "not slow"
```
