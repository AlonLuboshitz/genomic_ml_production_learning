# Genomics ML Pipeline

A production-style machine learning pipeline for genomic data classification.

## Quick Start

```bash
make install
make test
make train
make smoke-test
```

## Architecture

```
Data source → Validation → Preprocessing → MLflow tracking → Model artifact → FastAPI service
```

## Project Structure

```
├── configs/          # YAML configuration
├── data/             # Raw and processed data
├── models/           # Saved model artifacts
├── scripts/          # CLI entrypoints (train, evaluate, run-pipeline)
├── src/genomics_ml/  # Python package
│   ├── data/         # Loading and validation
│   ├── features/     # Preprocessing
│   ├── models/       # Training and prediction
│   ├── api/          # FastAPI service
│   ├── orchestration/ # Prefect flows
│   └── utils/        # Config and logging
└── tests/            # Pytest unit tests
```

## Commands

See `Makefile` for all available commands.

## Tech Stack

- Python 3.9+, scikit-learn, pandas, FastAPI, MLflow, Prefect, Docker, GitHub Actions
