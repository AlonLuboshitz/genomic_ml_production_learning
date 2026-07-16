# Orchestration — Prefect pipeline

<!--
  What this file covers:
  - Why we use Prefect
  - How the pipeline is structured (tasks + flow)
  - How to run it
  - How to view runs in Prefect Cloud
-->

## Why Prefect

The training pipeline has multiple steps: load data, validate, preprocess,
train, evaluate, log to MLflow + SQLite, and promote the best model.

Prefect orchestrates these steps by tracking each task's state, inputs,
and outputs. If a step fails, you see exactly where — and you can add
automatic retries.

## Pipeline structure

**`src/genomics_ml/orchestration/prefect_flow.py`**

| Component | Role |
|---|---|
| `load_data_task` | `@task` — loads CSV, validates schema, returns X, y |
| `train_model_task` | `@task` — splits, preprocesses, trains, evaluates, logs |
| `training_pipeline` | `@flow` — runs tasks in order, applies overrides, calls promotion |

**`scripts/run_pipeline.py`** — CLI entry point that accepts all the same
arguments as `scripts/train_model.py`.

## How to run

```bash
# Default
make run-pipeline            # or: python scripts/run_pipeline.py

# With overrides
python scripts/run_pipeline.py --model-type GradientBoostingClassifier
python scripts/run_pipeline.py --model-params '{"n_estimators": 200}'
python scripts/run_pipeline.py --experiment-name "my_study" --run-name "v2"
```

## Viewing results

Flow runs are logged to **Prefect Cloud**. Open your workspace at:

https://app.prefect.cloud

Each run shows:
- Flow run status (Completed / Failed / Running)
- Per-task duration and state
- Input parameters and return values
- Log output from each step

## Model promotion

After training, the flow calls `promote_if_better()` which compares the
new model's accuracy against the current champion (stored in the SQLite DB).
If better, it copies the model to `models/champion.pkl` and marks the run
as champion.
